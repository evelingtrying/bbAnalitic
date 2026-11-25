"""OCR parser for BetBoom balance-history screenshots using RapidOCR."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Callable, Any

import cv2
import numpy as np
from rapidocr import RapidOCR

_ocr_engine: RapidOCR | None = None

def _get_engine() -> RapidOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine

MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5,
    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "сент": 9,
    "окт": 10, "ноя": 11, "дек": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

def _normalize_month(month_text: str) -> int | None:
    value = month_text.lower().strip()
    if re.search(r"[а-яё]", value):
        value = value.replace("c", "с").replace("e", "е").replace("a", "а")
    for prefix, month_number in MONTHS.items():
        if value.startswith(prefix):
            return month_number
    return None

def _parse_date_time(text: str, year: int) -> tuple[str | None, str | None, int | None]:
    lowered = text.lower().replace("ё", "е")
    
    # Ищем время в формате 12:34, 12.34 или 12,34
    time_match = re.search(r"\b([01]?\d|2[0-3])[:.,]\s*([0-5]\d)\b", lowered)
    time_value = None
    if time_match:
        time_value = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

    # Обработка относительных дат
    if "сегодня" in lowered or "today" in lowered:
        date_value = datetime.now().date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month
    if "вчера" in lowered or "yesterday" in lowered:
        date_value = (datetime.now() - timedelta(days=1)).date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month

    # Ищем день и месяц (например, "22 ноября")
    date_match = re.search(r"\b(\d{1,2})\s+([a-zа-яё]+)", lowered)
    if date_match:
        day = int(date_match.group(1))
        month = _normalize_month(date_match.group(2))
        if month:
            try:
                date_value = datetime(year, month, day)
                return date_value.strftime("%d.%m.%Y"), time_value, month
            except ValueError:
                return None, time_value, None
                
    return None, time_value, None


def process_image_bytes(
    file_name: str,
    file_bytes: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bets": [],
        "diagnostics": {
            "file": file_name,
            "width": 0,
            "height": 0,
            "detected_cards": 0,
            "recognized_bets": 0,
            "ignored_transactions": 0,
            "skipped_cards": 0,
            "reason": None,
            "raw_text": ""
        },
    }

    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        result["diagnostics"]["reason"] = "Не удалось открыть изображение."
        return result

    height, width = image.shape[:2]
    
    # 🪄 УМНОЕ МАСШТАБИРОВАНИЕ: Если картинка мелкая, увеличиваем её, чтобы нейросеть видела текст
    if width < 1000:
        scale = 1000 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]

    result["diagnostics"]["width"] = width
    result["diagnostics"]["height"] = height

    try:
        ocr = _get_engine()
        ocr_output = ocr(image)
    except Exception as exc:
        result["diagnostics"]["reason"] = f"Ошибка при запуске RapidOCR: {exc}"
        return result

    if not ocr_output or not getattr(ocr_output, 'txts', None):
        result["diagnostics"]["reason"] = "Не удалось найти текст на изображении."
        return result

    boxes = ocr_output.boxes
    txts = ocr_output.txts
    
    result["diagnostics"]["raw_text"] = "\n".join(txts)

    # Извлекаем блоки текста
    blocks = []
    for i in range(len(txts)):
        text = txts[i]
        box = boxes[i]
        x_center = float(np.mean(box[:, 0]))
        y_center = float(np.mean(box[:, 1]))
        y_min = float(np.min(box[:, 1]))
        y_max = float(np.max(box[:, 1]))
        h = y_max - y_min
        
        blocks.append({
            "text": text,
            "x": x_center,
            "y": y_center,
            "h": h,
        })

    # Сортируем блоки сверху вниз, затем слева направо
    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # 🪄 ГРУППИРУЕМ БЛОКИ В ВИЗУАЛЬНЫЕ СТРОКИ
    lines = []
    current_line = []
    last_y = None

    for b in blocks:
        if last_y is None or abs(b["y"] - last_y) < b["h"] * 1.5:
            current_line.append(b)
            last_y = sum(item["y"] for item in current_line) / len(current_line)
        else:
            current_line.sort(key=lambda x: x["x"])
            lines.append(current_line)
            current_line = [b]
            last_y = b["y"]
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(current_line)

    parsed_bets = []

    # 🪄 НОВАЯ ЛОГИКА: Ищем строки с суммами, а контекст (дату/время) берем из соседних строк
    for i, line in enumerate(lines):
        line_text = " ".join([w['text'] for w in line])
        
        # Ищем любую сумму (с плюсом или минусом)
        amount_match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", line_text)
        if not amount_match:
            continue
            
        sign = amount_match.group(1)
        num_str = amount_match.group(2).replace(",", ".")
        try:
            amount = float(num_str)
            if sign == "-":
                amount = -amount
        except ValueError:
            continue
            
        # Собираем контекст (текущая строка + предыдущая + следующая)
        context = line_text
        if i > 0:
            context = " ".join([w['text'] for w in lines[i-1]]) + " " + context
        if i + 1 < len(lines):
            context = context + " " + " ".join([w['text'] for w in lines[i+1]])
            
        # Если в контексте есть слово "Транзакция" — пропускаем (это не ставка)
        if "транзакц" in context.lower() or "транзакц" in line_text.lower():
            result["diagnostics"]["ignored_transactions"] += 1
            continue
            
        date_value, time_value, _ = _parse_date_time(context, datetime.now().year)
        
        if date_value and time_value:
            parsed_bets.append({
                "Дата": date_value,
                "Время": time_value,
                "Результат": "Выигрыш" if amount > 0 else "Проигрыш",
                "Сумма_изменения": amount
            })
        else:
            result["diagnostics"]["skipped_cards"] += 1

    result["diagnostics"]["detected_cards"] = len(parsed_bets) + result["diagnostics"]["ignored_transactions"] + result["diagnostics"]["skipped_cards"]
    result["bets"] = parsed_bets
    result["diagnostics"]["recognized_bets"] = len(parsed_bets)
    
    if not parsed_bets:
        result["diagnostics"]["reason"] = "Текст прочитан, но не найдено ни одной ставки с датой и суммой."
        
    return result
