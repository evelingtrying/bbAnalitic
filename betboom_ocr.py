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
    time_match = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", lowered)
    time_value = None
    if time_match:
        time_value = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

    if "сегодня" in lowered or "today" in lowered:
        date_value = datetime.now().date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month
    if "вчера" in lowered or "yesterday" in lowered:
        date_value = (datetime.now() - timedelta(days=1)).date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month

    date_match = re.search(
        r"\b(\d{1,2})\s+([a-zа-яё]+)\s+(?:\d{1,2}[:.]\d{2})\b",
        lowered,
    )
    if not date_match:
        numeric_match = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](20\d{2}))?\b", lowered)
        if numeric_match:
            day = int(numeric_match.group(1))
            month = int(numeric_match.group(2))
            parsed_year = int(numeric_match.group(3)) if numeric_match.group(3) else year
            try:
                date_value = datetime(parsed_year, month, day)
                return date_value.strftime("%d.%m.%Y"), time_value, month
            except ValueError:
                return None, time_value, None
        return None, time_value, None

    day = int(date_match.group(1))
    month = _normalize_month(date_match.group(2))
    if month is None:
        return None, time_value, None
    try:
        date_value = datetime(year, month, day)
    except ValueError:
        return None, time_value, month
    return date_value.strftime("%d.%m.%Y"), time_value, month

def _parse_signed_amount(text: str) -> float | None:
    normalized = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("\n", " ")
    )
    # Ищем знак + или -, затем цифры, опционально копейки. Игнорируем знак рубля.
    match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", normalized)
    if not match:
        return None
    try:
        number = float(match.group(2).replace(",", "."))
    except ValueError:
        return None
    return number if match.group(1) == "+" else -number


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
    
    # 🪄 УМНОЕ МАСШТАБИРОВАНИЕ: Если картинка слишком маленькая, увеличиваем её до нормального размера
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

    # 🪄 СКЛЕИВАЕМ БЛОКИ В ЛОГИЧЕСКИЕ СТРОКИ
    lines = []
    current_line = []
    last_y = None

    for b in blocks:
        if last_y is None or abs(b["y"] - last_y) < b["h"] * 0.8:
            current_line.append(b)
            last_y = sum(item["y"] for item in current_line) / len(current_line)
        else:
            lines.append(current_line)
            current_line = [b]
            last_y = b["y"]
    if current_line:
        lines.append(current_line)

    parsed_cards = []
    
    # 🪄 ИЩЕМ "ПРОГНОЗ" И БЕРЕМ СТРОКУ ПОД НИМ
    for i, line_blocks in enumerate(lines):
        line_text = " ".join(b["text"] for b in line_blocks)
        lowered = line_text.lower().replace("ё", "е")
        
        if "прогноз" in lowered or "транзакц" in lowered:
            card_type = "transaction" if "транзакц" in lowered else "forecast"
            
            # Берем текст текущей строки и ОБЯЗАТЕЛЬНО следующей (где дата и сумма)
            context_text = line_text
            if i + 1 < len(lines):
                next_line_text = " ".join(b["text"] for b in lines[i+1])
                context_text += " " + next_line_text
            
            date_value, time_value, month_number = _parse_date_time(context_text, datetime.now().year)
            amount = _parse_signed_amount(context_text)

            parsed_cards.append({
                "type": card_type,
                "text": context_text,
                "amount": amount,
                "date": date_value,
                "time": time_value,
            })

    result["diagnostics"]["detected_cards"] = len(parsed_cards)

    for card in parsed_cards:
        if card["type"] == "transaction":
            result["diagnostics"]["ignored_transactions"] += 1
            continue

        if card["date"] is None or card["time"] is None or card["amount"] is None:
            result["diagnostics"]["skipped_cards"] += 1
            continue

        outcome = "Выигрыш" if card["amount"] > 0 else "Проигрыш" if card["amount"] < 0 else "Возврат"
        result["bets"].append({
            "Дата": card["date"],
            "Время": card["time"],
            "Результат": outcome,
            "Сумма_изменения": float(card["amount"]),
        })

    result["diagnostics"]["recognized_bets"] = len(result["bets"])
    return result
