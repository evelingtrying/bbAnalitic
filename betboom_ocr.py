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
    for prefix, month_number in MONTHS.items():
        if value.startswith(prefix):
            return month_number
    return None

def _parse_date_time(text: str, year: int) -> tuple[str | None, str | None, int | None]:
    lowered = text.lower().replace("ё", "е")
    
    # 🪄 Нормализация латинских букв в кириллицу (частая ошибка OCR)
    trans = str.maketrans('ceoapxykmthb', 'сеоархукмтнв')
    lowered = lowered.translate(trans)
    
    time_match = re.search(r"([01]?\d|2[0-3])[:.,-]\s*([0-5]\d)", lowered)
    time_value = None
    if time_match:
        time_value = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

    if "сегодня" in lowered or "today" in lowered:
        date_value = datetime.now().date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month
    if "вчера" in lowered or "yesterday" in lowered:
        date_value = (datetime.now() - timedelta(days=1)).date()
        return date_value.strftime("%d.%m.%Y"), time_value, date_value.month

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


def _parse_signed_amount(text: str) -> float | None:
    normalized = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("\n", " ")
    )
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
            "raw_text": "",
            "failed_cards": []
        },
    }

    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        result["diagnostics"]["reason"] = "Не удалось открыть изображение."
        return result

    height, width = image.shape[:2]
    
    # 🪄 Если картинка мелкая, растягиваем её до нормального размера
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

    # Извлекаем блоки текста с координатами
    blocks = []
    for i in range(len(txts)):
        text = txts[i]
        box = boxes[i]
        x_center = float(np.mean(box[:, 0]))
        y_center = float(np.mean(box[:, 1]))
        y_min = float(np.min(box[:, 1]))
        y_max = float(np.max(box[:, 1]))
        blocks.append({
            "text": text,
            "x": x_center,
            "y": y_center,
            "h": y_max - y_min,
        })

    # Сортируем блоки сверху вниз
    blocks.sort(key=lambda b: b["y"])

    # 🪄 Ищем "якоря" (Прогноз или Транзакция)
    anchors = []
    for b in blocks:
        text_lower = b["text"].lower().replace("ё", "е")
        # Чиним частые ошибки OCR
        text_lower = text_lower.replace("пpогноз", "прогноз").replace("tранзакция", "транзакция")
        
        if "прогноз" in text_lower or "транзакц" in text_lower:
            is_transaction = "транзакц" in text_lower
            anchors.append({
                "y": b["y"],
                "is_transaction": is_transaction
            })

    anchors.sort(key=lambda x: x["y"])

    parsed_bets = []
    
    # 🪄 Разбиваем картинку на карточки (от одного якоря до другого)
    for idx, anchor in enumerate(anchors):
        y_start = anchor["y"]
        y_end = anchors[idx+1]["y"] if idx + 1 < len(anchors) else float('inf')
        
        if anchor["is_transaction"]:
            result["diagnostics"]["ignored_transactions"] += 1
            continue
            
        # Собираем весь текст внутри этой карточки
        card_blocks = [b for b in blocks if y_start <= b["y"] < y_end]
        card_text = " ".join(b["text"] for b in card_blocks)
        
        date_value, time_value, _ = _parse_date_time(card_text, datetime.now().year)
        amount = _parse_signed_amount(card_text)
        
        if date_value and time_value and amount is not None:
            parsed_bets.append({
                "Дата": date_value,
                "Время": time_value,
                "Результат": "Выигрыш" if amount > 0 else "Проигрыш",
                "Сумма_изменения": amount
            })
        else:
            result["diagnostics"]["skipped_cards"] += 1
            result["diagnostics"]["failed_cards"].append(card_text)

    result["diagnostics"]["detected_cards"] = len(anchors)
    result["bets"] = parsed_bets
    result["diagnostics"]["recognized_bets"] = len(parsed_bets)
    
    if not parsed_bets and not anchors:
        result["diagnostics"]["reason"] = "Не найдено ни одной карточки ставки (нет слов 'Прогноз' или 'Транзакция')."
    elif not parsed_bets:
        result["diagnostics"]["reason"] = "Карточки найдены, но не удалось извлечь дату, время или сумму."

    return result
