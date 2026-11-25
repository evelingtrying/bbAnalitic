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
    result["diagnostics"]["width"] = width
    result["diagnostics"]["height"] = height

    if width < 700:
        result["diagnostics"]["reason"] = (
            f"Слишком низкое разрешение ({width} px по ширине). "
            "Загрузите оригинальный скриншот."
        )
        return result

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

    # 🪄 Извлекаем все блоки текста с их координатами
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
            "used": False
        })

    parsed_cards = []
    
    # 🪄 Ищем "якоря" (Прогноз или Транзакция) и собираем блоки вокруг них
    for i, block in enumerate(blocks):
        if block["used"]:
            continue

        lowered = block["text"].lower().replace("ё", "е")
        if "прогноз" in lowered or "транзакц" in lowered:
            card_type = "transaction" if "транзакц" in lowered else "forecast"
            
            # Находим все блоки на этой же горизонтальной линии
            line_blocks = [block]
            block["used"] = True
            
            for j, other in enumerate(blocks):
                if j == i or other["used"]:
                    continue
                # Если блок находится примерно на той же высоте (допуск = 1.5 высоты текста)
                if abs(other["y"] - block["y"]) < block["h"] * 1.5:
                    line_blocks.append(other)
                    other["used"] = True
            
            # Сортируем блоки слева направо
            line_blocks.sort(key=lambda b: b["x"])
            full_line_text = " ".join(b["text"] for b in line_blocks)
            
            # Парсим дату, время и сумму из собранной строки
            date_value, time_value, month_number = _parse_date_time(full_line_text, datetime.now().year)
            amount_match = re.search(r"([+-]\s*\d+(?:[.,]\d{1,2})?)", full_line_text)
            amount = _parse_signed_amount(amount_match.group(1)) if amount_match else None

            parsed_cards.append({
                "type": card_type,
                "text": full_line_text,
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
