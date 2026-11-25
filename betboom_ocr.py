"""OCR parser for BetBoom balance-history screenshots using RapidOCR.

RapidOCR is an ONNX-based engine that provides accurate bounding boxes for
each text line, preventing close cards from being merged together.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Callable, Any

import cv2
import numpy as np

# 🪄 Импортируем современный OCR-движок RapidOCR
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
    """Parse one screenshot using RapidOCR."""
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
        # RapidOCR возвращает результат в виде: ([[box, text, score], ...], elapse)
        ocr_results, _ = ocr(image)
    except Exception as exc:
        result["diagnostics"]["reason"] = f"Ошибка при запуске RapidOCR: {exc}"
        return result

    if not ocr_results:
        result["diagnostics"]["reason"] = "Не удалось найти текст на изображении."
        return result

    # 🪄 Группируем текстовые блоки в строки по Y-координате
    # Каждый элемент ocr_results: [box, text, score], где box = [[x,y], [x,y], [x,y], [x,y]]
    lines = []
    for box, text, score in ocr_results:
        y_center = sum(p[1] for p in box) / 4
        x_center = sum(p[0] for p in box) / 4

        # Ищем существующую строку, к которой принадлежит этот блок
        matched = False
        for line in lines:
            if abs(line["y"] - y_center) < 20:  # Допуск по высоте 20 пикселей
                line["items"].append({"x": x_center, "text": text})
                # Обновляем среднюю Y-координату строки
                line["y"] = (line["y"] * (len(line["items"]) - 1) + y_center) / len(line["items"])
                matched = True
                break

        if not matched:
            lines.append({"y": y_center, "items": [{"x": x_center, "text": text}]})

    # Сортируем строки сверху вниз
    lines.sort(key=lambda l: l["y"])

    parsed_cards = []
    for line in lines:
        # Сортируем элементы внутри строки слева направо
        line["items"].sort(key=lambda i: i["x"])
        
        line_text = " ".join(item["text"] for item in line["items"])
        lowered = line_text.lower().replace("ё", "е")

        if "транзакц" in lowered:
            card_type = "transaction"
        elif "прогноз" in lowered:
            card_type = "forecast"
        else:
            card_type = "unknown"

        if card_type == "unknown":
            continue

        date_value, time_value, month_number = _parse_date_time(line_text, datetime.now().year)
        
        # Сумма может быть в этой же строке, так как RapidOCR находит текст в одну строку
        amount_match = re.search(r"([+-]\s*\d+(?:[.,]\d{1,2})?)", line_text)
        amount = _parse_signed_amount(amount_match.group(1)) if amount_match else None

        # Если сумму не нашли в этой же строке, ищем в правой части (RapidOCR иногда разделяет)
        if amount is None:
            # Находим все элементы в строке, которые содержат цифры и знаки +/-
            for item in line["items"]:
                if re.search(r"[+-]\s*\d", item["text"]):
                    amount = _parse_signed_amount(item["text"])
                    if amount is not None:
                        break

        parsed_cards.append({
            "type": card_type,
            "text": line_text,
            "amount": amount,
            "date": date_value,
            "time": time_value,
        })

    result["diagnostics"]["detected_cards"] = len(parsed_cards)

    # 🪄 Финальная обработка и фильтрация
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
