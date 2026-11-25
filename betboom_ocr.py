"""OCR parser for BetBoom balance-history screenshots.

Each history card is located independently. Amount recognition uses only the
lower line in the right-hand column, separating the amount from the label and
ruble currency glyph to reduce false trailing digits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
import re
import shutil
from typing import Callable, Any

import cv2
import numpy as np
import pytesseract


_tesseract = shutil.which("tesseract")
if _tesseract:
    pytesseract.pytesseract.tesseract_cmd = _tesseract
elif os.path.exists("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5,
    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "сент": 9,
    "окт": 10, "ноя": 11, "дек": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def _find_card_rows(gray: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], float]:
    """Find repeated dark history cards by their horizontal brightness profile."""
    height, width = gray.shape
    corner_h = max(5, height // 30)
    corner_w = max(5, width // 30)
    corners = np.concatenate(
        [
            gray[:corner_h, :corner_w].ravel(),
            gray[:corner_h, -corner_w:].ravel(),
            gray[-corner_h:, :corner_w].ravel(),
            gray[-corner_h:, -corner_w:].ravel(),
        ]
    )
    background = float(np.median(corners))
    pixel_threshold = background + max(3.0, min(8.0, (255.0 - background) * 0.025))

    row_score = np.mean(gray > pixel_threshold, axis=1)
    
    # Более чувствительное определение строк, чтобы ловить тонкие отступы
    smooth_score = np.convolve(row_score, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    relative_cutoff = float(np.percentile(smooth_score, 95)) * 0.25
    cutoff = max(0.02, min(0.15, relative_cutoff))
    active = smooth_score >= cutoff

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= 15:
                segments.append((start, y))
            start = None
    if start is not None and height - start >= 15:
        segments.append((start, height))

    if not segments:
        return [], (0, width), background

    card_rows = np.zeros(height, dtype=bool)
    for y0, y1 in segments:
        card_rows[y0:y1] = True

    column_occupancy = np.mean(gray[card_rows, :] > pixel_threshold, axis=0)
    occupied = np.flatnonzero(column_occupancy > 0.35)
    if occupied.size == 0:
        return [], (0, width), background

    x0 = max(0, int(occupied[0]) - 8)
    x1 = min(width, int(occupied[-1]) + 9)
    if x1 - x0 < 250:
        return [], (0, width), background

    return segments, (x0, x1), background


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
    """Parse a signed amount; currency glyphs are excluded from the OCR crop."""
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


def _ocr_one_card(
    gray: np.ndarray,
    row: tuple[int, int],
    content_x: tuple[int, int],
) -> list[dict[str, Any]]:
    y0, y1 = row
    x0, x1 = content_x
    
    # Защита от слишком маленьких обрезков
    if y1 - y0 < 15:
        return []

    row_height = max(1, y1 - y0)
    split_x = int(x0 + (x1 - x0) * 0.80)

    # OCR левой части (дата, время, тип)
    left_date = gray[max(0, y0 - 2):min(gray.shape[0], y1 + 2), x0:split_x]
    left_date = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(left_date)
    left_date = 255 - left_date
    left_date = cv2.resize(left_date, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    card_text = pytesseract.image_to_string(
        left_date,
        lang="rus+eng",
        config="--oem 1 --psm 6",
    ).strip()

    lowered = card_text.lower().replace("ё", "е")
    
    # 🪄 РЕКУРСИВНОЕ РАЗДЕЛЕНИЕ: Если нашли 2 даты в одном блоке — режем пополам!
    date_matches = re.findall(r"\b\d{1,2}\s+[a-zа-яё]+", lowered)
    if len(date_matches) > 1:
        mid = (y0 + y1) // 2
        # Возвращаем список из двух результатов (верхняя и нижняя часть)
        return _ocr_one_card(gray, (y0, mid), content_x) + _ocr_one_card(gray, (mid, y1), content_x)

    # OCR правой части (сумма)
    amount_y0 = min(y1 - 1, y0 + int(row_height * 0.48))
    amount_y1 = min(gray.shape[0], y0 + int(row_height * 0.90))
    amount_crop = gray[amount_y0:amount_y1, split_x:x1]
    amount_crop = 255 - amount_crop
    amount_crop = cv2.resize(amount_crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    amount_text = pytesseract.image_to_string(
        amount_crop,
        lang="eng",
        config="--oem 1 --psm 6 -c tessedit_char_whitelist=+-0123456789.,",
    ).strip()

    if "транзакц" in lowered:
        card_type = "transaction"
    elif "прогноз" in lowered:
        card_type = "forecast"
    else:
        card_type = "unknown"

    return [{
        "type": card_type,
        "text": card_text,
        "amount_text": amount_text,
        "amount": _parse_signed_amount(amount_text),
        "date_time": card_text,
        "y": y0,
    }]


def process_image_bytes(
    file_name: str,
    file_bytes: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Parse one screenshot. Low-confidence cards are skipped automatically."""
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
        available_languages = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        result["diagnostics"]["reason"] = f"Не удалось запустить Tesseract: {exc}"
        return result
    if not {"rus", "eng"}.issubset(available_languages):
        missing = ", ".join(sorted({"rus", "eng"} - available_languages))
        result["diagnostics"]["reason"] = (
            f"Не установлены языки Tesseract: {missing}. "
            "Проверьте packages.txt."
        )
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, content_x, _background = _find_card_rows(gray)
    if not rows:
        result["diagnostics"]["reason"] = "Не удалось найти карточки истории на изображении."
        return result

    result["diagnostics"]["detected_cards"] = len(rows)
    parsed_cards: list[dict[str, Any]] = []
    workers = max(1, min(4, os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_ocr_one_card, gray, row, content_x): index
            for index, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(future_to_index):
            try:
                # Теперь результат — это список, так как могут быть склеенные карточки
                parsed_cards.extend(future.result())
            except Exception:
                pass
            completed += 1
            if progress_callback:
                progress_callback(completed, len(rows))

    year = datetime.now().year
    previous_month: int | None = None

    for card in parsed_cards:
        if card["type"] == "transaction":
            result["diagnostics"]["ignored_transactions"] += 1
            continue

        date_value, time_value, month_number = _parse_date_time(card["date_time"], year)
        if month_number is not None and previous_month is not None:
            if month_number - previous_month >= 9:
                year -= 1
                date_value, time_value, month_number = _parse_date_time(card["date_time"], year)
        if month_number is not None:
            previous_month = month_number

        amount = card["amount"]
        if card["type"] != "forecast" or date_value is None or time_value is None or amount is None:
            result["diagnostics"]["skipped_cards"] += 1
            continue

        outcome = "Выигрыш" if amount > 0 else "Проигрыш" if amount < 0 else "Возврат"
        result["bets"].append(
            {
                "Дата": date_value,
                "Время": time_value,
                "Результат": outcome,
                "Сумма_изменения": float(amount),
            }
        )

    result["diagnostics"]["recognized_bets"] = len(result["bets"])
    return result
