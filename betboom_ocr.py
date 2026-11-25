"""OCR parser for BetBoom balance-history screenshots.

The parser detects individual history cards first, then reads each card's
forecast/transaction label, date/time and signed amount independently.
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


# Streamlit Community Cloud installs the binary and language data via packages.txt.
_tesseract = shutil.which("tesseract")
if _tesseract:
    pytesseract.pytesseract.tesseract_cmd = _tesseract
elif os.path.exists("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


MONTHS = {
    # Russian month names and common OCR prefixes
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5,
    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "сент": 9,
    "окт": 10, "ноя": 11, "дек": 12,
    # English month names, in case the account interface is in English
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
    smooth_score = np.convolve(row_score, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    relative_cutoff = float(np.percentile(smooth_score, 95)) * 0.35
    cutoff = max(0.06, min(0.25, relative_cutoff))
    active = smooth_score >= cutoff

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= 18:
                segments.append((start, y))
            start = None
    if start is not None and height - start >= 18:
        segments.append((start, height))

    if not segments:
        return [], (0, width), background

    # Remove stray separators/header bands using the median card height.
    heights = np.asarray([b - a for a, b in segments], dtype=np.float32)
    median_h = float(np.median(heights))
    segments = [
        (a, b)
        for a, b in segments
        if 0.42 * median_h <= (b - a) <= 2.2 * median_h
    ]
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
    # Correct frequent Latin/Cyrillic OCR substitutions in Russian month names
    # without damaging an English month such as "September".
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
        # Fallback for numeric dates, if BetBoom changes the display format.
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
    # Amount OCR is run with a numeric whitelist, so a trailing ₽ cannot turn
    # +124 into +1242 (a common error when OCR reads the ruble sign as "2").
    match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", text.replace("\n", " "))
    if not match:
        return None
    number = float(match.group(2).replace(",", "."))
    return number if match.group(1) == "+" else -number


def _ocr_one_card(
    gray: np.ndarray,
    row: tuple[int, int],
    content_x: tuple[int, int],
) -> dict[str, Any]:
    y0, y1 = row
    x0, x1 = content_x
    row_height = max(1, y1 - y0)
    split_x = int(x0 + (x1 - x0) * 0.80)

    # Left 80% holds the card title and date/time. OCR this area as one card,
    # so a date or time from a neighbouring card can never be attached to it.
    left_date = gray[max(0, y0 - 2):min(gray.shape[0], y1 + 2), x0:split_x]
    left_date = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(left_date)
    left_date = 255 - left_date
    left_date = cv2.resize(left_date, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    card_text = pytesseract.image_to_string(
        left_date,
        lang="rus+eng",
        config="--oem 1 --psm 6",
    ).strip()

    # The amount is in the lower part of the right-hand column. A strict
    # numeric whitelist prevents the ruble glyph from being mistaken for a digit.
    amount_y0 = min(y1 - 1, y0 + int(row_height * 0.32))
    amount_y1 = min(gray.shape[0], y0 + int(row_height * 0.90))
    amount_crop = gray[amount_y0:amount_y1, split_x:x1]
    amount_crop = 255 - amount_crop
    amount_crop = cv2.resize(amount_crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    amount_text = pytesseract.image_to_string(
        amount_crop,
        lang="eng",
        config="--oem 1 --psm 6 -c tessedit_char_whitelist=+-0123456789.,",
    ).strip()

    lowered = card_text.lower().replace("ё", "е")
    if "транзакц" in lowered:
        card_type = "transaction"
    elif "прогноз" in lowered:
        card_type = "forecast"
    else:
        card_type = "unknown"

    amount = _parse_signed_amount(amount_text)
    return {
        "type": card_type,
        "text": card_text,
        "amount_text": amount_text,
        "amount": amount,
        "date_time": card_text,
        "y": y0,
    }


def process_image_bytes(
    file_name: str,
    file_bytes: bytes,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Parse one uploaded screenshot into bets, ignored transactions and review rows."""
    result: dict[str, Any] = {
        "bets": [],
        "review": [],
        "diagnostics": {
            "file": file_name,
            "width": 0,
            "height": 0,
            "detected_cards": 0,
            "recognized_bets": 0,
            "ignored_transactions": 0,
            "needs_review": 0,
            "reason": None,
        },
    }

    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        result["diagnostics"]["reason"] = "Не удалось открыть файл как изображение."
        return result

    height, width = image.shape[:2]
    result["diagnostics"]["width"] = width
    result["diagnostics"]["height"] = height

    # A 136 px wide screenshot has already lost most glyph detail. Upscaling
    # cannot restore it, so do not silently produce fabricated statistics.
    if width < 700:
        result["diagnostics"]["reason"] = (
            f"Слишком низкая ширина ({width} px). Нужен оригинальный скриншот "
            "шириной хотя бы 700 px; не отправляйте уменьшенную миниатюру."
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
            f"В системе не установлены языки Tesseract: {missing}. "
            "Добавьте tesseract-ocr-rus и tesseract-ocr-eng в packages.txt."
        )
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, content_x, _background = _find_card_rows(gray)
    if not rows:
        result["diagnostics"]["reason"] = (
            "Не удалось выделить карточки истории. Загрузите скриншот самой "
            "истории баланса BetBoom, где видны строки с прогнозами."
        )
        return result

    result["diagnostics"]["detected_cards"] = len(rows)
    parsed_cards: list[dict[str, Any] | None] = [None] * len(rows)
    workers = max(1, min(4, os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_ocr_one_card, gray, row, content_x): index
            for index, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                parsed_cards[index] = future.result()
            except Exception as exc:
                y0 = rows[index][0]
                parsed_cards[index] = {
                    "type": "unknown",
                    "text": f"Ошибка OCR: {exc}",
                    "amount_text": "",
                    "amount": None,
                    "date_time": "",
                    "y": y0,
                }
            completed += 1
            if progress_callback:
                progress_callback(completed, len(rows))

    # The history page is sorted newest -> oldest. The year is not printed on
    # each BetBoom card, so handle the January -> December rollover explicitly.
    year = datetime.now().year
    previous_month: int | None = None

    for card_index, card in enumerate(parsed_cards, start=1):
        if card is None:
            continue
        card_type = card["type"]

        if card_type == "transaction":
            result["diagnostics"]["ignored_transactions"] += 1
            continue

        date_value, time_value, month_number = _parse_date_time(card["date_time"], year)
        if month_number is not None and previous_month is not None:
            if month_number - previous_month >= 9:
                year -= 1
                # Reparse the date with the adjusted year.
                date_value, time_value, month_number = _parse_date_time(card["date_time"], year)
        if month_number is not None:
            previous_month = month_number

        amount = card["amount"]
        if card_type != "forecast" or date_value is None or time_value is None or amount is None:
            result["review"].append(
                {
                    "Файл": file_name,
                    "Карточка": card_index,
                    "Тип OCR": card_type,
                    "Распознанный текст": card["text"],
                    "Текст суммы": card["amount_text"],
                    "Причина": "Не удалось уверенно определить тип, дату, время или сумму",
                }
            )
            result["diagnostics"]["needs_review"] += 1
            continue

        outcome = "Выигрыш" if amount > 0 else "Проигрыш" if amount < 0 else "Возврат"
        result["bets"].append(
            {
                "Дата": date_value,
                "Время": time_value,
                "Результат": outcome,
                "Сумма_изменения": float(amount),
                "Источник": file_name,
                "Карточка": card_index,
            }
        )

    result["diagnostics"]["recognized_bets"] = len(result["bets"])
    return result
