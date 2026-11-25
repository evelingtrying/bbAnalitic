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
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5, "ма": 5,
    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "сент": 9, "окт": 10,
    "ноя": 11, "дек": 12,
}

def _parse_date_time(text: str, year: int) -> tuple[str | None, str | None]:
    lowered = text.lower().replace("ё", "е")
    # 🪄 Лечим латинские буквы в кириллице
    trans = str.maketrans('ceoapxykmthbCEOPAXYKMTHB', 'сеоархукмтнвСЕОАРХУКМТНВ')
    lowered = lowered.translate(trans)
    
    time_val = None
    time_match = re.search(r"([01]?\d|2[0-3])[:.,-]\s*([0-5]\d)", lowered)
    if time_match:
        time_val = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

    if "сегодня" in lowered:
        d = datetime.now().date()
        return d.strftime("%d.%m.%Y"), time_val
    if "вчера" in lowered:
        d = (datetime.now() - timedelta(days=1)).date()
        return d.strftime("%d.%m.%Y"), time_val

    date_match = re.search(r"\b(\d{1,2})\s+([a-zа-яё]+)", lowered)
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(2)
        for m_name, m_num in MONTHS.items():
            if month_str.startswith(m_name):
                try:
                    d = datetime(year, m_num, day)
                    return d.strftime("%d.%m.%Y"), time_val
                except ValueError:
                    pass
    return None, time_val


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
    
    # 🪄 АГРЕССИВНОЕ УВЕЛИЧЕНИЕ (до 1200px)
    if width < 1200:
        scale = 1200 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]

    # Инверсия для темной темы
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:
        gray = cv2.bitwise_not(gray)
    image_for_ocr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result["diagnostics"]["width"] = width
    result["diagnostics"]["height"] = height

    try:
        ocr = _get_engine()
        ocr_output = ocr(image_for_ocr)
    except Exception as exc:
        result["diagnostics"]["reason"] = f"Ошибка при запуске RapidOCR: {exc}"
        return result

    if not ocr_output or not getattr(ocr_output, 'txts', None):
        result["diagnostics"]["reason"] = "Не удалось найти текст на изображении."
        return result

    boxes = ocr_output.boxes
    txts = ocr_output.txts
    
    result["diagnostics"]["raw_text"] = "\n".join(txts)

    # Собираем блоки
    blocks = []
    for i in range(len(txts)):
        box = boxes[i]
        blocks.append({
            "text": txts[i],
            "x": float(np.mean(box[:, 0])),
            "y": float(np.mean(box[:, 1])),
            "h": float(np.max(box[:, 1]) - np.min(box[:, 1]))
        })

    # Сортируем сверху-вниз, слева-направо
    blocks.sort(key=lambda b: (b["y"], b["x"]))

    # Группируем в строки
    lines = []
    current_line = []
    last_y = None
    for b in blocks:
        if last_y is None or abs(b["y"] - last_y) < b["h"] * 1.5:
            current_line.append(b)
            last_y = b["y"]
        else:
            current_line.sort(key=lambda x: x["x"])
            lines.append(current_line)
            current_line = [b]
            last_y = b["y"]
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(current_line)

    parsed_bets = []
    
    # 🪄 МЕТОД СКОЛЬЗЯЩЕГО ОКНА
    i = 0
    while i < len(lines):
        # Берем текущую строку и 3 следующие
        window_lines = lines[i:i+4]
        window_text = " ".join(" ".join(b["text"] for b in line) for line in window_lines)
        lower_text = window_text.lower()
        
        # Если это транзакция — пропускаем весь блок
        if "транзакц" in lower_text or "tранзакц" in lower_text:
            result["diagnostics"]["ignored_transactions"] += 1
            i += len(window_lines)
            continue
            
        date_val, time_val = _parse_date_time(window_text, datetime.now().year)
        amount_match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", window_text)
        
        # Если нашли Дату, Время и Сумму — это ставка!
        if date_val and time_val and amount_match:
            sign = amount_match.group(1)
            try:
                amount = float(amount_match.group(2).replace(",", "."))
                if sign == '-':
                    amount = -amount
            except ValueError:
                i += 1
                continue
                
            parsed_bets.append({
                "Дата": date_val,
                "Время": time_val,
                "Результат": "Выигрыш" if amount > 0 else "Проигрыш",
                "Сумма_изменения": amount
            })
            # Пропускаем эти строки, чтобы не задублировать
            i += len(window_lines)
        else:
            i += 1

    result["diagnostics"]["detected_cards"] = len(parsed_bets) + result["diagnostics"]["ignored_transactions"]
    result["bets"] = parsed_bets
    result["diagnostics"]["recognized_bets"] = len(parsed_bets)
    
    if not parsed_bets and result["diagnostics"]["detected_cards"] == 0:
        result["diagnostics"]["reason"] = "Не найдено ни одной карточки ставки."

    return result
