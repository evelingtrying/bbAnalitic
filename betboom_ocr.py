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

def _parse_date_string(text: str, year: int) -> str | None:
    lowered = text.lower().replace("ё", "е")
    # Лечим латинские буквы в кириллице
    trans = str.maketrans('ceoapxykmthbCEOPAXYKMTHB', 'сеоархукмтнвСЕОАРХУКМТНВ')
    lowered = lowered.translate(trans)

    if "сегодня" in lowered:
        return datetime.now().date().strftime("%d.%m.%Y")
    if "вчера" in lowered:
        return (datetime.now() - timedelta(days=1)).date().strftime("%d.%m.%Y")

    date_match = re.search(r"\b(\d{1,2})\s+([a-zа-яё]+)", lowered)
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(2)
        for m_name, m_num in MONTHS.items():
            if month_str.startswith(m_name):
                try:
                    d = datetime(year, m_num, day)
                    return d.strftime("%d.%m.%Y")
                except ValueError:
                    pass
    return None


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
    
    # 🪄 УВЕЛИЧЕНИЕ: Если картинка маленькая, растягиваем её до 1500px
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]

    # 🪄 УЛУЧШЕНИЕ КАРТИНКИ: Контраст + инверсия для темной темы
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
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

    # Собираем блоки текста с их координатами
    blocks = []
    for i in range(len(txts)):
        box = boxes[i]
        blocks.append({
            "text": txts[i],
            "x": float(np.mean(box[:, 0])),
            "y": float(np.mean(box[:, 1])),
            "text_lower": txts[i].lower().replace("ё", "е")
        })

    parsed_bets = []
    
    # 🪄 2D ПАРСИНГ: Ищем суммы и смотрим, что находится рядом
    for b in blocks:
        # 1. Ищем сумму (плюс или минус с цифрами)
        amount_match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", b["text"])
        if not amount_match:
            continue
            
        sign = amount_match.group(1)
        try:
            amount = float(amount_match.group(2).replace(",", "."))
            if sign == '-':
                amount = -amount
        except ValueError:
            continue

        # 2. Ищем контекст вокруг этой суммы (в пределах 120 пикселей по Y)
        y_current = b["y"]
        context_blocks = [other for other in blocks if abs(other["y"] - y_current) < 120]
        
        # Проверяем, нет ли рядом слова "Транзакция"
        is_transaction = any("транзакц" in cb["text_lower"] for cb in context_blocks)
        if is_transaction:
            result["diagnostics"]["ignored_transactions"] += 1
            continue

        # Ищем дату и время в этом контексте
        date_val = None
        time_val = None
        
        for cb in context_blocks:
            if not date_val:
                parsed_d = _parse_date_string(cb["text"], datetime.now().year)
                if parsed_d:
                    date_val = parsed_d
            
            if not time_val:
                time_match = re.search(r"([01]?\d|2[0-3])[:.,]\s*([0-5]\d)", cb["text"])
                if time_match:
                    time_val = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

        # Если нашли и дату, и время — это ставка!
        if date_val and time_val:
            parsed_bets.append({
                "Дата": date_val,
                "Время": time_val,
                "Результат": "Выигрыш" if amount > 0 else "Проигрыш",
                "Сумма_изменения": amount
            })
        else:
            result["diagnostics"]["skipped_cards"] += 1
            result["diagnostics"]["failed_cards"].append(f"Сумма: {b['text']} | Контекст: {' '.join(cb['text'] for cb in context_blocks)}")

    # Удаляем дубликаты (если одна и та же сумма распозналась дважды)
    unique_bets = []
    for bet in parsed_bets:
        if bet not in unique_bets:
            unique_bets.append(bet)

    result["diagnostics"]["detected_cards"] = len(unique_bets) + result["diagnostics"]["ignored_transactions"]
    result["bets"] = unique_bets
    result["diagnostics"]["recognized_bets"] = len(unique_bets)
    
    if not unique_bets and result["diagnostics"]["detected_cards"] == 0:
        result["diagnostics"]["reason"] = "Не найдено ни одной карточки ставки."

    return result
