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

def _extract_amount(text: str) -> float | None:
    # Ищем плюс/минус, цифры и возможные копейки. Игнорируем буквы (P, р, О и т.д.)
    match = re.search(r"([+-])\s*(\d+(?:[.,]\d{1,2})?)", text)
    if not match:
        return None
    try:
        val = float(match.group(2).replace(",", "."))
        return -val if match.group(1) == '-' else val
    except ValueError:
        return None

def _extract_time(text: str) -> str | None:
    match = re.search(r"([01]?\d|2[0-3])[:.,-]\s*([0-5]\d)", text)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
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
    
    # 🪄 УВЕЛИЧЕНИЕ: Растягиваем картинку до 1500px
    if width < 1500:
        scale = 1500 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]

    # 🪄 УЛУЧШЕНИЕ: Выравниваем контраст и инвертируем цвета
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

    # Собираем все блоки текста с координатами
    blocks = []
    for i in range(len(txts)):
        box = boxes[i]
        blocks.append({
            "text": txts[i],
            "x": float(np.mean(box[:, 0])),
            "y": float(np.mean(box[:, 1])),
            "h": float(np.max(box[:, 1]) - np.min(box[:, 1]))
        })

    parsed_bets = []
    today_str = datetime.now().strftime("%d.%m.%Y")

    # 🪄 ИЩЕМ СУММЫ И ИЩЕМ К НИМ БЛИЖАЙШЕЕ ВРЕМЯ
    for b in blocks:
        amount = _extract_amount(b["text"])
        if amount is None:
            continue

        # Нашли сумму. Теперь ищем время на этой же линии (или очень близко)
        y_current = b["y"]
        closest_time = None
        min_dist = float('inf')

        for other in blocks:
            if other is b:
                continue
            t = _extract_time(other["text"])
            if t:
                dist = abs(other["y"] - y_current)
                if dist < min_dist and dist < 150: # В пределах 150 пикселей
                    min_dist = dist
                    closest_time = t

        if not closest_time:
            result["diagnostics"]["skipped_cards"] += 1
            result["diagnostics"]["failed_cards"].append(f"Сумма {amount} без времени: {b['text']}")
            continue

        # Проверяем, нет ли рядом слова "Транзакция"
        is_tx = False
        for other in blocks:
            if abs(other["y"] - y_current) < 100:
                if "транзакц" in other["text"].lower():
                    is_tx = True
                    break

        if is_tx:
            result["diagnostics"]["ignored_transactions"] += 1
            continue

        # Если всё ок — добавляем ставку
        parsed_bets.append({
            "Дата": today_str, # Если OCR не увидел "Сегодня", ставим текущую дату
            "Время": closest_time,
            "Результат": "Выигрыш" if amount > 0 else "Проигрыш",
            "Сумма_изменения": amount
        })

    # Удаляем дубликаты (если одна и та же ставка спарсилась дважды)
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
