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
