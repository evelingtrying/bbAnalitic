from __future__ import annotations

from datetime import datetime
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from betboom_ocr import process_image_bytes


st.set_page_config(
    page_title="BetBoom Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Визуальный стиль
# -----------------------------------------------------------------------------
st.markdown(
    r'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #141b2f;
        --panel-2: #19223a;
        --line: rgba(190, 202, 236, .11);
        --muted: #98a5c2;
        --text: #eef2ff;
        --purple: #a78bfa;
        --green: #34d399;
        --red: #fb7185;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(900px 420px at 12% -8%, rgba(99,102,241,.22), transparent 65%),
            radial-gradient(700px 360px at 95% 0%, rgba(168,85,247,.15), transparent 62%),
            linear-gradient(135deg, #0b1020 0%, #111831 52%, #101629 100%);
        color: var(--text);
    }
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label { color: var(--text); }
    h1 {
        font-weight: 800 !important;
        letter-spacing: -.045em;
        background: linear-gradient(100deg, #f5f3ff 0%, #c4b5fd 48%, #a5b4fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h2 { font-weight: 750 !important; letter-spacing: -.025em; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13,19,38,.98), rgba(18,25,46,.98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e8eaff; }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.025);
        border: 1px dashed rgba(196,181,253,.34);
        border-radius: 16px;
        padding: 10px;
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(196,181,253,.18) !important;
        border-radius: 12px !important;
        min-height: 44px;
        font-weight: 700 !important;
        color: #fff !important;
        background: linear-gradient(115deg, #6755d9 0%, #8b5cf6 55%, #a78bfa 100%) !important;
        box-shadow: 0 8px 22px rgba(124,58,237,.18);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 26px rgba(124,58,237,.28);
    }
    .stButton > button:disabled { opacity: .45; }

    .app-brand { display:flex; align-items:center; justify-content:center; gap:14px; margin: .15rem 0 .3rem; }
    .brand-gem {
        width:42px; height:42px; display:grid; place-items:center; border-radius:14px;
        color:#fff; font-size:23px;
        background:linear-gradient(145deg, #8b5cf6, #6366f1);
        box-shadow:0 10px 30px rgba(124,58,237,.32);
    }
    .app-subtitle { text-align:center; color:#9ba8c6; font-size:13px; margin:0 0 26px; }

    .summary-card {
