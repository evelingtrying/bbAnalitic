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
    """
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
        position:relative; overflow:hidden; min-height:124px; padding:20px 21px;
        border:1px solid var(--line); border-radius:19px;
        background:linear-gradient(145deg, rgba(30,41,68,.88), rgba(20,28,49,.90));
        box-shadow:0 12px 34px rgba(0,0,0,.14);
    }
    .summary-card:after {
        content:""; position:absolute; width:130px; height:130px; right:-58px; top:-68px;
        border-radius:50%; background:rgba(167,139,250,.09); filter:blur(1px);
    }
    .summary-label { color:#9aa7c4; text-transform:uppercase; font-size:10px; font-weight:800; letter-spacing:.11em; margin-bottom:12px; }
    .summary-value { color:#f8faff; font-size:29px; line-height:1.1; font-weight:800; letter-spacing:-.045em; }
    .summary-value.purple { color:#c4b5fd; }
    .summary-value.green { color:#6ee7b7; }
    .summary-value.red { color:#fda4af; }
    .summary-hint { color:#8391af; font-size:11px; margin-top:8px; }

    .section-heading { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin:5px 0 16px; }
    .section-kicker { color:#a78bfa; font-size:10px; text-transform:uppercase; letter-spacing:.16em; font-weight:800; margin-bottom:5px; }
    .section-title { color:#f0f3ff; font-size:23px; font-weight:800; letter-spacing:-.035em; }
    .section-note { color:#8996b3; font-size:12px; }

    .stats-group {
        border:1px solid var(--line); border-radius:21px; padding:20px 20px 17px;
        background:linear-gradient(145deg, rgba(25,35,59,.83), rgba(18,26,46,.90));
        box-shadow:0 14px 36px rgba(0,0,0,.12); height:100%;
    }
    .group-head { display:flex; align-items:center; gap:11px; margin-bottom:17px; }
    .group-icon { width:34px; height:34px; display:grid; place-items:center; border-radius:11px; font-size:17px; }
    .group-icon.win { background:rgba(16,185,129,.13); color:#6ee7b7; }
    .group-icon.loss { background:rgba(244,63,94,.13); color:#fda4af; }
    .group-name { color:#e9edff; font-weight:800; font-size:15px; }
    .group-sub { color:#8592af; font-size:11px; margin-top:2px; }

    .stat-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; }
    .stat-tile {
        min-height:130px; box-sizing:border-box; padding:16px 15px; border-radius:15px;
        background:rgba(9,15,30,.45); border:1px solid rgba(185,198,232,.09);
        display:flex; flex-direction:column; justify-content:space-between;
        transition:transform .18s ease, border-color .18s ease, background .18s ease;
    }
    .stat-tile:hover { transform:translateY(-2px); border-color:rgba(167,139,250,.30); background:rgba(15,22,42,.8); }
    .tile-label { color:#94a1bd; font-size:10px; line-height:1.35; font-weight:700; min-height:27px; }
    .tile-value { font-size:clamp(17px, 1.65vw, 23px); line-height:1.12; font-weight:800; letter-spacing:-.045em; white-space:nowrap; }
    .tile-value.win { color:#6ee7b7; }
    .tile-value.loss { color:#fda4af; }
    .tile-value.neutral { color:#d8ccff; }
    .tile-foot { color:#71809e; font-size:10px; }

    .activity-row { display:flex; gap:14px; }
    .activity-card {
        flex:1; padding:17px 19px; border-radius:16px; border:1px solid var(--line);
        background:linear-gradient(145deg, rgba(30,41,68,.68), rgba(18,26,46,.82));
    }
    .activity-label { color:#96a3bf; font-size:11px; font-weight:700; margin-bottom:8px; }
    .activity-value { color:#f4f2ff; font-size:25px; font-weight:800; letter-spacing:-.04em; }

    .winrate-wrap { margin:17px 0 4px; padding:14px 16px; border-radius:14px; background:rgba(8,14,28,.42); border:1px solid rgba(185,198,232,.08); }
    .winrate-top { display:flex; justify-content:space-between; color:#9aa7c2; font-size:11px; margin-bottom:9px; }
    .winrate-track { height:8px; background:rgba(251,113,133,.22); border-radius:99px; overflow:hidden; }
    .winrate-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#34d399,#6ee7b7); }

    .stTabs [data-baseweb="tab-list"] { gap:9px; background:rgba(10,16,31,.56); border:1px solid var(--line); padding:7px; border-radius:16px; }
    .stTabs [data-baseweb="tab"] { border-radius:11px; color:#9ca9c6; font-weight:700; padding:9px 17px; }
    .stTabs [aria-selected="true"] { background:linear-gradient(120deg,rgba(109,87,220,.9),rgba(139,92,246,.9)) !important; color:white !important; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    [data-testid="stMetric"] { background:rgba(20,28,49,.7); border:1px solid var(--line); padding:15px; border-radius:14px; }
    hr { border-color:var(--line); }

    @media (max-width: 1100px) {
        .stat-grid { grid-template-columns:1fr; }
        .stat-tile { min-height:108px; }
    }
    @media (max-width: 900px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .summary-value { font-size:24px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BET_COLUMNS = ["Дата", "Время", "Результат", "Сумма_изменения"]


def empty_bets() -> pd.DataFrame:
    return pd.DataFrame(columns=BET_COLUMNS)


def normalize_bets(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None or data.empty:
        return empty_bets()
    df = data.copy()
    for column in BET_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0 if column == "Сумма_изменения" else ""
    df["Дата"] = df["Дата"].astype(str).str.strip()
    df["Время"] = df["Время"].astype(str).str.strip()
    df["Сумма_изменения"] = pd.to_numeric(df["Сумма_изменения"], errors="coerce")
    df = df.dropna(subset=["Сумма_изменения"])
    df = df[df["Дата"].str.match(r"^\d{2}\.\d{2}\.\d{4}$", na=False)]
    df = df[df["Время"].str.match(r"^\d{2}:\d{2}$", na=False)]
    df["Результат"] = df["Сумма_изменения"].apply(
        lambda value: "Выигрыш" if value > 0 else "Проигрыш" if value < 0 else "Возврат"
    )
    return df[BET_COLUMNS].reset_index(drop=True)


def fmt_money(value: float, decimals: int = 0) -> str:
    if decimals:
        text = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:,.0f}"
    return text.replace(",", " ")


def fmt_signed_money(value: float, decimals: int = 0) -> str:
    prefix = "+" if value > 0 else ""
    return f"{prefix}{fmt_money(value, decimals)} ₽"


def month_title(period: str) -> str:
    names = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
        5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
        9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
    }
    year, month = period.split("-")
    return f"{names[int(month)]} {year}"


def render_stat_tile(label: str, value: str, foot: str, tone: str) -> str:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_foot = html.escape(foot)
    return (
        '<div class="stat-tile">'
        f'<div class="tile-label">{safe_label}</div>'
        f'<div class="tile-value {tone}">{safe_value}</div>'
        f'<div class="tile-foot">{safe_foot}</div>'
        '</div>'
    )


# -----------------------------------------------------------------------------
# Состояние приложения
# -----------------------------------------------------------------------------
if "bets_df" not in st.session_state:
    st.session_state.bets_df = empty_bets()
if "last_import_summary" not in st.session_state:
    st.session_state.last_import_summary = None


# -----------------------------------------------------------------------------
# Шапка
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="app-brand"><div class="brand-gem">◆</div><h1 style="margin:0">BetBoom Analytics</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Персональная аналитика ставок · автоматическое распознавание истории баланса</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Загрузка — автоматический импорт без ручной проверки строк
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💎 BetBoom Analytics")
    st.caption("Автоматический импорт истории баланса")
    st.divider()
    st.markdown("### Загрузка скриншотов")
    uploaded_files = st.file_uploader(
        "Оригинальные изображения истории",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Лучше загружать оригиналы шириной от 1000 пикселей.",
    )

    process_clicked = st.button(
        "✦ Распознать и добавить",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_clicked and uploaded_files:
        all_bets: list[dict] = []
        total_detected = 0
        total_transactions = 0
        total_skipped = 0
        file_errors: list[str] = []
        progress = st.progress(0.0)
        status = st.empty()

        for index, uploaded in enumerate(uploaded_files):
            status.caption(f"Распознавание изображения {index + 1} из {len(uploaded_files)}…")
            parsed = process_image_bytes(uploaded.name, uploaded.getvalue())
            all_bets.extend(parsed["bets"])
            diag = parsed["diagnostics"]
            total_detected += int(diag.get("detected_cards", 0))
            total_transactions += int(diag.get("ignored_transactions", 0))
            total_skipped += int(diag.get("skipped_cards", 0))
            if diag.get("reason"):
                file_errors.append(str(diag["reason"]))
            progress.progress((index + 1) / len(uploaded_files))

        status.empty()
        progress.empty()

        old_df = normalize_bets(st.session_state.bets_df)
        new_df = normalize_bets(pd.DataFrame(all_bets))
        combined = pd.concat([old_df, new_df], ignore_index=True)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(
            subset=["Дата", "Время", "Сумма_изменения"],
            keep="first",
        ).reset_index(drop=True)
        added = len(combined) - len(old_df)
        st.session_state.bets_df = normalize_bets(combined)
        st.session_state.last_import_summary = {
            "added": max(0, added),
            "recognized": len(new_df),
            "detected": total_detected,
            "transactions": total_transactions,
            "skipped": total_skipped,
            "duplicates": max(0, before_dedup - len(combined)),
            "errors": file_errors,
        }
        st.rerun()

    st.divider()
    st.markdown("### База данных")
    base_df = normalize_bets(st.session_state.bets_df)
    if not base_df.empty:
        csv_data = base_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "↓ Скачать базу CSV",
            data=csv_data,
            file_name="betboom_bets.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Очистить базу", use_container_width=True):
            st.session_state.bets_df = empty_bets()
            st.session_state.last_import_summary = None
            st.rerun()
        st.caption("База хранится в текущей сессии. Периодически сохраняйте CSV.")
    else:
        st.caption("Пока нет импортированных ставок.")


# -----------------------------------------------------------------------------
# Итог автоматического импорта
# -----------------------------------------------------------------------------
summary = st.session_state.last_import_summary
if summary:
    st.markdown("### Последний импорт")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Добавлено", summary["added"])
    s2.metric("Распознано", summary["recognized"])
    s3.metric("Транзакции исключены", summary["transactions"])
    s4.metric("Автоматически пропущено", summary["skipped"])
    if summary["duplicates"]:
        st.caption(f"Повторные записи автоматически объединены: {summary['duplicates']}.")
    if summary["errors"]:
        with st.expander("Не удалось обработать некоторые изображения"):
            for error in summary["errors"]:
                st.write(f"• {error}")


# -----------------------------------------------------------------------------
# Данные и расчёты
# -----------------------------------------------------------------------------
df = normalize_bets(st.session_state.bets_df)
if df.empty:
    st.markdown(
        """
        <div class="summary-card" style="text-align:center;padding:34px 24px;margin-top:12px">
            <div style="font-size:30px;margin-bottom:10px">📸</div>
            <div style="font-size:20px;font-weight:800;color:#eef2ff;margin-bottom:7px">Загрузите историю ставок</div>
            <div style="color:#98a5c2;font-size:13px">Выберите оригинальные скриншоты в левой панели — ставки будут распознаны и добавлены автоматически.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Sort newest-first in the UI, chronological for balance chart.
df["ДатаВремя"] = pd.to_datetime(
    df["Дата"] + " " + df["Время"],
    format="%d.%m.%Y %H:%M",
    errors="coerce",
)
df = df.dropna(subset=["ДатаВремя"]).sort_values("ДатаВремя").reset_index(drop=True)
df["Месяц"] = df["ДатаВремя"].dt.strftime("%Y-%m")
df["Номер ставки"] = range(1, len(df) + 1)
df["Баланс"] = df["Сумма_изменения"].cumsum()

wins_df = df[df["Сумма_изменения"] > 0]
losses_df = df[df["Сумма_изменения"] < 0]
pushes_df = df[df["Сумма_изменения"] == 0]
total_bets = len(df)
settled_bets = len(wins_df) + len(losses_df)
winrate = (len(wins_df) / settled_bets * 100) if settled_bets else 0.0
total_pnl = float(df["Сумма_изменения"].sum())
avg_loss_size = abs(float(losses_df["Сумма_изменения"].mean())) if not losses_df.empty else 0.0


# -----------------------------------------------------------------------------
# Основные карточки
# -----------------------------------------------------------------------------
st.markdown('<div class="section-heading"><div><div class="section-kicker">Ваш банкролл</div><div class="section-title">Общая статистика</div></div></div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    st.markdown(
        f'<div class="summary-card"><div class="summary-label">Всего ставок</div><div class="summary-value">{total_bets:,}</div><div class="summary-hint">Включая возвраты</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    wr_tone = "green" if winrate >= 50 else "purple" if winrate >= 40 else "red"
    st.markdown(
        f'<div class="summary-card"><div class="summary-label">Винрейт</div><div class="summary-value {wr_tone}">{winrate:.1f}%</div><div class="summary-hint">От рассчитанных ставок</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    pnl_tone = "green" if total_pnl >= 0 else "red"
    st.markdown(
        f'<div class="summary-card"><div class="summary-label">Чистый P&amp;L</div><div class="summary-value {pnl_tone}">{fmt_signed_money(total_pnl)}</div><div class="summary-hint">Сумма всех изменений баланса</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="summary-card"><div class="summary-label">Средний проигрыш</div><div class="summary-value">{fmt_money(avg_loss_size, 2)} ₽</div><div class="summary-hint">Среднее списание по прогнозам</div></div>',
        unsafe_allow_html=True,
    )

st.caption(
    f"Выигрышей: {len(wins_df)}  ·  Проигрышей: {len(losses_df)}  ·  "
    f"Возвратов: {len(pushes_df)}  ·  P&L: {fmt_signed_money(total_pnl, 2)}"
)

chart_tab, history_tab, stats_tab = st.tabs(
    ["📈 График баланса", "🗓️ История по месяцам", "✦ Ключевые цифры"]
)


# -----------------------------------------------------------------------------
# График
# -----------------------------------------------------------------------------
with chart_tab:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Номер ставки"],
            y=df["Баланс"],
            mode="lines",
            name="Баланс",
            line=dict(color="#a78bfa", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,.12)",
            hovertemplate="Ставка %{x}<br>Баланс %{y:,.2f} ₽<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(203,213,225,.55)", opacity=.7)
    fig.update_layout(
        title=dict(text="Динамика банкролла", font=dict(size=19, color="#eef2ff")),
        xaxis_title="Номер ставки",
        yaxis_title="Изменение баланса, ₽",
        hovermode="x unified",
        margin=dict(l=18, r=18, t=58, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d5dced"),
        xaxis=dict(gridcolor="rgba(255,255,255,.055)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,.055)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# История — без столбца «Источник»
# -----------------------------------------------------------------------------
with history_tab:
    periods = sorted(df["Месяц"].dropna().unique(), reverse=True)
    if periods:
        selected_period = st.selectbox("Период", periods, format_func=month_title)
        month_df = df[df["Месяц"] == selected_period].copy()
        month_df["Сумма, ₽"] = month_df["Сумма_изменения"].map(lambda value: fmt_money(value, 2))
        display_df = month_df.sort_values("ДатаВремя", ascending=False)[
            ["Дата", "Время", "Результат", "Сумма, ₽"]
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Дата": st.column_config.TextColumn("Дата", width="medium"),
                "Время": st.column_config.TextColumn("Время", width="small"),
                "Результат": st.column_config.TextColumn("Результат", width="medium"),
                "Сумма, ₽": st.column_config.TextColumn("Изменение, ₽", width="medium"),
            },
        )
    else:
        st.info("Нет данных для отображения.")


# -----------------------------------------------------------------------------
# Ключевые цифры — обновлённый дизайн
# -----------------------------------------------------------------------------
with stats_tab:
    st.markdown(
        '<div class="section-heading"><div><div class="section-kicker">Разбор результатов</div><div class="section-title">Ключевые цифры</div><div class="section-note">Сводка по выигрышам, проигрышам и игровой активности</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="winrate-wrap">
            <div class="winrate-top"><span>Доля выигрышных ставок</span><strong style="color:#e9eaff">{winrate:.1f}%</strong></div>
            <div class="winrate-track"><div class="winrate-fill" style="width:{max(0, min(100, winrate)):.2f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    largest_win = float(wins_df["Сумма_изменения"].max()) if not wins_df.empty else 0.0
    total_won = float(wins_df["Сумма_изменения"].sum())
    avg_win = float(wins_df["Сумма_изменения"].mean()) if not wins_df.empty else 0.0
    largest_loss = float(losses_df["Сумма_изменения"].min()) if not losses_df.empty else 0.0
    total_lost = float(losses_df["Сумма_изменения"].sum())
    avg_loss = float(losses_df["Сумма_изменения"].mean()) if not losses_df.empty else 0.0

    left, right = st.columns(2, gap="large")
    with left:
        win_tiles = "".join([
            render_stat_tile("Самый крупный", fmt_signed_money(largest_win), "Максимальная выплата", "win"),
            render_stat_tile("Сумма выигрышей", fmt_signed_money(total_won, 2), f"{len(wins_df)} выигрышных ставок", "win"),
            render_stat_tile("Средний выигрыш", fmt_signed_money(avg_win, 2), "На одну выигрышную ставку", "win"),
        ])
        st.markdown(
            '<div class="stats-group"><div class="group-head"><div class="group-icon win">↗</div><div><div class="group-name">Выигрыши</div><div class="group-sub">Положительные изменения баланса</div></div></div>'
            f'<div class="stat-grid">{win_tiles}</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        loss_tiles = "".join([
            render_stat_tile("Самый крупный", fmt_signed_money(largest_loss), "Максимальное списание", "loss"),
            render_stat_tile("Сумма проигрышей", fmt_signed_money(total_lost, 2), f"{len(losses_df)} проигрышных ставок", "loss"),
            render_stat_tile("Средний проигрыш", fmt_signed_money(avg_loss, 2), "На одну проигрышную ставку", "loss"),
        ])
        st.markdown(
            '<div class="stats-group"><div class="group-head"><div class="group-icon loss">↘</div><div><div class="group-name">Проигрыши</div><div class="group-sub">Отрицательные изменения баланса</div></div></div>'
            f'<div class="stat-grid">{loss_tiles}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    unique_days = df["ДатаВремя"].dt.date.nunique()
    common_loss = (
        float(losses_df["Сумма_изменения"].abs().mode().iloc[0])
        if not losses_df.empty and not losses_df["Сумма_изменения"].abs().mode().empty
        else 0.0
    )
    extra1, extra2, extra3 = st.columns(3, gap="medium")
    with extra1:
        st.markdown(
            f'<div class="activity-card"><div class="activity-label">📅 Дней со ставками</div><div class="activity-value">{unique_days}</div></div>',
            unsafe_allow_html=True,
        )
    with extra2:
        st.markdown(
            f'<div class="activity-card"><div class="activity-label">🎯 Самая частая сумма проигрыша</div><div class="activity-value">{fmt_money(common_loss, 2)} ₽</div></div>',
            unsafe_allow_html=True,
        )
    with extra3:
        st.markdown(
            f'<div class="activity-card"><div class="activity-label">📊 Рассчитанных ставок</div><div class="activity-value">{settled_bets}</div></div>',
            unsafe_allow_html=True,
        )
