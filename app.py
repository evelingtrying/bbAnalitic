from __future__ import annotations

from datetime import datetime
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from betboom_ocr import process_image_bytes


st.set_page_config(
    page_title="BetBoom Analytics",
    page_icon="💎",
    layout="wide",
)

# ------------------------------
# Дизайн
# ------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    h1 { background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; font-weight: 800 !important; text-align: center; }
    .custom-card { background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.09);
         border-radius: 16px; padding: 20px; min-height: 112px; }
    .card-title { color: #94a3b8; font-weight: 600; font-size: 12px; text-transform: uppercase;
         letter-spacing: 1px; margin-bottom: 9px; }
    .card-value { font-weight: 800; font-size: 30px; line-height: 1.2; color: #fff; }
    .positive { color: #10b981 !important; }
    .negative { color: #ef4444 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
         border-right: 1px solid rgba(255,255,255,.06); }
    .stButton > button { background: linear-gradient(135deg, #6366f1 0%, #c084fc 100%);
         color: white; border: 0; border-radius: 10px; font-weight: 650; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,.025);
         border: 1px dashed rgba(255,255,255,.2); border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


BET_COLUMNS = [
    "Дата",
    "Время",
    "Результат",
    "Сумма_изменения",
    "Источник",
    "Карточка",
]


def empty_bets() -> pd.DataFrame:
    return pd.DataFrame(columns=BET_COLUMNS)


def normalize_bets(data: pd.DataFrame) -> pd.DataFrame:
    """Clean imported or edited records and derive the result from the amount."""
    if data is None or data.empty:
        return empty_bets()

    df = data.copy()
    for column in BET_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column != "Сумма_изменения" else 0.0

    df["Дата"] = df["Дата"].astype(str).str.strip()
    df["Время"] = df["Время"].astype(str).str.strip()
    df["Сумма_изменения"] = pd.to_numeric(df["Сумма_изменения"], errors="coerce")
    df = df.dropna(subset=["Сумма_изменения"])
    df = df[df["Дата"].str.match(r"^\d{2}\.\d{2}\.\d{4}$", na=False)]
    df = df[df["Время"].str.match(r"^\d{2}:\d{2}$", na=False)]

    df["Результат"] = df["Сумма_изменения"].apply(
        lambda value: "Выигрыш" if value > 0 else "Проигрыш" if value < 0 else "Возврат"
    )
    df["Источник"] = df["Источник"].fillna("").astype(str)
    df["Карточка"] = df["Карточка"].fillna("").astype(str)
    return df[BET_COLUMNS].reset_index(drop=True)


def fmt_money(value: float, decimals: int = 0) -> str:
    if decimals:
        text = f"{value:,.{decimals}f}"
        text = text.rstrip("0").rstrip(".")
    else:
        text = f"{value:,.0f}"
    return text.replace(",", " ")


def month_title(period: str) -> str:
    names = {
        1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
        5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
        9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
    }
    year, month = period.split("-")
    return f"{names[int(month)]} {year}"


# ------------------------------
# Состояние приложения
# ------------------------------
if "bets_df" not in st.session_state:
    st.session_state.bets_df = empty_bets()
if "pending_df" not in st.session_state:
    st.session_state.pending_df = None
if "pending_review" not in st.session_state:
    st.session_state.pending_review = []
if "pending_diagnostics" not in st.session_state:
    st.session_state.pending_diagnostics = []
if "pending_id" not in st.session_state:
    st.session_state.pending_id = str(uuid.uuid4())


st.markdown("# 💎 BetBoom Analytics")
st.caption(
    "В статистику попадают только карточки «Прогноз». Карточки «Транзакция» "
    "исключаются; выплаты по прогнозам («Пополнение») остаются выигрышами."
)


# ------------------------------
# Загрузка скриншотов
# ------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Управление")
    st.write("Загрузите оригинальные скриншоты истории баланса BetBoom.")
    uploaded_files = st.file_uploader(
        "Скриншоты истории",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    process_clicked = st.button(
        "🚀 Распознать скриншоты",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_clicked and uploaded_files:
        all_bets: list[dict] = []
        all_review: list[dict] = []
        all_diagnostics: list[dict] = []
        bar = st.progress(0.0)
        status = st.empty()
        file_count = len(uploaded_files)

        for file_index, uploaded in enumerate(uploaded_files):
            status.info(f"Обрабатываю {uploaded.name} ({file_index + 1}/{file_count})…")
            file_bytes = uploaded.getvalue()

            def update_card_progress(done: int, total: int, current_index: int = file_index) -> None:
                overall = (current_index + done / max(total, 1)) / file_count
                bar.progress(min(1.0, overall))

            parsed = process_image_bytes(
                uploaded.name,
                file_bytes,
                progress_callback=update_card_progress,
            )
            all_bets.extend(parsed["bets"])
            all_review.extend(parsed["review"])
            all_diagnostics.append(parsed["diagnostics"])
            bar.progress((file_index + 1) / file_count)

        status.empty()
        bar.empty()

        st.session_state.pending_df = normalize_bets(pd.DataFrame(all_bets))
        st.session_state.pending_review = all_review
        st.session_state.pending_diagnostics = all_diagnostics
        st.session_state.pending_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("### 💾 База данных")
    base_df = normalize_bets(st.session_state.bets_df)
    if not base_df.empty:
        csv_data = base_df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 Скачать базу CSV",
            data=csv_data,
            file_name="betboom_bets.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("🗑️ Очистить базу", use_container_width=True):
            st.session_state.bets_df = empty_bets()
            st.rerun()
        st.caption("Скачивайте CSV после обновлений: база хранится в текущей сессии.")
    else:
        st.caption("База пока пустая.")


# ------------------------------
# Проверка распознанных данных
# ------------------------------
if st.session_state.pending_df is not None:
    st.markdown("## Проверка распознавания")

    diagnostics_df = pd.DataFrame(st.session_state.pending_diagnostics)
    if not diagnostics_df.empty:
        display_diagnostics = diagnostics_df[
            [
                "file", "width", "height", "detected_cards", "recognized_bets",
                "ignored_transactions", "needs_review", "reason",
            ]
        ].rename(
            columns={
                "file": "Файл",
                "width": "Ширина",
                "height": "Высота",
                "detected_cards": "Карточек найдено",
                "recognized_bets": "Ставок распознано",
                "ignored_transactions": "Транзакций исключено",
                "needs_review": "Нужно проверить",
                "reason": "Комментарий",
            }
        )
        st.dataframe(display_diagnostics, use_container_width=True, hide_index=True)

    pending_df = st.session_state.pending_df.copy()
    if not pending_df.empty:
        st.write("Проверьте даты и суммы. Отметку «Учитывать» можно снять у ошибочных строк.")
        editable = pending_df.copy()
        editable.insert(0, "Учитывать", True)
        edited = st.data_editor(
            editable,
            key=f"pending_editor_{st.session_state.pending_id}",
            use_container_width=True,
            hide_index=True,
            disabled=["Результат", "Источник", "Карточка"],
            column_config={
                "Учитывать": st.column_config.CheckboxColumn("Учитывать", default=True),
                "Дата": st.column_config.TextColumn("Дата", help="Формат ДД.ММ.ГГГГ"),
                "Время": st.column_config.TextColumn("Время", help="Формат ЧЧ:ММ"),
                "Сумма_изменения": st.column_config.NumberColumn("Сумма, ₽", format="%.2f"),
            },
        )
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("✅ Добавить проверенные ставки", use_container_width=True):
                approved = edited[edited["Учитывать"]].drop(columns=["Учитывать"])
                approved = normalize_bets(approved)
                before_count = len(normalize_bets(st.session_state.bets_df))
                combined = pd.concat(
                    [normalize_bets(st.session_state.bets_df), approved],
                    ignore_index=True,
                )
                # Дедупликация нужна, когда загружаются перекрывающиеся скриншоты.
                # Ключ включает дату, время и изменение баланса.
                combined = combined.drop_duplicates(
                    subset=["Дата", "Время", "Сумма_изменения"],
                    keep="first",
                )
                st.session_state.bets_df = normalize_bets(combined)
                added_count = len(st.session_state.bets_df) - before_count
                st.session_state.pending_df = None
                st.session_state.pending_review = []
                st.session_state.pending_diagnostics = []
                st.success(f"Новых записей добавлено: {max(0, added_count)}. Всего в базе: {len(st.session_state.bets_df)}.")
                st.rerun()
        with col_cancel:
            if st.button("Отменить импорт", use_container_width=True):
                st.session_state.pending_df = None
                st.session_state.pending_review = []
                st.session_state.pending_diagnostics = []
                st.rerun()
    else:
        st.warning("Ни одной ставки не удалось распознать автоматически. Проверьте качество скриншотов.")
        if st.button("Закрыть проверку"):
            st.session_state.pending_df = None
            st.session_state.pending_review = []
            st.session_state.pending_diagnostics = []
            st.rerun()

    if st.session_state.pending_review:
        with st.expander(f"Карточки, требующие ручной проверки ({len(st.session_state.pending_review)})"):
            st.dataframe(
                pd.DataFrame(st.session_state.pending_review),
                use_container_width=True,
                hide_index=True,
            )


# ------------------------------
# Дашборд
# ------------------------------
df = normalize_bets(st.session_state.bets_df)
if df.empty:
    st.info("👈 Загрузите оригинальные скриншоты истории ставок в левой панели.")
    st.stop()

# Дата и время сортируются вместе, чтобы баланс строился в хронологическом порядке.
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

st.markdown("## Общая статистика")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="custom-card"><div class="card-title">Всего ставок</div>'
        f'<div class="card-value">{total_bets:,}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    wr_color = "positive" if winrate > 50 else "" if winrate >= 40 else "negative"
    st.markdown(
        f'<div class="custom-card"><div class="card-title">Винрейт</div>'
        f'<div class="card-value {wr_color}">{winrate:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    pnl_color = "positive" if total_pnl >= 0 else "negative"
    st.markdown(
        f'<div class="custom-card"><div class="card-title">Чистый P&amp;L</div>'
        f'<div class="card-value {pnl_color}">{fmt_money(total_pnl)} ₽</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f'<div class="custom-card"><div class="card-title">Средний проигрыш</div>'
        f'<div class="card-value">{fmt_money(avg_loss_size, 2)} ₽</div></div>',
        unsafe_allow_html=True,
    )

st.caption(
    f"Выигрышей: {len(wins_df)} · Проигрышей: {len(losses_df)} · "
    f"Возвратов: {len(pushes_df)} · Чистый результат: {fmt_money(total_pnl, 2)} ₽"
)

chart_tab, history_tab, stats_tab = st.tabs(
    ["📊 График баланса", "📝 История по месяцам", "🏆 Ключевые цифры"]
)

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
            fillcolor="rgba(167,139,250,0.12)",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", opacity=0.5)
    fig.update_layout(
        title="Динамика банкролла",
        xaxis_title="Номер ставки",
        yaxis_title="Изменение баланса, ₽",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True)

with history_tab:
    periods = sorted(df["Месяц"].dropna().unique(), reverse=True)
    if periods:
        selected_period = st.selectbox(
            "Месяц",
            periods,
            format_func=month_title,
        )
        month_df = df[df["Месяц"] == selected_period].copy()
        month_df["Сумма, ₽"] = month_df["Сумма_изменения"].map(lambda value: fmt_money(value, 2))
        st.dataframe(
            month_df[["Дата", "Время", "Результат", "Сумма, ₽", "Источник"]].sort_values(
                ["ДатаВремя"], ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Нет данных для отображения.")

with stats_tab:
    left, right = st.columns(2)
    with left:
        largest_win = float(wins_df["Сумма_изменения"].max()) if not wins_df.empty else 0.0
        total_won = float(wins_df["Сумма_изменения"].sum())
        avg_win = float(wins_df["Сумма_изменения"].mean()) if not wins_df.empty else 0.0
        st.metric("💎 Самый крупный выигрыш", f"+{fmt_money(largest_win, 2)} ₽")
        st.metric("📥 Сумма всех выигрышей", f"+{fmt_money(total_won, 2)} ₽")
        st.metric("📈 Средний выигрыш", f"+{fmt_money(avg_win, 2)} ₽")
    with right:
        largest_loss = float(losses_df["Сумма_изменения"].min()) if not losses_df.empty else 0.0
        total_lost = float(losses_df["Сумма_изменения"].sum())
        avg_loss = float(losses_df["Сумма_изменения"].mean()) if not losses_df.empty else 0.0
        st.metric("💀 Самый крупный проигрыш", f"{fmt_money(largest_loss, 2)} ₽")
        st.metric("📤 Сумма всех проигрышей", f"{fmt_money(total_lost, 2)} ₽")
        st.metric("📉 Средний проигрыш", f"{fmt_money(avg_loss, 2)} ₽")

    unique_days = df["ДатаВремя"].dt.date.nunique()
    common_loss = (
        abs(losses_df["Сумма_изменения"]).mode().iloc[0]
        if not losses_df.empty and not losses_df["Сумма_изменения"].abs().mode().empty
        else 0.0
    )
    extra1, extra2 = st.columns(2)
    extra1.metric("📅 Дней со ставками", unique_days)
    extra2.metric("🎯 Самая частая сумма проигрыша", f"{fmt_money(float(common_loss), 2)} ₽")
