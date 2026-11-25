from __future__ import annotations

from datetime import datetime
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Импортируем ваш новый OCR-модуль
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

    /* Карточки метрик */
    .custom-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.6s ease-out forwards;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .custom-card:hover { transform: translateY(-5px) !important; box-shadow: 0 8px 40px rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.5); background: rgba(255, 255, 255, 0.08); }
    .card-title { color: #94a3b8; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .card-value { font-weight: 800; font-size: 32px; line-height: 1.2; color: #ffffff; }
    .card-delta { display: inline-block; padding: 4px 10px; border-radius: 8px; font-weight: 600; font-size: 14px; margin-top: 8px; width: fit-content; }
    .pnl-delta.plus { color: #10b981; background-color: rgba(16, 185, 129, 0.1); }
    .pnl-delta.minus { color: #ef4444; background-color: rgba(239, 68, 68, 0.1); }
    .winrate-green { color: #10b981 !important; }
    .winrate-yellow { color: #f59e0b !important; }
    .winrate-red { color: #ef4444 !important; }
    .card-left { margin-top: 25px; opacity: 0.85; }
    .card-right { margin-top: 50px; opacity: 0.7; }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; justify-content: center; background: rgba(255, 255, 255, 0.02); padding: 10px 20px; border-radius: 50px; width: fit-content; margin: 0 auto; border: 1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(5px); }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: transparent; border-radius: 30px; color: #94a3b8; font-weight: 600; font-size: 15px; padding: 0 24px; transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); border: none; }
    .stTabs [data-baseweb="tab"]:hover { color: #e2e8f0; background-color: rgba(255, 255, 255, 0.05); }
    .stTabs [aria-selected="true"] { color: #ffffff !important; background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(192, 132, 252, 0.8) 100%) !important; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2); transform: scale(1.05); border-radius: 30px; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Таблица */
    .custom-table { width: 100%; border-collapse: collapse; margin-top: 10px; background: rgba(255, 255, 255, 0.02); border-radius: 12px; overflow: hidden; }
    .custom-table th, .custom-table td { padding: 12px 15px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
    .custom-table th { background: rgba(255, 255, 255, 0.05); color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }
    .custom-table tr:hover { background: rgba(255, 255, 255, 0.05); }
    .win-text { color: #10b981; font-weight: bold; }
    .loss-text { color: #ef4444; font-weight: bold; }

    .streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.05) !important; border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; font-size: 14px !important; color: #94a3b8 !important; margin-top: 5px; }
    .streamlit-expanderHeader:hover { background-color: rgba(255, 255, 255, 0.1) !important; color: #ffffff !important; }
    .streamlit-expanderContent { background-color: rgba(0, 0, 0, 0.2) !important; border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; color: #e2e8f0 !important; }

    .btn-danger > button {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.8) 0%, rgba(220, 38, 38, 0.8) 100%) !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    .btn-danger > button:hover {
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Шапка сайта
st.markdown(
    """
    <div class="app-brand">
        <div class="brand-gem">💎</div>
        <h1 style="margin:0;padding:0;">BetBoom Analytics</h1>
    </div>
    <div class="app-subtitle">Личный дашборд истории ставок</div>
    """,
    unsafe_allow_html=True
)

# 🪄 ФУНКЦИЯ КРАСИВОГО ОТОБРАЖЕНИЯ ДЕНЕГ (с копейками, если они есть)
def format_money(val: float) -> str:
    if val == int(val):
        return f"{int(val):,} ₽".replace(",", " ")
    else:
        return f"{val:,.2f} ₽".replace(",", " ").replace(".00", "")

# -----------------------------------------------------------------------------
# Боковая панель
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Управление")
    st.markdown("Загрузите скриншоты, чтобы обновить статистику.")
    
    uploaded_files = st.file_uploader("Выберите скриншоты", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("🚀 Обработать скриншоты"):
        if uploaded_files:
            with st.spinner('Нейросеть читает ваши ставки...'):
                all_new_bets = []
                diagnostics = []
                
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    res = process_image_bytes(uploaded_file.name, file_bytes)
                    all_new_bets.extend(res["bets"])
                    diagnostics.append(res["diagnostics"])
                
                if all_new_bets:
                    new_df = pd.DataFrame(all_new_bets)
                    if 'df' not in st.session_state:
                        st.session_state.df = new_df
                    else:
                        combined = pd.concat([st.session_state.df, new_df])
                        st.session_state.df = combined.drop_duplicates(subset=['Дата', 'Время', 'Сумма_изменения'])
                    st.success(f"Данные обновлены! Распознано новых ставок: {len(new_df)}")
                else:
                    st.error("Не удалось распознать ставки.")
                    for diag in diagnostics:
                        if diag.get("reason"):
                            st.warning(f"{diag['file']}: {diag['reason']}")
        else:
            st.warning("Пожалуйста, выберите файлы.")

    st.markdown("---")
    st.markdown("### 💾 База данных")
    
    if 'df' in st.session_state and not st.session_state.df.empty:
        csv = st.session_state.df.to_csv(sep=';', index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Скачать базу (CSV)", 
            data=csv, 
            file_name='bets_all.csv', 
            mime='text/csv'
        )
        
        st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
        if st.button("🗑️ Очистить все данные"):
            st.session_state.df = pd.DataFrame()
            st.success("База данных очищена!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption("Не забудьте скачать CSV, чтобы не потерять данные.")

# -----------------------------------------------------------------------------
# Основная логика
# -----------------------------------------------------------------------------
if 'df' in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    df['Дата_парсинг'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y', errors='coerce')
    df['Время_парсинг'] = pd.to_datetime(df['Время'], format='%H:%M', errors='coerce').dt.time
    df = df.sort_values(by=['Дата_парсинг', 'Время_парсинг'], ascending=[False, False])
    df['Месяц'] = df['Дата_парсинг'].dt.strftime('%B %Y')

    df_rev = df.iloc[::-1].reset_index(drop=True)
    df_rev['Баланс'] = df_rev['Сумма_изменения'].cumsum()
    df_rev['Номер ставки'] = df_rev.index + 1

    # ВЕРХНИЕ 4 КАРТОЧКИ
    col1, col2, col3, col4 = st.columns(4)
    
    total_bets = len(df)
    wins = len(df[df['Результат'] == 'Выигрыш'])
    winrate = (wins / total_bets) * 100 if total_bets > 0 else 0
    total_pnl = df['Сумма_изменения'].sum()
    avg_bet = abs(df[df['Сумма_изменения'] < 0]['Сумма_изменения'].mean()) if not df[df['Сумма_изменения'] < 0].empty else 0

    with col1:
        st.markdown(f'<div class="custom-card card-left"><div class="card-title">Всего ставок</div><div class="card-value">{total_bets:,}</div></div>', unsafe_allow_html=True)
    
    with col2:
        if winrate > 50: wr_class = "winrate-green"
        elif winrate >= 40: wr_class = "winrate-yellow"
        else: wr_class = "winrate-red"
        st.markdown(f'<div class="custom-card"><div class="card-title">Винрейт</div><div class="card-value {wr_class}">{winrate:.1f}%</div></div>', unsafe_allow_html=True)
    
    with col3:
        if total_pnl > 0: delta_class, delta_sign = "plus", "+"
        elif total_pnl < 0: delta_class, delta_sign = "minus", ""
        else: delta_class, delta_sign = "plus", ""
        st.markdown(f'<div class="custom-card"><div class="card-title">Чистый P&L</div><div class="card-value">{format_money(total_pnl)}</div><div class="card-delta pnl-delta {delta_class}">{delta_sign}{format_money(total_pnl)}</div></div>', unsafe_allow_html=True)
                
    with col4:
        st.markdown(f'<div class="custom-card card-right"><div class="card-title">Средняя ставка</div><div class="card-value">{format_money(avg_bet)}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ВКЛАДКИ
    tab1, tab2, tab3 = st.tabs(["📊 График баланса", "📝 История по месяцам", "🏆 Ключевые цифры"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_rev['Номер ставки'], y=df_rev['Баланс'], mode='lines', name='Баланс', 
            line=dict(color='#818cf8', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(129, 140, 248, 0.1)'
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", opacity=0.5)
        fig.update_layout(
            title=dict(text="Динамика банкролла", font=dict(color='#f8fafc', size=20)), 
            xaxis=dict(title="Номер ставки", gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'), 
            yaxis=dict(title="Прибыль/Убыток (₽)", gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'), 
            hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), 
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        months = df['Месяц'].dropna().unique()
        if len(months) > 0:
            month_tabs = st.tabs([f"📅 {m}" for m in months])
            for m_tab, month_name in zip(month_tabs, months):
                with m_tab:
                    df_month = df[df['Месяц'] == month_name]
                    html_table = '<table class="custom-table"><thead><tr><th>Дата</th><th>Время</th><th>Результат</th><th>Сумма (₽)</th></tr></thead><tbody>'
                    for _, row in df_month.iterrows():
                        result_html = f'<span class="win-text">{row["Результат"]}</span>' if row["Результат"] == 'Выигрыш' else f'<span class="loss-text">{row["Результат"]}</span>'
                        amount_class = "win-text" if row["Сумма_изменения"] > 0 else "loss-text"
                        html_table += f'<tr><td>{row["Дата"]}</td><td>{row["Время"]}</td><td>{result_html}</td><td><span class="{amount_class}">{format_money(row["Сумма_изменения"])}</span></td></tr>'
                    html_table += '</tbody></table>'
                    st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("Нет данных для отображения.")

    with tab3:
        st.markdown("### Детальная статистика")
        wins_df = df[df['Результат'] == 'Выигрыш']
        losses_df = df[df['Результат'] == 'Проигрыш']
        
        max_win_row = wins_df.loc[wins_df['Сумма_изменения'].idxmax()] if not wins_df.empty else None
        max_loss_row = losses_df.loc[losses_df['Сумма_изменения'].idxmin()] if not losses_df.empty else None
        
        c1, c2 = st.columns(2)
        with c1:
            if max_win_row is not None:
                st.markdown(f'<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">💎 Самый крупный выигрыш</div><div class="card-value winrate-green">+{format_money(max_win_row["Сумма_изменения"])}</div></div>', unsafe_allow_html=True)
                with st.expander("Показать детали"):
                    st.write(f"📅 **Дата:** {max_win_row['Дата']}"); st.write(f"⏰ **Время:** {max_win_row['Время']}")
            
            st.markdown(f'<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">📥 Сумма всех выигрышей</div><div class="card-value winrate-green">+{format_money(wins_df["Сумма_изменения"].sum())}</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"🎯 **Количество выигрышных ставок:** {len(wins_df)}")
                st.write(f"📈 **Средний выигрыш:** +{format_money(wins_df['Сумма_изменения'].mean())}")
            
            unique_days = df['Дата_парсинг'].nunique()
            st.markdown(f'<div class="custom-card" style="margin-top: 10px;"><div class="card-title">📅 Дней со ставками</div><div class="card-value" style="color: #818cf8;">{unique_days}</div></div>', unsafe_allow_html=True)
            
        with c2:
            if max_loss_row is not None:
                st.markdown(f'<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">💀 Самый крупный проигрыш</div><div class="card-value winrate-red">{format_money(max_loss_row["Сумма_изменения"])}</div></div>', unsafe_allow_html=True)
                with st.expander("Показать детали"):
                    st.write(f"📅 **Дата:** {max_loss_row['Дата']}"); st.write(f"⏰ **Время:** {max_loss_row['Время']}")
            
            st.markdown(f'<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">📤 Сумма всех проигрышей</div><div class="card-value winrate-red">{format_money(losses_df["Сумма_изменения"].sum())}</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"🎯 **Количество проигрышных ставок:** {len(losses_df)}")
                st.write(f"📉 **Средний проигрыш:** {format_money(losses_df['Сумма_изменения'].mean())}")
            
            most_common_bet = abs(df['Сумма_изменения'].mode()[0]) if not df.empty else 0
            st.markdown(f'<div class="custom-card" style="margin-top: 10px;"><div class="card-title">🎯 Самая частая ставка</div><div class="card-value" style="color: #c084fc;">{format_money(most_common_bet)}</div></div>', unsafe_allow_html=True)

else:
    st.info("👈 Добро пожаловать! Чтобы начать, загрузите скриншоты истории ставок в левой панели.")
