import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
import pytesseract
import numpy as np
import re

# 🪄 ВАЖНО: Указываем путь к Tesseract для облачной среды Linux
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Настройка страницы
st.set_page_config(page_title="BetBoom Analytics", page_icon="💎", layout="wide")

# ==========================================
# 🪄 МАГИЯ CSS: СОВРЕМЕННЫЙ ДИЗАЙН
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    /* --- КАРТОЧКИ --- */
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

    /* --- РАСКРЫВАЮЩИЕСЯ ОКНА --- */
    .streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.05) !important; border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; font-size: 14px !important; color: #94a3b8 !important; margin-top: 5px; }
    .streamlit-expanderHeader:hover { background-color: rgba(255, 255, 255, 0.1) !important; color: #ffffff !important; }
    .streamlit-expanderContent { background-color: rgba(0, 0, 0, 0.2) !important; border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.05) !important; color: #e2e8f0 !important; }

    h1 { background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; text-align: center; padding-bottom: 20px; animation: fadeInUp 0.5s ease-out; }
    
    /* --- ВКЛАДКИ --- */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; justify-content: center; background: rgba(255, 255, 255, 0.02); padding: 10px 20px; border-radius: 50px; width: fit-content; margin: 0 auto; border: 1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(5px); }
    .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: transparent; border-radius: 30px; color: #94a3b8; font-weight: 600; font-size: 15px; padding: 0 24px; transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1); border: none; }
    .stTabs [data-baseweb="tab"]:hover { color: #e2e8f0; background-color: rgba(255, 255, 255, 0.05); }
    .stTabs [aria-selected="true"] { color: #ffffff !important; background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(192, 132, 252, 0.8) 100%) !important; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.2); transform: scale(1.05); border-radius: 30px; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* --- ТАБЛИЦА --- */
    .custom-table { width: 100%; border-collapse: collapse; margin-top: 10px; background: rgba(255, 255, 255, 0.02); border-radius: 12px; overflow: hidden; }
    .custom-table th, .custom-table td { padding: 12px 15px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }
    .custom-table th { background: rgba(255, 255, 255, 0.05); color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }
    .custom-table tr:hover { background: rgba(255, 255, 255, 0.05); }
    .win-text { color: #10b981; font-weight: bold; }
    .loss-text { color: #ef4444; font-weight: bold; }

    /* --- ДИЗАЙН БОКОВОЙ ПАНЕЛИ --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton > button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(192, 132, 252, 0.8) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        filter: brightness(1.1);
    }
    .btn-danger > button {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.8) 0%, rgba(220, 38, 38, 0.8) 100%) !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    .btn-danger > button:hover {
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4);
    }
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 ФУНКЦИЯ ОБРАБОТКИ СКРИНШОТОВ (OCR)
# ==========================================
def process_uploaded_images(uploaded_files):
    all_bets_data = []
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_logs = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Обработка файла {idx+1} из {total_files}...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None: 
            error_logs.append(f"Файл {uploaded_file.name}: не удалось прочитать изображение.")
            continue

        h, w = img.shape[:2]
        
        # Нарезка на куски по 4000 пикселей с перекрытием 600
        max_height = 4000
        overlap = 600
        
        chunks = []
        y = 0
        while y < h:
            y2 = min(y + max_height, h)
            chunks.append(img[y:y2, :])
            if y2 == h: break
            y = y2 - overlap

        current_date = None
        current_time = None

        for chunk in chunks:
            gray = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            try:
                text = pytesseract.image_to_string(gray, lang='rus+eng', config='--psm 6')
            except Exception as e:
                error_logs.append(f"Ошибка Tesseract: {str(e)}")
                continue

            lines = text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                lower_line = line.lower()

                # 🪄 Контекстный фильтр транзакций (смотрим на 3 строки назад и 1 вперед)
                context = " ".join(lines[max(0, i-3):min(len(lines), i+2)]).lower()
                is_transaction = 'транзакция' in context and 'прогноз' not in context

                # 1. Поиск даты
                date_match = re.search(r'(\d{1,2}\s+(?:янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)[а-яА-Я]*|сегодня|вчера)', lower_line)
                if date_match:
                    d_str = date_match.group(1).capitalize()
                    if d_str.lower() == 'сегодня':
                        current_date = datetime.now().strftime('%d.%m.%Y')
                    elif d_str.lower() == 'вчера':
                        current_date = (datetime.now() - timedelta(days=1)).strftime('%d.%m.%Y')
                    else:
                        current_date = d_str

                # 2. Поиск времени
                time_match = re.search(r'(\d{1,2}[:\.]\d{2})', line)
                if time_match:
                    current_time = time_match.group(1).replace('.', ':')

                # 3. Поиск суммы (Строгий парсинг: берет только целые числа, игнорируя мусор и копейки)
                amount_match = re.search(r'([+-])\s*(\d+(?:[\s.,]\d{3})*)', line)
                
                if amount_match and current_date and current_time:
                    if is_transaction:
                        current_time = None
                        continue

                    sign = amount_match.group(1)
                    # Очищаем строку от пробелов, точек и запятых, оставляя только цифры
                    amount_str = amount_match.group(2).replace(' ', '').replace(',', '').replace('.', '')
                    
                    try:
                        amount_val = int(amount_str)
                        if sign == '-':
                            amount_val = -amount_val
                        
                        new_bet = {
                            "Дата": current_date.strip(), 
                            "Время": current_time.strip(), 
                            "Результат": "Выигрыш" if amount_val > 0 else "Проигрыш", 
                            "Сумма_изменения": amount_val
                        }
                        
                        if not all_bets_data or all_bets_data[-1] != new_bet:
                            all_bets_data.append(new_bet)
                            
                        current_time = None 
                    except ValueError: 
                        pass

        progress_bar.progress((idx + 1) / total_files)

    status_text.empty()
    progress_bar.empty()
    
    df_res = pd.DataFrame(all_bets_data)
    
    if df_res.empty:
        return df_res, error_logs
        
    return df_res, []

# ==========================================
# 📊 ОСНОВНАЯ ЛОГИКА
# ==========================================
st.markdown("<h1>💎 BetBoom Analytics</h1>", unsafe_allow_html=True)

# Боковая панель для загрузки данных
with st.sidebar:
    st.markdown("### ⚙️ Управление")
    st.markdown("Загрузите скриншоты, чтобы обновить статистику.")
    
    uploaded_files = st.file_uploader("Выберите скриншоты", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("🚀 Обработать скриншоты"):
        if uploaded_files:
            with st.spinner('Нейросеть читает ваши ставки... Это займет около 10-20 секунд...'):
                new_df, errors = process_uploaded_images(uploaded_files)
                if not new_df.empty:
                    if 'df' not in st.session_state:
                        st.session_state.df = new_df
                    else:
                        combined = pd.concat([st.session_state.df, new_df])
                        st.session_state.df = combined.reset_index(drop=True)
                    st.success(f"Данные обновлены! Распознано ставок: {len(new_df)}")
                    st.rerun()
                else:
                    st.error("Не удалось распознать ставки.")
                    if errors:
                        st.warning("Технические детали:")
                        for err in errors:
                            st.text(err)
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
            st.success("База данных очищена! Загрузите новые скриншоты.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Если данные есть в памяти, показываем дашборд
if 'df' in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    current_year = datetime.now().year 
    
    def clean_date(date_str):
        date_str = str(date_str).lower().strip()
        if re.match(r'\d{2}\.\d{2}\.\d{4}', date_str): 
            return date_str
            
        month_map = {
            'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04', 'ма[йя]': '05',
            'июн': '06', 'июл': '07', 'авг': '08', 'сен': '09', 'окт': '10',
            'ноя': '11', 'дек': '12'
        }
        
        for prefix, num in month_map.items():
            if prefix in date_str:
                day_match = re.search(r'\d{1,2}', date_str)
                if day_match:
                    day = day_match.group(0)
                    return f"{day.zfill(2)}.{num}.{current_year}"
        return date_str

    df['Дата_парсинг'] = df['Дата'].apply(clean_date)
    df['Дата_парсинг'] = pd.to_datetime(df['Дата_парсинг'], format='%d.%m.%Y', errors='coerce')
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
        st.markdown(f'<div class="custom-card"><div class="card-title">Чистый P&L</div><div class="card-value">{total_pnl:,.0f} ₽</div><div class="card-delta pnl-delta {delta_class}">{delta_sign}{total_pnl:,.0f} ₽</div></div>', unsafe_allow_html=True)
                
    with col4:
        st.markdown(f'<div class="custom-card card-right"><div class="card-title">Средняя ставка</div><div class="card-value">{avg_bet:,.0f} ₽</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ВКЛАДКИ
    tab1, tab2, tab3 = st.tabs(["📊 График баланса", "📝 История по месяцам", "🏆 Ключевые цифры"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_rev['Номер ставки'], y=df_rev['Баланс'], mode='lines', name='Баланс', line=dict(color='#818cf8', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(129, 140, 248, 0.1)'))
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", opacity=0.5)
        fig.update_layout(title=dict(text="Динамика банкролла", font=dict(color='#f8fafc', size=20)), xaxis=dict(title="Номер ставки", gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'), yaxis=dict(title="Прибыль/Убыток (₽)", gridcolor='rgba(255,255,255,0.05)', color='#94a3b8'), hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif"))
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
                        html_table += f'<tr><td>{row["Дата"]}</td><td>{row["Время"]}</td><td>{result_html}</td><td><span class="{amount_class}">{row["Сумма_изменения"]:,.0f} ₽</span></td></tr>'
                    html_table += '</tbody></table>'
                    st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("Нет данных для отображения.")

    with tab3:
        st.markdown("### Детальная статистика")
        wins_df = df[df['Результат'] == 'Выигрыш']
        losses_df = df[df['Результат'] == 'Проигрыш']
        
        max_win, max_win_date, max_win_time = (wins_df.loc[wins_df['Сумма_изменения'].idxmax()]['Сумма_изменения'], wins_df.loc[wins_df['Сумма_изменения'].idxmax()]['Дата'], wins_df.loc[wins_df['Сумма_изменения'].idxmax()]['Время']) if not wins_df.empty else (0, "-", "-")
        max_loss, max_loss_date, max_loss_time = (losses_df.loc[losses_df['Сумма_изменения'].idxmin()]['Сумма_изменения'], losses_df.loc[losses_df['Сумма_изменения'].idxmin()]['Дата'], losses_df.loc[losses_df['Сумма_изменения'].idxmin()]['Время']) if not losses_df.empty else (0, "-", "-")
        
        total_won_sum, total_lost_sum = wins_df['Сумма_изменения'].sum(), losses_df['Сумма_изменения'].sum()
        unique_days = df['Дата_парсинг'].nunique()
        most_common_bet = abs(df['Сумма_изменения'].mode()[0]) if not df.empty else 0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">💎 Самый крупный выигрыш</div><div class="card-value winrate-green">+' + f'{max_win:,.0f} ₽</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"📅 **Дата:** {max_win_date}"); st.write(f"⏰ **Время:** {max_win_time}")
            st.markdown('<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">📥 Сумма всех выигрышей</div><div class="card-value winrate-green">+' + f'{total_won_sum:,.0f} ₽</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"🎯 **Количество выигрышных ставок:** {len(wins_df)}"); st.write(f"📈 **Средний выигрыш:** +{wins_df['Сумма_изменения'].mean():,.0f} ₽")
            st.markdown(f'<div class="custom-card" style="margin-top: 10px;"><div class="card-title">📅 Дней со ставками</div><div class="card-value" style="color: #818cf8;">{unique_days}</div></div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">💀 Самый крупный проигрыш</div><div class="card-value winrate-red">' + f'{max_loss:,.0f} ₽</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"📅 **Дата:** {max_loss_date}"); st.write(f"⏰ **Время:** {max_loss_time}")
            st.markdown('<div class="custom-card" style="margin-bottom: -10px;"><div class="card-title">📤 Сумма всех проигрышей</div><div class="card-value winrate-red">' + f'{total_lost_sum:,.0f} ₽</div></div>', unsafe_allow_html=True)
            with st.expander("Показать детали"):
                st.write(f"🎯 **Количество проигрышных ставок:** {len(losses_df)}"); st.write(f"📉 **Средний проигрыш:** {losses_df['Сумма_изменения'].mean():,.0f} ₽")
            st.markdown(f'<div class="custom-card" style="margin-top: 10px;"><div class="card-title">🎯 Самая частая ставка</div><div class="card-value" style="color: #c084fc;">{most_common_bet:,.0f} ₽</div></div>', unsafe_allow_html=True)

else:
    st.info("👈 Добро пожаловать! Чтобы начать, загрузите скриншоты истории ставок в левой панели.")
