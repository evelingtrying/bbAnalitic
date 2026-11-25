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
