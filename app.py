"""
TDA Market Monitor — сайт на Streamlit.
Запуск:  streamlit run app.py
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tda_core as tc

st.set_page_config(page_title="TDA Market Monitor", page_icon="📈", layout="wide")

# Цвета: ряд — синий; пороги — упорядоченная тёплая шкала + разный пунктир
SERIES_COLOR = "#2a78d6"
THR_STYLE = {
    1: ("#eda100", "dot"),
    2: ("#ec835a", "dash"),
    3: ("#d03b3b", "longdash"),
    4: ("#7a1c1c", "solid"),
}
METRIC_TITLES = {
    "H0_WD": "Wasserstein Distance — H₀ (компоненты связности)",
    "H1_WD": "Wasserstein Distance — H₁ (циклы)",
    "H0_L1": "Persistence Landscape, норма L₁ — H₀",
    "H0_L2": "Persistence Landscape, норма L₂ — H₀",
    "H1_L1": "Persistence Landscape, норма L₁ — H₁",
    "H1_L2": "Persistence Landscape, норма L₂ — H₁",
}


# -----------------------------------------------------------------------------
# Кэшируемые обёртки
# -----------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_fetch(ticker, source, start, end):
    df = tc.fetch_series(ticker, source, start, end)
    return df, df.attrs.get("source", source)


@st.cache_data(show_spinner=False)
def cached_compute(X, dates, window, homs):
    return tc.compute_all(X, pd.DatetimeIndex(dates), window=window, homs=homs)


# -----------------------------------------------------------------------------
# Боковая панель
# -----------------------------------------------------------------------------
st.sidebar.title("Параметры")

source_label = st.sidebar.radio(
    "Источник данных",
    ["Авто (WSJ → Yahoo)", "WSJ", "Yahoo Finance", "Загрузить файлы (CSV/XLSX)"],
    help="WSJ иногда блокирует запросы с серверов. Тогда сработает Yahoo, "
         "или загрузите выгрузку вручную (например, файлы Nasdaq Baltic).",
)
source = {"Авто (WSJ → Yahoo)": "auto", "WSJ": "wsj",
          "Yahoo Finance": "yahoo", "Загрузить файлы (CSV/XLSX)": "upload"}[source_label]

uploads = None
if source == "upload":
    uploads = st.sidebar.file_uploader(
        "Файлы с котировками (колонки Date + Close/Value)",
        type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    tickers_raw = ""
else:
    tickers_raw = st.sidebar.text_input(
        "Индекс(ы) через запятую",
        value="RUT",
        help="WSJ: RUT, SPX, DJIA, COMP или путь со страной, напр. HK/XHKG/HSI, "
             "либо полная ссылка на страницу Historical Prices. "
             "Yahoo: ^RUT, ^GSPC, ^STOXX50E …  Несколько индексов → многомерное "
             "облако точек, как в вашем коде для OMX Baltic.")

c1, c2 = st.sidebar.columns(2)
start = c1.date_input("С", value=date(2016, 1, 4), min_value=date(1990, 1, 1))
end = c2.date_input("По", value=date.today())
field = st.sidebar.selectbox("Цена", ["Close", "Open"], index=0)
window = st.sidebar.slider("Окно (торговых дней)", 20, 120, 50, 5)

st.sidebar.markdown("**Вложение Такенса** (только для одного индекса)")
e1, e2 = st.sidebar.columns(2)
embed_dim = e1.number_input("Размерность d", 2, 10, 4)
delay = e2.number_input("Задержка τ", 1, 10, 1)

homs = tuple(k for k, on in [(0, st.sidebar.checkbox("H₀", True)),
                             (1, st.sidebar.checkbox("H₁", True))] if on)
ks = st.sidebar.multiselect("Пороги μ + kσ", [1, 2, 3, 4], default=[1, 2, 3, 4])
thr_mode_label = st.sidebar.radio(
    "μ и σ считать по",
    ["всей выборке", "только прошлым данным (расширяющееся окно)"],
    help="«Вся выборка» — как в статье. «Прошлые данные» — честный режим для "
         "отслеживания в реальном времени: порог на дату t не знает будущего.")
thr_mode = "full" if thr_mode_label == "всей выборке" else "expanding"

run = st.sidebar.button("Построить", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Графики
# -----------------------------------------------------------------------------
def metric_figure(series: pd.Series, title: str, thr: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Значение",
                             line=dict(color=SERIES_COLOR, width=1.6),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:.6f}<extra></extra>"))
    for k in sorted(thr):
        color, dash = THR_STYLE[k]
        t = thr[k]
        fig.add_trace(go.Scatter(x=t.index, y=t.values, name=f"μ + {k}σ",
                                 line=dict(color=color, width=1.4, dash=dash),
                                 hovertemplate=f"μ+{k}σ: %{{y:.6f}}<extra></extra>"))
        last = t.dropna()
        if len(last):
            fig.add_annotation(x=last.index[-1], y=last.iloc[-1], text=f"+{k}σ",
                               showarrow=False, xanchor="left", xshift=4,
                               font=dict(size=11, color="#52514e"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=380, margin=dict(l=10, r=50, t=50, b=10),
        hovermode="x unified", plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False),
                   rangeselector=dict(buttons=[
                       dict(count=1, label="1г", step="year", stepmode="backward"),
                       dict(count=3, label="3г", step="year", stepmode="backward"),
                       dict(step="all", label="Всё")])),
        yaxis=dict(gridcolor="#e6e5e0", zeroline=False),
    )
    return fig


def status_level(value, thr_at_last: dict):
    crossed = [k for k in sorted(thr_at_last) if pd.notna(thr_at_last[k]) and value > thr_at_last[k]]
    return max(crossed) if crossed else 0


# -----------------------------------------------------------------------------
# Основная страница
# -----------------------------------------------------------------------------
st.title("📈 TDA Market Monitor")
st.caption("Wasserstein Distance и нормы L₁/L₂ persistence landscapes (H₀, H₁) "
           "с порогами μ + kσ. Введите индекс слева и нажмите «Построить».")

if not run and "result" not in st.session_state:
    st.info("Пример: `RUT` (Russell 2000) или несколько индексов: `SPX, DJIA, COMP, RUT`.")
    st.stop()

if run:
    if not homs:
        st.error("Выберите хотя бы одну гомологию (H₀ или H₁)."); st.stop()
    if start >= end:
        st.error("Дата начала должна быть раньше даты конца."); st.stop()

    series, sources = [], []
    with st.spinner("Загружаю котировки…"):
        if source == "upload":
            if not uploads:
                st.error("Загрузите хотя бы один файл."); st.stop()
            for f in uploads:
                try:
                    df = tc.load_uploaded(f, f.name)
                    name = f.name.rsplit(".", 1)[0]
                    col = field if field in df else "Close"
                    series.append(df[col].rename(name)); sources.append("файл")
                except Exception as e:  # noqa: BLE001
                    st.error(f"{f.name}: {e}")
        else:
            for t in [x.strip() for x in tickers_raw.split(",") if x.strip()]:
                try:
                    df, src = cached_fetch(t, source, start, end)
                    if field not in df or df[field].isna().all():
                        st.warning(f"{t}: нет колонки {field}, использую Close.")
                        col = "Close"
                    else:
                        col = field
                    series.append(df[col].rename(t.upper())); sources.append(src)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Не удалось загрузить {t}. {e}")
    if not series:
        st.stop()

    prices = pd.concat(series, axis=1, join="inner").sort_index()
    prices = prices.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
    prices = prices[(prices > 0).all(axis=1)]
    X, dates = tc.build_point_cloud(prices, int(embed_dim), int(delay))

    bar = st.progress(0.0, text="Считаю диаграммы персистентности…")
    try:
        wd_df, norms_df = cached_compute(X, np.asarray(dates), int(window), homs)
    except Exception as e:  # noqa: BLE001
        bar.empty(); st.error(str(e)); st.stop()
    bar.empty()
    st.session_state["result"] = dict(prices=prices, wd=wd_df, norms=norms_df,
                                      sources=sources, X_dim=X.shape[1], window=int(window),
                                      n_series=prices.shape[1])

res = st.session_state["result"]
prices, wd_df, norms_df = res["prices"], res["wd"], res["norms"]
all_metrics = pd.concat([wd_df, norms_df], axis=1)

cloud_note = (f"{res['n_series']} индекс(ов) → облако в ℝ^{res['X_dim']}"
              if res["n_series"] > 1 else
              f"1 индекс → вложение Такенса, облако в ℝ^{res['X_dim']}")
st.markdown(f"**Данные:** {', '.join(prices.columns)} · "
            f"{prices.index[0]:%d.%m.%Y} – {prices.index[-1]:%d.%m.%Y} · "
            f"{len(prices)} торговых дней · источник: {', '.join(sorted(set(res['sources'])))} · "
            f"{cloud_note}")

thr_all = {m: tc.thresholds(all_metrics[m].dropna(), ks or [1], thr_mode) for m in all_metrics}

# ---- Текущее состояние --------------------------------------------------------
st.subheader("Текущее состояние")
cols = st.columns(len(all_metrics.columns))
labels = {0: "норма", 1: "> μ+1σ", 2: "> μ+2σ", 3: "> μ+3σ", 4: "> μ+4σ"}
icons = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🟥"}
for col, m in zip(cols, all_metrics.columns):
    s = all_metrics[m].dropna()
    lvl = status_level(s.iloc[-1], {k: v.iloc[-1] for k, v in thr_all[m].items()})
    col.metric(m.replace("_", " "), f"{s.iloc[-1]:.5f}", f"{icons[lvl]} {labels[lvl]}",
               delta_color="off", help=f"на {s.index[-1]:%d.%m.%Y}")

tab_wd, tab_norm, tab_sig, tab_px, tab_data = st.tabs(
    ["Wasserstein", "Нормы L₁ / L₂", "Сигналы", "Котировки", "Данные"])

with tab_wd:
    for m in wd_df.columns:
        st.plotly_chart(metric_figure(wd_df[m].dropna(), f"{METRIC_TITLES[m]}, окно {res['window']} дн.",
                                      thr_all[m]), use_container_width=True)
    st.caption("Сравниваются два соседних непересекающихся окна: [t−2w+1, t−w] и [t−w+1, t].")

with tab_norm:
    for m in norms_df.columns:
        st.plotly_chart(metric_figure(norms_df[m].dropna(), METRIC_TITLES[m], thr_all[m]),
                        use_container_width=True)
    if res["n_series"] == 1 and res["X_dim"] == 1:
        st.warning("Облако одномерное — H₁ всегда пусто. Увеличьте размерность Такенса.")

with tab_sig:
    st.markdown("Периоды, когда показатель был **выше** порога.")
    m_sel = st.selectbox("Показатель", list(all_metrics.columns),
                         format_func=lambda m: METRIC_TITLES[m])
    k_sel = st.selectbox("Порог", sorted(thr_all[m_sel]), format_func=lambda k: f"μ + {k}σ",
                         index=min(1, len(thr_all[m_sel]) - 1))
    s = all_metrics[m_sel].dropna()
    ep = tc.episodes(s, thr_all[m_sel][k_sel])
    if ep:
        ep_df = pd.DataFrame(ep, columns=["Начало", "Конец", "Дней", "Пик"])
        ep_df["Начало"] = ep_df["Начало"].dt.strftime("%d.%m.%Y")
        ep_df["Конец"] = ep_df["Конец"].dt.strftime("%d.%m.%Y")
        st.dataframe(ep_df.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.success("Превышений нет.")
    summary = pd.DataFrame({
        f"μ+{k}σ": {m: int((all_metrics[m].dropna() > thr_all[m][k]).sum()) for m in all_metrics}
        for k in sorted(ks or [1])})
    st.markdown("**Число дней выше порога**")
    st.dataframe(summary, use_container_width=True)

with tab_px:
    norm_px = prices / prices.iloc[0] * 100
    fig = go.Figure()
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for i, c in enumerate(norm_px.columns):
        fig.add_trace(go.Scatter(x=norm_px.index, y=norm_px[c], name=c,
                                 line=dict(color=palette[i % 8], width=1.6)))
    fig.update_layout(title="Котировки (начало периода = 100)", height=400,
                      hovermode="x unified", plot_bgcolor="#fcfcfb",
                      margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(gridcolor="#e6e5e0"), xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    out = all_metrics.copy()
    for m in all_metrics:
        for k, t in thr_all[m].items():
            out[f"{m}_thr{k}s"] = t
    st.dataframe(out.round(8), use_container_width=True)
    st.download_button("Скачать CSV", out.to_csv().encode("utf-8"),
                       file_name="tda_metrics.csv", mime="text/csv")
