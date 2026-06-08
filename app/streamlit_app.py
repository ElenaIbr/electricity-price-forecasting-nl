from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_URL = os.getenv("FORECAST_API_URL", "http://127.0.0.1:8000")
LOCAL_TZ = "Europe/Amsterdam"

ACTUAL_COLOR = "#1f77b4"
FORECAST_COLOR = "#9e9e9e"


st.set_page_config(
    page_title="NL Электроэнергия · Day-Ahead",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.7rem; }
[data-testid="stMetricLabel"] { color: #555; font-weight: 500; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Цена электроэнергии: Day-Ahead рынок NL")
st.caption("Почасовая стоимость электроэнергии: факт и прогноз.")


def to_local(ts):
    if hasattr(ts, "tz_convert"):
        return ts.tz_convert(LOCAL_TZ)
    return ts


@st.cache_data(show_spinner="Загружаем прогноз из API...", ttl=300)
def fetch_forecast(target_date: date | None = None) -> dict:
    params = {}

    if target_date is not None:
        params["target_date"] = target_date.isoformat()

    response = requests.get(
        f"{API_BASE_URL}/forecast",
        params=params,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Forecast API error {response.status_code}: {response.text}")

    return response.json()


@st.cache_data(show_spinner="Загружаем фактические цены из API...", ttl=300)
def fetch_actuals(target_date: date | None = None) -> dict:
    params = {}

    if target_date is not None:
        params["target_date"] = target_date.isoformat()

    response = requests.get(
        f"{API_BASE_URL}/actuals",
        params=params,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Actuals API error {response.status_code}: {response.text}")

    return response.json()


def forecast_to_series(payload: dict) -> pd.Series:
    hourly = payload.get("hourly", [])

    if not hourly:
        return pd.Series(dtype=float, name="predicted_price")

    rows = []
    for item in hourly:
        rows.append({
            "timestamp": pd.to_datetime(item["timestamp"], utc=True),
            "predicted_price": float(item["predicted_price"]),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df["predicted_price"]


def actuals_to_series(payload: dict) -> pd.Series:
    hourly = payload.get("hourly", [])

    if not hourly:
        return pd.Series(dtype=float, name="actual_price")

    rows = []
    for item in hourly:
        rows.append({
            "timestamp": pd.to_datetime(item["timestamp"], utc=True),
            "actual_price": float(item["actual_price"]),
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df["actual_price"]


with st.sidebar:
    st.header("Прогноз")

    use_custom_date = st.checkbox("Выбрать дату вручную", value=False)

    target_input = None
    if use_custom_date:
        target_input = st.date_input(
            "Дата",
            value=pd.Timestamp.now(tz=LOCAL_TZ).date(),
            format="DD.MM.YYYY",
        )

    refresh = st.button("🔄 Обновить данные", use_container_width=True)

    if refresh:
        fetch_forecast.clear()
        fetch_actuals.clear()

    st.divider()
    st.caption(f"Forecast API: `{API_BASE_URL}/forecast`")
    st.caption(f"Actuals API: `{API_BASE_URL}/market/actuals`")


try:
    forecast_payload = fetch_forecast(target_input)
    actuals_payload = {"hourly": [], "target_date": forecast_payload.get("target_date")}
except Exception as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()


forecast = forecast_to_series(forecast_payload)
actuals = actuals_to_series(actuals_payload)

target_date = forecast_payload.get("target_date") or actuals_payload.get("target_date")
model_version = forecast_payload.get("model_version")
forecast_made_at = forecast_payload.get("forecast_made_at")

if forecast.empty and actuals.empty:
    st.warning("API не вернул ни факт, ни прогноз для выбранной даты.")
    st.stop()


target_date_str = pd.Timestamp(target_date).strftime("%d.%m.%Y")

caption_parts = [f"Дата: **{target_date_str}**"]

if model_version:
    caption_parts.append(f"Модель: **{model_version}**")

if forecast_made_at:
    caption_parts.append(
        f"Прогноз сформирован: **{pd.to_datetime(forecast_made_at).strftime('%d.%m.%Y %H:%M UTC')}**"
    )

st.caption(" · ".join(caption_parts))

source_for_kpi = None
values_for_kpi = None

if not actuals.empty:
    source_for_kpi = "факт"
    values_for_kpi = actuals
elif not forecast.empty:
    source_for_kpi = "прогноз"
    values_for_kpi = forecast

c1, c2, c3 = st.columns(3)

if values_for_kpi is not None:
    c1.metric(f"Среднее ({source_for_kpi})", f"{values_for_kpi.mean():.2f} €/MWh")
    c2.metric(
        f"Максимум ({source_for_kpi})",
        f"{values_for_kpi.max():.2f} €/MWh",
        help=f"В {to_local(values_for_kpi.idxmax()).strftime('%H:%M')} Amsterdam",
    )
    c3.metric(
        f"Минимум ({source_for_kpi})",
        f"{values_for_kpi.min():.2f} €/MWh",
        help=f"В {to_local(values_for_kpi.idxmin()).strftime('%H:%M')} Amsterdam",
    )
else:
    c1.metric("Среднее", "—")
    c2.metric("Максимум", "—")
    c3.metric("Минимум", "—")


# График
fig = go.Figure()

if not actuals.empty:
    fig.add_trace(go.Scatter(
        x=to_local(actuals.index),
        y=actuals.values,
        mode="lines+markers",
        name="Факт",
        line=dict(color=ACTUAL_COLOR, width=3),
        marker=dict(size=6, color=ACTUAL_COLOR),
        hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.2f}</b> €/MWh (факт)<extra></extra>",
    ))

if not forecast.empty:
    fig.add_trace(go.Scatter(
        x=to_local(forecast.index),
        y=forecast.values,
        mode="lines+markers",
        name="Прогноз",
        line=dict(color=FORECAST_COLOR, width=3, dash="dot"),
        marker=dict(size=6, color=FORECAST_COLOR),
        hovertemplate="%{x|%d.%m %H:%M}<br><b>%{y:.2f}</b> €/MWh (прогноз)<extra></extra>",
    ))

fig.update_layout(
    xaxis_title=None,
    yaxis_title="€ / MWh",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    margin=dict(l=40, r=20, t=20, b=60),
    height=440,
    plot_bgcolor="white",
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    griddash="dot",
    dtick=6 * 3600 * 1000,
    tickformat="%H:%M\n%d.%m",
    ticks="outside",
    ticklen=4,
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    griddash="dot",
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor="rgba(0,0,0,0.2)",
)

st.plotly_chart(fig, use_container_width=True)


# Таблица
st.markdown("#### Почасовые значения")

table_rows = []

for ts, value in actuals.items():
    ts_local = to_local(ts)
    table_rows.append({
        "Дата": ts_local.strftime("%d.%m.%Y"),
        "Час": ts_local.strftime("%H:%M"),
        "Цена €/MWh": f"{value:.2f}",
        "Тип": "Факт",
    })

for ts, value in forecast.items():
    ts_local = to_local(ts)
    table_rows.append({
        "Дата": ts_local.strftime("%d.%m.%Y"),
        "Час": ts_local.strftime("%H:%M"),
        "Цена €/MWh": f"{value:.2f}",
        "Тип": "Прогноз",
    })

df_table = pd.DataFrame(table_rows)

if df_table.empty:
    st.info("Нет данных для отображения.")
else:
    def _style_forecast(row):
        if row["Тип"] == "Прогноз":
            return ["color: #9e9e9e; font-style: italic;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_table.style.apply(_style_forecast, axis=1),
        hide_index=True,
        use_container_width=True,
    )


# MAE, если есть факт и прогноз за одну дату
if not actuals.empty and not forecast.empty:
    common = forecast.index.intersection(actuals.index)

    if len(common) > 0:
        mae = (forecast.loc[common] - actuals.loc[common]).abs().mean()
        st.metric(
            f"Ошибка прогноза за {target_date_str} (MAE)",
            f"{mae:.2f} €/MWh",
            help=f"Средняя абсолютная ошибка на {len(common)} часах",
        )


st.caption(
    "Время отображается в Europe/Amsterdam."
)
