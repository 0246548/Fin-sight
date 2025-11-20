import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import requests  # para llamar a SerpAPI

# =========================
# Carga de variables .env
# =========================
load_dotenv()
KEY = os.getenv("ANTHROPIC_KEY")
SERAPIKEY = os.getenv("SERAPI_KEY")
SERPAPI_KEY = SERAPIKEY  # alias para usar el mismo nombre que en la otra app
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# Cliente de Anthropic (Claude)
client = None
if KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=KEY)
    except Exception:
        client = None

# =========================
# CONFIGURACIÓN INICIAL
# =========================
st.set_page_config(
    page_title="FinSight - Dashboard Financiero",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema fino extra (no rompe si cambia el CSS de Streamlit)
st.markdown("""
<style>
/* Fondo y textos */
.stApp { background-color: #0e1117; color: #e6e6e6; }
.block-container { padding-top: 1.5rem; }

/* Cards y tablas */
div[data-testid="stMetricValue"] { color: #00c3ff !important; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #cccccc !important; }
thead tr th { color: #e6e6e6 !important; }
tbody tr td { color: #e6e6e6 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #1c1e26; border-radius: 10px; padding: 6px 12px; }
.stTabs [aria-selected="true"] { background: #222633; border: 1px solid #2a2f40; }
</style>
""", unsafe_allow_html=True)

# =========================
# ENCABEZADO PRINCIPAL
# =========================
st.markdown(
    """
    <style>
    .main-title { text-align: center; font-size: 42px; font-weight: 700; color: #00c3ff; margin-bottom: -10px; }
    .sub-title { text-align: center; font-size: 18px; color: #cccccc; margin-bottom: 30px; }
    </style>
    <p class="main-title">FinSight 📊</p>
    <p class="sub-title">Modelo Integral de Análisis Financiero y Comparativo</p>
    """,
    unsafe_allow_html=True
)

# =========================
# CONFIGURAR GEMINI
# =========================
genai.configure(api_key="TU_API_KEY_DE_GEMINI_AQUI")  # <-- cambia esto
model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# SIDEBAR - ENTRADAS
# =========================
st.sidebar.header("⚙️ Parámetros de Análisis")
stonk = st.sidebar.text_input("Símbolo de la acción:", value="AMD")
indice = st.sidebar.selectbox("Índice de referencia:", ["SPY", "QQQ", "^GSPC", "^IXIC"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Escribe un ticker válido como AAPL, NVDA, MSFT, etc.")
st.sidebar.markdown("Creado por **Emiliano Ramírez** © 2025")
st.sidebar.markdown("---")
news_key_sidebar = st.sidebar.text_input("🔑 NewsAPI (opcional para Noticias)", type="password")

# ===== Helpers reutilizables =====
def pick_close(df: pd.DataFrame):
    """Intenta devolver la columna de precio de cierre (Adj Close/Close/variantes)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c]).strip()
            for col in df.columns.values
        ]
    for cand in ["Adj Close", "Close", "close"]:
        if cand in df.columns:
            return df[cand]
    for c in df.columns:
        if "close" in str(c).lower():
            return df[c]
    return None

def ensure_1d(x):
    """Convierte DataFrame(n,1)/array(n,1) a Serie 1D numérica."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if hasattr(x, "to_numpy"):
        x = x.to_numpy().ravel()
    return pd.to_numeric(pd.Series(x), errors="coerce")

def rsi_from_series(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI clásico sin librerías externas."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =========================
# DESCARGA DE DATOS
# =========================
if stonk:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    st.write(f"📅 Datos desde **{start_date.date()}** hasta **{end_date.date()}**")

    data = yf.download(stonk, start=start_date, end=end_date, interval='1d').reset_index()
    if data.empty:
        st.warning("⚠️ No se encontraron datos para este símbolo.")
    else:
        # =========================
        # LIMPIEZA Y VALIDACIÓN
        # =========================
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Aplanar columnas si vienen como MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                "_".join([str(c) for c in col if c]).strip()
                for col in data.columns.values
            ]

        # Detectar columna de Volumen (robusto a tuplas)
        vol_cols = [str(c) for c in data.columns if 'volume' in str(c).lower()]
        if vol_cols:
            data['Volume'] = pd.to_numeric(data[vol_cols[0]], errors='coerce')
        else:
            st.warning("⚠️ No se encontró columna de Volumen en los datos.")
            data['Volume'] = 0

        # Columnas OHLC
        open_col  = [c for c in data.columns if 'open'  in str(c).lower()]
        high_col  = [c for c in data.columns if 'high'  in str(c).lower()]
        low_col   = [c for c in data.columns if 'low'   in str(c).lower()]
        close_col = [c for c in data.columns if 'close' in str(c).lower()]

        # =========================
        # INTERFAZ CON TABS
        # =========================
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Gráficos",
            "📈 Comparación",
            "📉 Rendimiento y Riesgo",
            "🏢 Empresa / CEO",
            "📰 Noticias",
            "🧪 Monte Carlo / VaR"
        ])

        # ---------- TAB 1: GRÁFICOS ----------
        with tab1:
            st.subheader(f"Gráfico OHLC - {stonk}")
            col_rango1, col_rango2 = st.columns(2)
            rango_opcion = col_rango1.selectbox(
                "Rango de tiempo para visualizar:",
                ["1Y", "3Y", "5Y", "MAX"],
                index=2
            )
            mostrar_inds = col_rango2.multiselect(
                "Indicadores a mostrar:",
                ["EMA 20", "EMA 50", "RSI 14"],
                default=["EMA 20", "EMA 50", "RSI 14"]
            )

            # Filtrar data según rango (sin modificar el DataFrame original)
            if rango_opcion == "1Y":
                cutoff = end_date - timedelta(days=365)
            elif rango_opcion == "3Y":
                cutoff = end_date - timedelta(days=3 * 365)
            elif rango_opcion == "5Y":
                cutoff = end_date - timedelta(days=5 * 365)
            else:
                cutoff = start_date

            df_plot = data[data["Date"] >= cutoff].copy()

            fig, ax1 = plt.subplots(figsize=(14, 6))
            sns.set_theme(style="darkgrid", palette="bright")

            ax1.plot(df_plot['Date'], df_plot[open_col[0]],  label='Open',  linestyle='--', linewidth=1.5, color='#00c3ff')
            ax1.plot(df_plot['Date'], df_plot[high_col[0]],  label='High',  linestyle=':',  linewidth=1.5, color='#00ff85')
            ax1.plot(df_plot['Date'], df_plot[low_col[0]],   label='Low',   linestyle='-.', linewidth=1.5, color='#ffb400')
            ax1.plot(df_plot['Date'], df_plot[close_col[0]], label='Close',  linewidth=2,   color='#ff3c3c')

            ax1.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax1.set_xlabel("Fecha", fontsize=12, color="white")
            ax1.set_ylabel("Precio (USD)", fontsize=12, color="white")
            ax1.set_title(f"{stonk} - Evolución de precio", fontsize=16, color="white")
            ax1.tick_params(colors="white")
            ax1.legend(loc='upper left', facecolor="#1c1e26", edgecolor="white")

            ax2 = ax1.twinx()
            ax2.bar(df_plot['Date'], df_plot['Volume'], width=1.0, color='white', alpha=0.15, label='Volumen')
            ax2.set_ylabel("Volumen", fontsize=12, color="white")
            ax2.tick_params(colors="white")
            st.pyplot(fig)

            # Gráfico de velas japonés (último año)
            one_year_ago = end_date - timedelta(days=365)
            df1y = data[data["Date"] >= one_year_ago]
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df1y["Date"],
                open=df1y[open_col[0]],
                high=df1y[high_col[0]],
                low=df1y[low_col[0]],
                close=df1y[close_col[0]]
            )])
            fig_candle.update_layout(
                title=f"Velas Japonesas - {stonk} (último año)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=520
            )
            st.plotly_chart(fig_candle, use_container_width=True)

            # ====== Gráfico técnico: EMAs y RSI ======
            st.subheader("Indicadores técnicos")
            precio_close_full = pick_close(data)
            if precio_close_full is not None:
                precio_close_full = ensure_1d(precio_close_full)
                df_ta = pd.DataFrame({"Date": data["Date"], "Close": precio_close_full}).dropna()

                df_ta["EMA20"] = df_ta["Close"].ewm(span=20, adjust=False).mean()
                df_ta["EMA50"] = df_ta["Close"].ewm(span=50, adjust=False).mean()
                df_ta["RSI14"] = rsi_from_series(df_ta["Close"], period=14)

                # Precio + EMAs
                fig_ema = go.Figure()
                fig_ema.add_trace(go.Scatter(x=df_ta["Date"], y=df_ta["Close"], name="Precio", line=dict(color="#00c3ff")))
                if "EMA 20" in mostrar_inds:
                    fig_ema.add_trace(go.Scatter(x=df_ta["Date"], y=df_ta["EMA20"], name="EMA 20", line=dict(color="#ff3c3c")))
                if "EMA 50" in mostrar_inds:
                    fig_ema.add_trace(go.Scatter(x=df_ta["Date"], y=df_ta["EMA50"], name="EMA 50", line=dict(color="#ffb400")))
                fig_ema.update_layout(template="plotly_dark", height=420, title=f"{stonk} - Medias móviles")
                st.plotly_chart(fig_ema, use_container_width=True)

                # RSI
                if "RSI 14" in mostrar_inds:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df_ta["Date"], y=df_ta["RSI14"], name="RSI(14)"))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff3c3c")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00c3ff")
                    fig_rsi.update_yaxes(range=[0, 100])
                    fig_rsi.update_layout(template="plotly_dark", height=250, title=f"{stonk} - RSI(14)")
                    st.plotly_chart(fig_rsi, use_container_width=True)
            else:
                st.info("No se encontró columna de cierre para calcular indicadores técnicos.")

        # ---------- TAB 2: COMPARACIÓN ----------
        with tab2:
            st.subheader(f"Comparación con índice {indice} (Base 0)")

            # Descarga índice
            idx = yf.download(
                indice,
                start=start_date, end=end_date,
                interval='1d', auto_adjust=True
            ).reset_index()

            # Columnas de cierre
            px_t = pick_close(data)
            px_i = pick_close(idx)

            if (px_t is None) or (px_i is None):
                st.error("No se encontró columna de precio para acción o índice.")
            else:
                # --- Normalización robusta ---
                def make_clean_df(dates, prices, colname):
                    p = ensure_1d(prices)
                    d = pd.to_datetime(dates, errors="coerce").dt.tz_localize(None).dt.normalize()
                    df = pd.DataFrame({"Date": d, colname: p})
                    df = df.dropna(subset=[colname])
                    df = df.groupby("Date", as_index=False)[colname].last()
                    return df.sort_values("Date")

                df_t = make_clean_df(data["Date"], px_t, "Acción")
                df_i = make_clean_df(idx["Date"],  px_i, "Índice")

                merged = pd.merge(df_t, df_i, on="Date", how="inner").dropna()

                if merged.empty:
                    st.warning("No se pudo construir la comparación (sin fechas en común tras limpiar/normalizar).")
                else:
                    # Rebase a 0
                    merged["Acción (base 0)"] = merged["Acción"] / merged["Acción"].iloc[0] - 1
                    merged["Índice (base 0)"] = merged["Índice"] / merged["Índice"].iloc[0] - 1

                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(
                        x=merged["Date"], y=merged["Acción (base 0)"],
                        mode="lines", name=stonk, line=dict(color="#00c3ff")
                    ))
                    fig_comp.add_trace(go.Scatter(
                        x=merged["Date"], y=merged["Índice (base 0)"],
                        mode="lines", name=indice, line=dict(color="#ff3c3c")
                    ))
                    fig_comp.update_yaxes(tickformat=".0%")
                    fig_comp.update_layout(
                        template="plotly_dark",
                        height=520,
                        title=f"{stonk} vs {indice} (rendimiento base 0)",
                        xaxis_title="Fecha",
                        yaxis_title="Rendimiento",
                        legend_title_text="Serie",
                        xaxis=dict(
                            rangeselector=dict(buttons=list([
                                dict(count=3, label="3M", step="month", stepmode="backward"),
                                dict(count=6, label="6M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(count=3, label="3Y", step="year", stepmode="backward"),
                                dict(step="all", label="MAX")
                            ])),
                            rangeslider=dict(visible=False),
                            type="date"
                        )
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # ===== Métricas PRO: Beta / Correlación / Rend. relativo =====
                    try:
                        tmp = merged.copy()
                        tmp["ret_acc"] = tmp["Acción"].pct_change()
                        tmp["ret_idx"] = tmp["Índice"].pct_change()
                        beta = (tmp["ret_acc"].cov(tmp["ret_idx"]) /
                                tmp["ret_idx"].var()) if tmp["ret_idx"].var() not in [0, np.nan] else np.nan
                        corr = tmp["ret_acc"].corr(tmp["ret_idx"])
                        diff_rel = merged["Acción (base 0)"].iloc[-1] - merged["Índice (base 0)"].iloc[-1]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Beta (muestral)", f"{beta:.2f}" if pd.notna(beta) else "N/D")
                        c2.metric("Correlación", f"{corr:.2f}" if pd.notna(corr) else "N/D")
                        c3.metric("Rend. Relativo", f"{diff_rel:.2%}")
                    except Exception:
                        st.info("No fue posible calcular Beta/Correlación con los datos disponibles.")

        # ---------- TAB 3: RENDIMIENTO ----------
        with tab3:
            st.subheader("Rendimiento y Riesgo por Periodo")

            precio = pick_close(data)
            precio = ensure_1d(precio) if precio is not None else None
            if precio is None:
                st.error("No fue posible obtener la columna de precios para cálculo de rendimientos.")
            else:
                dfp = pd.DataFrame({"Date": data["Date"], "P": precio}).dropna().sort_values("Date")

                def sub_period(days=None, since=None):
                    s = dfp.copy()
                    if days:
                        cutoff = end_date - timedelta(days=days)
                        s = s[s["Date"] >= cutoff]
                    if since:
                        s = s[s["Date"] >= since]
                    return s["P"]

                def rend(sub):
                    return sub.iloc[-1] / sub.iloc[0] - 1 if len(sub) > 1 else None

                def vol(sub):
                    r = sub.pct_change().dropna()
                    return r.std() * (252 ** 0.5) if len(r) > 1 else None

                ytd_start = datetime(end_date.year, 1, 1)
                periods = {
                    "YTD": {"since": ytd_start},
                    "3M": {"days": 90},
                    "6M": {"days": 180},
                    "1Y": {"days": 365},
                    "3Y": {"days": 3 * 365},
                    "5Y": {"days": 5 * 365}
                }

                rows = []
                for label, params in periods.items():
                    serie = sub_period(**params)
                    rows.append({
                        "Periodo": label,
                        "Rendimiento (Aritmético)": rend(serie),
                        "Volatilidad (Anualizada)": vol(serie)
                    })

                tabla = pd.DataFrame(rows)
                st.dataframe(
                    tabla.style.format({
                        "Rendimiento (Aritmético)": "{:.2%}",
                        "Volatilidad (Anualizada)": "{:.2%}"
                    })
                )
                c1, c2, c3 = st.columns(3)
                try:
                    c1.metric("Rendimiento 1Y", f"{tabla.loc[tabla['Periodo']=='1Y','Rendimiento (Aritmético)'].values[0]:.2%}")
                    c2.metric("Volatilidad 1Y", f"{tabla.loc[tabla['Periodo']=='1Y','Volatilidad (Anualizada)'].values[0]:.2%}")
                    c3.metric("Rendimiento YTD", f"{tabla.loc[tabla['Periodo']=='YTD','Rendimiento (Aritmético)'].values[0]:.2%}")
                except Exception:
                    pass

        # ---------- TAB 4: EMPRESA / CEO ----------
        with tab4:
            ticker = stonk
            st.subheader(f"CEO y contexto ejecutivo de {ticker}")

            tk = yf.Ticker(ticker)
            try:
                info_ceo = tk.info
            except Exception:
                info_ceo = {}
                st.warning("No se pudo cargar la información ejecutiva desde yfinance.")

            company_name = info_ceo.get("longName", ticker)

            ceo_name = None
            officers = info_ceo.get("companyOfficers", [])

            if isinstance(officers, list):
                for o in officers:
                    title = str(o.get("title", "")).lower()
                    name = o.get("name")
                    if "chief executive officer" in title or "ceo" in title:
                        ceo_name = name
                        break
                if ceo_name is None and len(officers) > 0:
                    ceo_name = officers[0].get("name")

            if ceo_name is None:
                st.warning("No pude identificar el nombre del CEO.")
            else:
                st.markdown(f"### 👔 {ceo_name}")
                st.caption(f"Máximo responsable ejecutivo de **{company_name}**")

                # FOTO DEL CEO (SerpAPI)
                if not SERPAPI_KEY:
                    st.warning("Falta SERPAPI_KEY en el .env para buscar la foto del CEO.")
                else:
                    try:
                        params_img = {
                            "engine": "google_images",
                            "q": f"{ceo_name} CEO {company_name}",
                            "api_key": SERPAPI_KEY,
                            "ijn": "0",
                        }
                        resp_img = requests.get(
                            "https://serpapi.com/search.json",
                            params=params_img,
                            timeout=10
                        )
                        data_img = resp_img.json()
                        images = data_img.get("images_results", [])
                    except Exception as e:
                        images = []
                        st.error(f"Error al consultar imágenes: {e}")

                    if images:
                        img_url = (
                            images[0].get("original")
                            or images[0].get("thumbnail")
                            or images[0].get("link")
                        )
                        if img_url:
                            st.image(
                                img_url,
                                caption=f"{ceo_name} — {company_name}",
                                width=300,
                            )
                    else:
                        st.info("No se encontró imagen del CEO.")

                # DESCRIPCIÓN DE LA EMPRESA (Anthropic)
                if client is not None:
                    business_summary = info_ceo.get("longBusinessSummary", "")
                    business_summary = business_summary[:1500]

                    prompt_ceo = (
                        f"Nombre del CEO: {ceo_name}\n"
                        f"Empresa: {company_name}\n\n"
                        f"Resumen oficial del negocio:\n{business_summary}\n\n"
                        "Escribe una explicación clara y profesional (máximo 2 párrafos) que describa:\n"
                        "- Qué hace esta empresa y en qué industrias opera.\n"
                        "- Cómo encaja el rol del CEO dentro de la estrategia general del negocio.\n"
                        "- Evita inventar datos; usa solo la información del resumen.\n"
                        "- No hables de la vida personal del CEO, solo del negocio."
                    )

                    try:
                        ceo_analysis = client.messages.create(
                            model=ANTHROPIC_MODEL,
                            max_tokens=600,
                            temperature=0.2,
                            messages=[{"role": "user", "content": prompt_ceo}],
                        )
                        st.subheader("🏢 ¿Qué hace la empresa? (Anthropic)")
                        st.write(ceo_analysis.content[0].text)
                    except Exception as e:
                        st.error(f"Error al generar análisis del CEO: {e}")
                else:
                    st.warning("No se pudo generar análisis porque falta la ANTHROPIC_KEY.")

        # ---------- TAB 5: NOTICIAS ----------
        with tab5:
            st.subheader(f"📰 Noticias recientes sobre {stonk}")

            if not SERPAPI_KEY:
                st.warning("No se encontró SERPAPI_KEY en el .env, no se pueden consultar noticias.")
            else:
                try:
                    params = {
                        "engine": "google_news",
                        "q": f"{stonk} stock",
                        "api_key": SERPAPI_KEY,
                        "hl": "en",
                        "num": 5
                    }
                    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
                    data_news = resp.json()
                    news_results = data_news.get("news_results", [])
                except Exception as e:
                    news_results = []
                    st.error(f"Ocurrió un error al consultar SerpAPI: {e}")

                rows = []
                for item in news_results[:5]:
                    title = item.get("title")
                    link = item.get("link")
                    source = item.get("source") or {}
                    publisher = source.get("name") if isinstance(source, dict) else source
                    date_str = item.get("date") or item.get("published_at")

                    rows.append(
                        {
                            "Fecha": date_str,
                            "Título": title,
                            "Fuente": publisher,
                            "Link": link,
                        }
                    )

                if not rows:
                    st.warning("No se encontraron noticias recientes para este ticker en SerpAPI.")
                else:
                    df_news = pd.DataFrame(rows)
                    st.dataframe(df_news, use_container_width=True)

                    titulares_texto = ""
                    for r in rows:
                        if r["Título"]:
                            titulares_texto += f"- {r['Título']} (Fuente: {r['Fuente']}, Fecha: {r['Fecha']})\n"

                    if client is not None and titulares_texto.strip():
                        mensaje_noticias = client.messages.create(
                            model=ANTHROPIC_MODEL,
                            max_tokens=900,
                            temperature=0.2,
                            system=(
                                "Rol:\nEres un analista de riesgo financiero que resume noticias relevantes "
                                "para un inversionista institucional. Debes identificar tono general, riesgos "
                                "y posibles catalizadores para el precio de la acción. No recomiendes comprar "
                                "o vender, solo interpreta la narrativa."
                            ),
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"Titulares recientes sobre la empresa {stonk}:\n\n"
                                                f"{titulares_texto}\n"
                                                "Resume el sentimiento general, identifica riesgos clave y posibles "
                                                "catalizadores (positivos o negativos) para el precio. "
                                                "Máximo 900 caracteres."
                                            ),
                                        }
                                    ],
                                }
                            ],
                        )
                        st.subheader("Análisis cualitativo de noticias (Anthropic)")
                        st.write(mensaje_noticias.content[0].text)
                    else:
                        st.info("No hay titulares suficientes para analizar con Anthropic.")

        # ---------- TAB 6: MONTE CARLO / VaR ----------
        with tab6:
            st.subheader(f"🧪 Simulación Monte Carlo y VaR para {stonk}")

            precios_mc = pick_close(data)
            precios_mc = ensure_1d(precios_mc) if precios_mc is not None else None

            if precios_mc is None or len(precios_mc) < 30:
                st.warning("No hay suficientes datos históricos para correr la simulación.")
            else:
                # Parámetros interactivos
                c1, c2, c3 = st.columns(3)
                horizon_days = c1.slider("Horizonte de simulación (días)", 30, 252 * 3, 252)
                n_paths = c2.slider("Número de trayectorias", 100, 1000, 300, step=50)
                var_conf = c3.slider("Nivel de confianza VaR (%)", 90, 99, 95)

                # Escenario de estrés
                stress_option = st.selectbox(
                    "Escenario de estrés",
                    [
                        "Base (sin estrés)",
                        "Volatilidad x2",
                        "Volatilidad x3",
                        "Crash inicial -10%",
                        "Shock en día 50: -20%"
                    ]
                )

                # Estimación de drift y volatilidad con rendimientos logarítmicos
                s = pd.Series(precios_mc).dropna()
                log_ret = np.log(s[1:].values / s[:-1].values)
                mu = np.mean(log_ret)
                sigma = np.std(log_ret)
                dt = 1.0  # 1 día
                S0 = float(s.iloc[-1])

                if sigma <= 0:
                    st.warning("La volatilidad histórica es cero o inválida, no se puede simular.")
                else:
                    # Ajustes por escenario de estrés
                    mu_eff = mu
                    sigma_eff = sigma

                    if stress_option == "Volatilidad x2":
                        sigma_eff = sigma * 2
                    elif stress_option == "Volatilidad x3":
                        sigma_eff = sigma * 3
                    # (mu lo dejamos igual para que el estrés venga por volatilidad)

                    # Simulación Monte Carlo (log-precios)
                    np.random.seed(42)
                    Z = np.random.normal(size=(n_paths, horizon_days))
                    increments = (mu_eff - 0.5 * sigma_eff**2) * dt + sigma_eff * np.sqrt(dt) * Z

                    # Aplicar shocks discretos
                    if stress_option == "Crash inicial -10%":
                        increments[:, 0] = np.log(0.9)  # caída del 10% en el primer día
                    elif stress_option == "Shock en día 50: -20%":
                        idx_shock = min(49, horizon_days - 1)
                        increments[:, idx_shock] = np.log(0.8)  # caída del 20%

                    log_S = np.zeros((n_paths, horizon_days + 1))
                    log_S[:, 0] = np.log(S0)
                    log_S[:, 1:] = np.log(S0) + np.cumsum(increments, axis=1)
                    paths = np.exp(log_S)  # (n_paths, horizon+1)

                    # ========================
                    # Fan chart (percentiles)
                    # ========================
                    dias = np.arange(horizon_days + 1)
                    percs = [10, 25, 50, 75, 90]
                    perc_vals = np.percentile(paths, percs, axis=0)  # shape (5, T)

                    fig_fan = go.Figure()

                    # Banda 10–90
                    fig_fan.add_trace(go.Scatter(
                        x=dias,
                        y=perc_vals[4],
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
                    fig_fan.add_trace(go.Scatter(
                        x=dias,
                        y=perc_vals[0],
                        fill='tonexty',
                        fillcolor='rgba(0,195,255,0.12)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='P10–P90'
                    ))

                    # Banda 25–75
                    fig_fan.add_trace(go.Scatter(
                        x=dias,
                        y=perc_vals[3],
                        line=dict(color='rgba(0,0,0,0)'),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
                    fig_fan.add_trace(go.Scatter(
                        x=dias,
                        y=perc_vals[1],
                        fill='tonexty',
                        fillcolor='rgba(0,195,255,0.25)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='P25–P75'
                    ))

                    # Mediana
                    fig_fan.add_trace(go.Scatter(
                        x=dias,
                        y=perc_vals[2],
                        line=dict(color='#00c3ff', width=2),
                        name='Mediana (P50)'
                    ))

                    fig_fan.update_layout(
                        template="plotly_dark",
                        height=520,
                        title=f"Simulación Monte Carlo del precio ({stress_option})",
                        xaxis_title="Días",
                        yaxis_title="Precio simulado",
                    )
                    st.plotly_chart(fig_fan, use_container_width=True)

                    # ========================
                    # VaR y Expected Shortfall
                    # ========================
                    final_prices = paths[:, -1]
                    final_returns = final_prices / S0 - 1.0

                    alpha = 1.0 - var_conf / 100.0
                    var_pct = np.percentile(final_returns, 100 * alpha)
                    es_mask = final_returns <= var_pct
                    es_pct = final_returns[es_mask].mean() if np.any(es_mask) else np.nan

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precio actual", f"{S0:.2f}")
                    m2.metric(f"VaR {var_conf:.0f}%", f"{var_pct:.2%}")
                    m3.metric("Expected Shortfall", f"{es_pct:.2%}" if not np.isnan(es_pct) else "N/D")

                    # Histograma de rendimientos finales con línea de VaR
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=final_returns,
                        nbinsx=40,
                        marker=dict(color='rgba(0,195,255,0.7)'),
                        name="Rendimientos simulados"
                    ))
                    fig_hist.add_vline(
                        x=var_pct,
                        line_dash="dash",
                        line_color="#ff3c3c",
                        annotation_text="VaR",
                        annotation_position="top left"
                    )
                    fig_hist.update_layout(
                        template="plotly_dark",
                        height=420,
                        title="Distribución de rendimientos finales",
                        xaxis_title="Rendimiento",
                        yaxis_title="Frecuencia",
                        bargap=0.05
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.caption(
                        "La simulación asume un movimiento browniano geométrico estimado con rendimientos "
                        "logarítmicos diarios históricos. Es un modelo simplificado, pero sirve para intuir "
                        "el rango de precios posibles y el riesgo de cola bajo distintos escenarios de estrés."
                    )

                    # ========================
                    # Probabilidad de alcanzar metas
                    # ========================
                    st.markdown("### 🎯 Probabilidad de alcanzar objetivos de precio")

                    col_up, col_down = st.columns(2)
                    default_up = float(np.round(S0 * 1.2, 2))
                    default_down = float(np.round(S0 * 0.8, 2))

                    target_up = col_up.number_input(
                        "Precio objetivo al alza",
                        min_value=0.0,
                        value=default_up
                    )
                    target_down = col_down.number_input(
                        "Precio de stop-loss",
                        min_value=0.0,
                        value=default_down
                    )

                    # Probabilidad de terminar por arriba / abajo
                    prob_up_T = float((final_prices >= target_up).mean())
                    prob_down_T = float((final_prices <= target_down).mean())

                    # Probabilidad de tocar en algún momento del camino
                    hit_up_any = float((paths >= target_up).any(axis=1).mean())
                    hit_down_any = float((paths <= target_down).any(axis=1).mean())

                    cu, cd = st.columns(2)
                    with cu:
                        st.metric(
                            "Prob. terminar ≥ objetivo",
                            f"{prob_up_T:.1%}"
                        )
                        st.metric(
                            "Prob. tocar ≥ objetivo en algún momento",
                            f"{hit_up_any:.1%}"
                        )
                    with cd:
                        st.metric(
                            "Prob. terminar ≤ stop-loss",
                            f"{prob_down_T:.1%}"
                        )
                        st.metric(
                            "Prob. tocar ≤ stop-loss en algún momento",
                            f"{hit_down_any:.1%}"
                        )

                    st.caption(
                        "Las probabilidades se basan únicamente en la distribución simulada bajo las "
                        "suposiciones del modelo. No representan garantías ni recomendaciones de inversión."
                    )
