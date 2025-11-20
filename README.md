# FinSight 📊

Dashboard financiero interactivo hecho en Streamlit para la materia de Ingeniería Financiera.

## Funcionalidad principal

- Descarga datos históricos de acciones con `yfinance`.
- Gráficos OHLC, velas japonesas, EMAs y RSI.
- Comparación de la acción contra un índice (SPY, QQQ, etc.).
- Cálculo de rendimiento y volatilidad por periodo.
- Pestaña de Empresa/CEO con resumen del negocio.
- Noticias recientes de la empresa usando SerpAPI + resumen con IA.
- Simulación Monte Carlo del precio y cálculo de VaR y Expected Shortfall.

## Cómo correr la app

1. Clonar el repositorio:

```bash
git clone https://github.com/0246548/Fin-sight.git
cd Fin-sight
python -m venv venv
source venv/bin/activate   # Mac / Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
ANTHROPIC_KEY=tu_api_key_de_claude
SERAPI_KEY=tu_api_key_de_serpapi
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
streamlit run app.py
