import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
import traceback

# ----- Import Project Modules -----
from core.metrics import FinanceMetrics
from core.portfolio import PortfolioAnalyzer
from core.risk import RiskAnalyzer
from models.forecast import FinanceForecaster
from models.anomaly import AnomalyDetector
from intelligence.advisor import BusinessAdvisor
from llm.llm_engine import LLMEngine


# ============================================================
# GLOBAL CONFIG
# ============================================================

TICKERS = ["AAPL", "MSFT", "TSLA"]
START_DATE = "2024-01-01"
END_DATE = "2025-01-01"

np.random.seed(42)


# ============================================================
# STEP 1 — FETCH MARKET DATA
# ============================================================

def fetch_market_data(tickers, start=START_DATE, end=END_DATE):

    data_frames = []

    for ticker in tickers:

        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True
            )

            if data is None or data.empty:
                print(f"Skipping {ticker} (no data)")
                continue

            df = data[["Close"]].copy()
            df.rename(columns={"Close": f"{ticker}_Close"}, inplace=True)

            data_frames.append(df)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    if len(data_frames) == 0:
        raise ValueError("No market data downloaded")

    market_df = pd.concat(data_frames, axis=1)

    market_df = market_df.dropna(how="all")

    market_df.reset_index(inplace=True)

    market_df.rename(columns={"Date": "date"}, inplace=True)

    return market_df


# ============================================================
# STEP 2 — GENERATE INTERNAL BUSINESS DATA
# ============================================================

def create_internal_financials(df):

    if df.empty:
        raise ValueError("Market dataframe is empty")

    n = len(df)

    df["revenue"] = np.random.randint(2000, 8000, n)
    df["cost"] = np.random.randint(800, 4000, n)
    df["expenses"] = np.random.randint(300, 1500, n)

    df["profit"] = df["revenue"] - df["cost"] - df["expenses"]

    df["profit_margin"] = df["profit"] / df["revenue"]

    df["cashflow"] = df["profit"] + np.random.randint(200, 800, n)

    return df


# ============================================================
# LOAD DATA PIPELINE
# ============================================================

print("Downloading market data...")

market_df = fetch_market_data(TICKERS)

print("Generating internal financial metrics...")

combined_df = create_internal_financials(market_df)


# ============================================================
# INITIALIZE LLM
# ============================================================

print("Loading LLM model...")

try:
    llm = LLMEngine(device=-1)
except Exception as e:
    print("LLM failed to load:", e)
    llm = None


# ============================================================
# CORE FINANCE AI ENGINE
# ============================================================

def finance_ai_engine(user_query):

    try:

        # -----------------------------------------------
        # KPI METRICS
        # -----------------------------------------------

        fm = FinanceMetrics(combined_df)

        fm.compute_basic_metrics()
        fm.compute_expense_ratios()
        fm.compute_rolling_metrics()

        asset_cols = [col for col in combined_df.columns if "_Close" in col]

        if asset_cols:
            fm.compute_market_metrics(asset_cols)

        kpi_df = fm.get_kpis()


        # -----------------------------------------------
        # PORTFOLIO ANALYSIS
        # -----------------------------------------------

        portfolio = PortfolioAnalyzer(combined_df, asset_columns=asset_cols)

        try:
            portfolio.compute_roi()
        except:
            pass


        # -----------------------------------------------
        # RISK ANALYSIS
        # -----------------------------------------------

        risk = RiskAnalyzer(combined_df)

        risk.detect_revenue_drop()
        risk.detect_expense_spike()

        if asset_cols:
            risk.detect_portfolio_risk(asset_cols)

        risk.detect_profit_variance()
        risk.compute_risk_score()

        risk_df = risk.risk_flags


        # -----------------------------------------------
        # FORECASTING
        # -----------------------------------------------

        try:

            forecaster = FinanceForecaster(
                combined_df,
                target_column="revenue",
                period=30
            )

            forecaster.train_model()

            forecast_df = forecaster.make_forecast()

        except Exception as e:

            print("Forecast failed:", e)

            forecast_df = pd.DataFrame()


        # -----------------------------------------------
        # ANOMALY DETECTION
        # -----------------------------------------------

        anomaly = AnomalyDetector(combined_df)

        anomaly.detect_zscore("revenue")
        anomaly.detect_rolling("profit", window=3)

        if asset_cols:
            anomaly.detect_isolation_forest(asset_cols)

        anomaly.aggregate_anomalies()

        anomaly_df = anomaly.anomaly_flags


        # -----------------------------------------------
        # BUSINESS INSIGHTS ENGINE
        # -----------------------------------------------

        advisor = BusinessAdvisor(
            kpi_df,
            risk_df=risk_df,
            forecast_df=forecast_df,
            anomaly_df=anomaly_df
        )

        insights = advisor.generate_insights()


        # -----------------------------------------------
        # LLM INTERPRETATION
        # -----------------------------------------------

        if llm:

            llm_response = llm.generate_insight(
                insights,
                user_query=user_query
            )

        else:

            llm_response = insights

        return str(llm_response)


    except Exception as e:

        error_message = f"""
Finance AI System Error

{str(e)}

Traceback:
{traceback.format_exc()}
"""

        print(error_message)

        return error_message


# ============================================================
# GRADIO INTERFACE
# ============================================================

def launch_gradio():

    interface = gr.Interface(

        fn=finance_ai_engine,

        inputs=gr.Textbox(
            lines=8,
            placeholder="Ask your finance question...",
            label="Enter Your Finance Question"
        ),

        outputs=gr.Textbox(
            label="AI Finance Advisor",
            lines=25
        ),

        title="💼 AI Finance Personal Business Analyst",

        description="""
Advanced AI-powered financial analysis assistant.

Capabilities:
• Market Data Analysis  
• Portfolio Risk Detection  
• Revenue Forecasting  
• Financial KPI Insights  
• AI Generated Business Advice  
""",

        theme="soft"
    )

    return interface


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    app = launch_gradio()

    app.launch(share=True)