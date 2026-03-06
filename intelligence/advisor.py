import pandas as pd

class BusinessAdvisor:
    """
    Generates actionable insights from financial KPIs, forecasts, risk scores, and anomalies
    """

    def __init__(self, kpi_df, risk_df=None, forecast_df=None, anomaly_df=None):
        """
        :param kpi_df: DataFrame with KPIs (from metrics.py)
        :param risk_df: DataFrame with risk scores (from risk.py)
        :param forecast_df: DataFrame with forecasted values (from forecast.py)
        :param anomaly_df: DataFrame with anomaly scores (from anomaly.py)
        """
        self.kpi_df = kpi_df.copy()
        self.risk_df = risk_df.copy() if risk_df is not None else None
        self.forecast_df = forecast_df.copy() if forecast_df is not None else None
        self.anomaly_df = anomaly_df.copy() if anomaly_df is not None else None
        self.insights = []

    # ------------------------------
    # Step 1: Profit & Loss Insights
    # ------------------------------
    def analyze_profit_loss(self):
        """
        Analyze profit trends and generate basic insights
        """
        if 'profit_margin' in self.kpi_df.columns:
            latest_margin = self.kpi_df['profit_margin'].iloc[-1]
            prev_margin = self.kpi_df['profit_margin'].iloc[-2] if len(self.kpi_df) > 1 else latest_margin
            if latest_margin < prev_margin:
                self.insights.append(f"Profit margin decreased from {prev_margin:.2f}% to {latest_margin:.2f}%. Consider reviewing high-cost areas.")
            else:
                self.insights.append(f"Profit margin increased to {latest_margin:.2f}%. Positive trend.")

    # ------------------------------
    # Step 2: Expense Insights
    # ------------------------------
    def analyze_expenses(self):
        """
        Recommend cost optimization if expense ratio is high
        """
        if 'expense_ratio' in self.kpi_df.columns:
            latest_ratio = self.kpi_df['expense_ratio'].iloc[-1]
            if latest_ratio > 50:  # Example threshold
                self.insights.append(f"Expense ratio is {latest_ratio:.2f}%. Consider cost-cutting strategies in operational departments.")

    # ------------------------------
    # Step 3: Risk-Based Recommendations
    # ------------------------------
    def analyze_risks(self):
        """
        Use risk scores to generate recommendations
        """
        if self.risk_df is not None and 'risk_score' in self.risk_df.columns:
            latest_risk = self.risk_df['risk_score'].iloc[-1]
            if latest_risk > 0.5:  # Example threshold
                self.insights.append(f"High risk score detected ({latest_risk:.2f}). Investigate anomalies and high-risk assets immediately.")

    # ------------------------------
    # Step 4: Forecast Insights
    # ------------------------------
    def analyze_forecast(self):
        """
        Highlight expected trends based on forecasts
        """
        if self.forecast_df is not None and 'yhat' in self.forecast_df.columns:
            predicted_change = self.forecast_df['yhat'].iloc[-1] - self.forecast_df['yhat'].iloc[-2]
            if predicted_change < 0:
                self.insights.append(f"Forecast predicts a decrease in {self.kpi_df.columns[0]} by {abs(predicted_change):.2f} units. Plan corrective actions.")
            else:
                self.insights.append(f"Forecast predicts growth of {predicted_change:.2f} units. Opportunity to scale operations.")

    # ------------------------------
    # Step 5: Anomaly-Based Insights
    # ------------------------------
    def analyze_anomalies(self):
        """
        Highlight anomalies and their impact
        """
        if self.anomaly_df is not None and 'anomaly_score' in self.anomaly_df.columns:
            latest_anomaly = self.anomaly_df['anomaly_score'].iloc[-1]
            if latest_anomaly > 0.3:
                self.insights.append(f"Anomalies detected in financial/market data (score: {latest_anomaly:.2f}). Review operations and financial records.")

    # ------------------------------
    # Step 6: Generate All Insights
    # ------------------------------
    def generate_insights(self):
        self.insights = []  # Reset
        self.analyze_profit_loss()
        self.analyze_expenses()
        self.analyze_risks()
        self.analyze_forecast()
        self.analyze_anomalies()
        return self.insights

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    kpi_df = pd.read_csv("data/kpi_sample.csv")
    risk_df = pd.read_csv("data/risk_sample.csv")
    forecast_df = pd.read_csv("data/forecast_sample.csv")
    anomaly_df = pd.read_csv("data/anomaly_sample.csv")
    
    advisor = BusinessAdvisor(kpi_df, risk_df, forecast_df, anomaly_df)
    insights = advisor.generate_insights()
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")