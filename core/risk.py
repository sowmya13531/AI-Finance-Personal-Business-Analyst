import pandas as pd
import numpy as np


class RiskAnalyzer:
    """
    Analyzes financial and portfolio data to identify risks and anomalies.
    """

    def __init__(self, combined_df):
        """
        :param combined_df: unified DataFrame with KPIs & market data
        """
        self.df = combined_df.copy()
        self.risk_flags = pd.DataFrame(index=self.df.index)

    # ------------------------------
    # Step 1: Revenue Drop Alert
    # ------------------------------
    def detect_revenue_drop(self, threshold_pct=10):

        if "revenue" in self.df.columns:
            self.risk_flags["revenue_drop"] = (
                self.df["revenue"].pct_change() * -100 > threshold_pct
            )

        return self.risk_flags.get("revenue_drop")

    # ------------------------------
    # Step 2: Expense Spike Alert
    # ------------------------------
    def detect_expense_spike(self, threshold_pct=10):

        if "expenses" in self.df.columns:
            self.risk_flags["expense_spike"] = (
                self.df["expenses"].pct_change() * 100 > threshold_pct
            )

        return self.risk_flags.get("expense_spike")

    # ------------------------------
    # Step 3: Portfolio Risk (Volatility)
    # ------------------------------
    def detect_portfolio_risk(self, asset_columns, vol_threshold=0.05):

        created_cols = []

        for col in asset_columns:

            if col in self.df.columns:

                vol_col = f"{col}_high_vol"

                self.risk_flags[vol_col] = (
                    self.df[col].pct_change().rolling(3).std() > vol_threshold
                )

                created_cols.append(vol_col)

        return self.risk_flags[created_cols]

    # ------------------------------
    # Step 4: Profit Variance
    # ------------------------------
    def detect_profit_variance(self, window=3, threshold_pct=15):

        if "profit" in self.df.columns:

            rolling_mean = self.df["profit"].rolling(window).mean()

            deviation = abs(self.df["profit"] - rolling_mean) / rolling_mean * 100

            self.risk_flags["profit_variance"] = deviation > threshold_pct

        return self.risk_flags.get("profit_variance")

    # ------------------------------
    # Step 5: Aggregate Risk Score
    # ------------------------------
    def compute_risk_score(self):

        if self.risk_flags.shape[1] == 0:
            self.risk_flags["risk_score"] = 0
        else:
            self.risk_flags["risk_score"] = (
                self.risk_flags.sum(axis=1) / self.risk_flags.shape[1]
            )

        return self.risk_flags["risk_score"]