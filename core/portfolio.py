import pandas as pd
import numpy as np

class PortfolioAnalyzer:
    """
    Analyze investment portfolios or company financial units.
    Compute ROI, cumulative returns, Sharpe ratio, and asset contributions.
    """

    def __init__(self, combined_df, asset_columns=None):
        """
        :param combined_df: DataFrame from metrics.py or market_data.py
        :param asset_columns: list of assets or departments to analyze (e.g., ['AAPL_Close', 'MSFT_Close'])
        """
        self.df = combined_df.copy()
        self.asset_columns = asset_columns if asset_columns else []

    # ------------------------------
    # Step 1: Compute ROI for each asset
    # ------------------------------
    def compute_roi(self):
        """
        Calculates ROI for each asset / investment
        """
        roi_dict = {}
        for col in self.asset_columns:
            if col in self.df.columns:
                start_price = self.df[col].iloc[0]
                end_price = self.df[col].iloc[-1]
                roi = (end_price - start_price) / start_price * 100
                roi_dict[col] = roi
        
        self.roi = roi_dict
        return self.roi

    # ------------------------------
    # Step 2: Compute Cumulative Returns
    # ------------------------------
    def compute_cumulative_returns(self):
        """
        Calculates cumulative returns for assets
        """
        cum_returns = pd.DataFrame()
        for col in self.asset_columns:
            if col in self.df.columns:
                cum_returns[col] = (self.df[col].pct_change() + 1).cumprod() - 1
        
        self.cum_returns = cum_returns
        return self.cum_returns

    # ------------------------------
    # Step 3: Compute Sharpe Ratio
    # ------------------------------
    def compute_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Calculates Sharpe ratio for each asset
        """
        sharpe_dict = {}
        for col in self.asset_columns:
            if col in self.df.columns:
                returns = self.df[col].pct_change().dropna()
                sharpe = (returns.mean() - risk_free_rate/252) / returns.std()
                sharpe_dict[col] = sharpe
        
        self.sharpe_ratio = sharpe_dict
        return self.sharpe_ratio

    # ------------------------------
    # Step 4: Weighted Portfolio Metrics
    # ------------------------------
    def compute_weighted_returns(self, weights):
        """
        Computes weighted returns of portfolio
        :param weights: dict {asset_column: weight}
        """
        portfolio_return = pd.Series(0, index=self.df.index)
        for col, w in weights.items():
            if col in self.df.columns:
                portfolio_return += self.df[col].pct_change().fillna(0) * w
        self.portfolio_return = portfolio_return
        return self.portfolio_return

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    combined_df = pd.read_csv("data/combined_sample.csv")
    assets = ['AAPL_Close', 'MSFT_Close']
    weights = {'AAPL_Close': 0.6, 'MSFT_Close': 0.4}
    
    pa = PortfolioAnalyzer(combined_df, asset_columns=assets)
    print("ROI:", pa.compute_roi())
    print("Cumulative Returns:\n", pa.compute_cumulative_returns().head())
    print("Sharpe Ratio:", pa.compute_sharpe_ratio())
    print("Weighted Portfolio Returns:\n", pa.compute_weighted_returns(weights).head())