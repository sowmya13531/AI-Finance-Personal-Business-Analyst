import pandas as pd

class FinanceMetrics:
    """
    Compute financial KPIs and derived metrics
    from combined finance + market data.
    """

    def __init__(self, combined_df):
        """
        :param combined_df: unified DataFrame from market_data.py
        """
        self.df = combined_df.copy()

    # ------------------------------
    # Step 1: Profit & Growth Metrics
    # ------------------------------
    def compute_basic_metrics(self):
        """
        Calculates profit, profit margin, revenue growth
        """
        # Example: Internal revenue/ cost columns
        if 'revenue' in self.df.columns and 'cost' in self.df.columns:
            self.df['profit'] = self.df['revenue'] - self.df['cost']
            self.df['profit_margin'] = self.df['profit'] / self.df['revenue'] * 100

            # Revenue growth %
            self.df['revenue_growth_pct'] = self.df['revenue'].pct_change() * 100

        return self.df

    # ------------------------------
    # Step 2: Expense & Risk Ratios
    # ------------------------------
    def compute_expense_ratios(self):
        """
        Calculates expense-related metrics
        """
        if 'expenses' in self.df.columns and 'revenue' in self.df.columns:
            self.df['expense_ratio'] = self.df['expenses'] / self.df['revenue'] * 100

        return self.df

    # ------------------------------
    # Step 3: Rolling Metrics for Trends
    # ------------------------------
    def compute_rolling_metrics(self, window=3):
        """
        Smooth KPIs using rolling averages
        """
        self.df['rolling_revenue'] = self.df['revenue'].rolling(window).mean()
        self.df['rolling_profit'] = self.df['profit'].rolling(window).mean()
        self.df['rolling_expense_ratio'] = self.df['expense_ratio'].rolling(window).mean()

        return self.df

    # ------------------------------
    # Step 4: Volatility & Returns (Market Data)
    # ------------------------------
    def compute_market_metrics(self, ticker_columns):
        """
        Compute stock-related metrics: returns & volatility
        :param ticker_columns: list of market price columns (e.g., ['AAPL_Close', 'MSFT_Close'])
        """
        for col in ticker_columns:
            if col in self.df.columns:
                # Daily / monthly returns
                self.df[f'{col}_returns'] = self.df[col].pct_change() * 100
                # Volatility (rolling std)
                self.df[f'{col}_volatility'] = self.df[f'{col}_returns'].rolling(3).std()

        return self.df

    # ------------------------------
    # Step 5: Generate KPI DataFrame for Advisor / LLM
    # ------------------------------
    def get_kpis(self):
        """
        Returns a KPI-focused DataFrame for further analysis
        """
        kpi_cols = ['revenue', 'cost', 'profit', 'profit_margin',
                    'revenue_growth_pct', 'expenses', 'expense_ratio',
                    'rolling_revenue', 'rolling_profit', 'rolling_expense_ratio']
        kpi_cols += [col for col in self.df.columns if '_returns' in col or '_volatility' in col]
        kpi_df = self.df[kpi_cols]
        return kpi_df

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Example: assume combined_df from market_data.py
    import pandas as pd
    combined_df = pd.read_csv("data/combined_sample.csv")
    
    fm = FinanceMetrics(combined_df)
    fm.compute_basic_metrics()
    fm.compute_expense_ratios()
    fm.compute_rolling_metrics()
    fm.compute_market_metrics(['AAPL_Close', 'MSFT_Close'])
    
    kpis = fm.get_kpis()
    print(kpis.head())