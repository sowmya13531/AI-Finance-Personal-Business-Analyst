import pandas as pd
import yfinance as yf

class MarketDataFetcher:
    """
    Fetches internal finance data and external market data,
    preprocesses it, and outputs unified DataFrames.
    """

    def __init__(self, internal_data_path=None, tickers=None):
        """
        :param internal_data_path: path to company CSV/Excel files
        :param tickers: list of stock tickers to fetch market data
        """
        self.internal_data_path = internal_data_path
        self.tickers = tickers if tickers else []

    # ------------------------------
    # Step 1: Fetch Internal Company Data
    # ------------------------------
    def fetch_internal_data(self):
        """
        Reads internal finance data (Excel/CSV) and preprocesses it.
        """
        if not self.internal_data_path:
            print("No internal data path provided.")
            return pd.DataFrame()
        
        # Example: read revenue/costs data
        df = pd.read_excel(self.internal_data_path)
        df.columns = df.columns.str.lower()  # standardize columns
        df['date'] = pd.to_datetime(df['date'])
        
        # Compute basic metrics
        df['profit'] = df['revenue'] - df['cost']
        df['profit_margin'] = df['profit'] / df['revenue'] * 100
        
        # Aggregate monthly/quarterly
        df_monthly = df.resample('M', on='date').sum()
        
        return df_monthly

    # ------------------------------
    # Step 2: Fetch External Market Data
    # ------------------------------
    def fetch_market_data(self):
        """
        Fetch historical stock prices & financials from yfinance.
        """
        all_data = {}
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}")
            t = yf.Ticker(ticker)
            
            # Historical price data
            hist = t.history(period="1y")  # last 1 year
            hist.reset_index(inplace=True)
            all_data[ticker] = hist
        
        return all_data

    # ------------------------------
    # Step 3: Combine Internal + Market Data
    # ------------------------------
    def get_combined_data(self):
        """
        Combines internal finance data and market data.
        Returns unified DataFrame(s) for ML and LLM modules.
        """
        internal_df = self.fetch_internal_data()
        market_data = self.fetch_market_data()
        
        # Combine logic depends on your use case
        # Example: merge on date if internal_df has 'date' column
        combined = internal_df.copy()
        for ticker, df in market_data.items():
            df.rename(columns=lambda x: f"{ticker}_{x}" if x not in ['Date'] else x, inplace=True)
            combined = combined.merge(df, left_on='date', right_on='Date', how='left')
        
        return combined

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'TSLA']
    fetcher = MarketDataFetcher(internal_data_path="data/revenue.xlsx", tickers=tickers)
    combined_df = fetcher.get_combined_data()
    print(combined_df.head())