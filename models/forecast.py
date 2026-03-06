import pandas as pd
from prophet import Prophet

class FinanceForecaster:
    """
    Predicts future financial metrics using time series models (Prophet)
    """

    def __init__(self, df, target_column='revenue', period=30):
        """
        :param df: DataFrame with 'date' and target column
        :param target_column: column to forecast (revenue, cost, profit)
        :param period: number of days/months to forecast
        """
        self.df = df.copy()
        self.target_column = target_column
        self.period = period
        self.model = None
        self.forecast_df = None

    # ------------------------------
    # Step 1: Prepare Data for Prophet
    # ------------------------------
    def prepare_data(self):
        """
        Convert DataFrame to Prophet format
        """
        df_prophet = self.df[['date', self.target_column]].rename(columns={'date': 'ds', self.target_column: 'y'})
        return df_prophet

    # ------------------------------
    # Step 2: Train Prophet Model
    # ------------------------------
    def train_model(self):
        """
        Train Prophet time series model
        """
        df_prophet = self.prepare_data()
        self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.model.fit(df_prophet)
        return self.model

    # ------------------------------
    # Step 3: Make Forecast
    # ------------------------------
    def make_forecast(self):
        """
        Generate forecast for future periods
        """
        future = self.model.make_future_dataframe(periods=self.period)
        forecast = self.model.predict(future)
        self.forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return self.forecast_df

    # ------------------------------
    # Step 4: Visualize Forecast (Optional)
    # ------------------------------
    def plot_forecast(self):
        """
        Returns plot of forecasted values
        """
        import matplotlib.pyplot as plt
        if self.forecast_df is None:
            self.make_forecast()
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['date'], self.df[self.target_column], label='Actual')
        plt.plot(self.forecast_df['ds'], self.forecast_df['yhat'], label='Forecast')
        plt.fill_between(self.forecast_df['ds'], self.forecast_df['yhat_lower'], self.forecast_df['yhat_upper'], color='gray', alpha=0.2)
        plt.legend()
        plt.title(f'{self.target_column.capitalize()} Forecast')
        plt.show()

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    combined_df = pd.read_csv("data/combined_sample.csv")
    
    # Forecast revenue
    revenue_forecaster = FinanceForecaster(combined_df, target_column='revenue', period=30)
    revenue_forecaster.train_model()
    forecast_df = revenue_forecaster.make_forecast()
    print(forecast_df.tail())
    revenue_forecaster.plot_forecast()