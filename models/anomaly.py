import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Detect anomalies in financial and portfolio data.
    Supports Z-score, Rolling deviation, and Isolation Forest methods.
    """

    def __init__(self, df):
        """
        Initialize anomaly detector.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing financial metrics and market data.
        """

        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")

        self.df = df.copy()
        self.anomaly_flags = pd.DataFrame(index=self.df.index)

    # --------------------------------------------------
    # Z-SCORE ANOMALY DETECTION
    # --------------------------------------------------
    def detect_zscore(self, column, threshold=3):
        """
        Detect anomalies using Z-score method.
        """

        if column not in self.df.columns:
            print(f"Skipping Z-score: column '{column}' not found")
            return pd.Series([False] * len(self.df), index=self.df.index)

        series = self.df[column].astype(float)

        std = series.std()

        if std == 0 or np.isnan(std):
            self.anomaly_flags[f"{column}_zscore_anomaly"] = False
            return self.anomaly_flags[f"{column}_zscore_anomaly"]

        z_score = (series - series.mean()) / std

        self.anomaly_flags[f"{column}_zscore_anomaly"] = abs(z_score) > threshold

        return self.anomaly_flags[f"{column}_zscore_anomaly"]

    # --------------------------------------------------
    # ROLLING WINDOW ANOMALY DETECTION
    # --------------------------------------------------
    def detect_rolling(self, column, window=3, threshold_pct=30):
        """
        Detect anomalies based on deviation from rolling mean.
        """

        if column not in self.df.columns:
            print(f"Skipping Rolling detection: column '{column}' not found")
            return pd.Series([False] * len(self.df), index=self.df.index)

        series = self.df[column].astype(float)

        rolling_mean = series.rolling(window=window, min_periods=1).mean()

        deviation = abs(series - rolling_mean) / rolling_mean.replace(0, np.nan) * 100

        self.anomaly_flags[f"{column}_rolling_anomaly"] = deviation > threshold_pct

        self.anomaly_flags[f"{column}_rolling_anomaly"].fillna(False, inplace=True)

        return self.anomaly_flags[f"{column}_rolling_anomaly"]

    # --------------------------------------------------
    # ISOLATION FOREST ANOMALY DETECTION
    # --------------------------------------------------
    def detect_isolation_forest(self, columns):
        """
        Multi-dimensional anomaly detection using Isolation Forest.
        """

        if not columns:
            print("Isolation Forest skipped: no columns provided")
            return pd.Series([False] * len(self.df), index=self.df.index)

        valid_columns = [c for c in columns if c in self.df.columns]

        if not valid_columns:
            print("Isolation Forest skipped: no valid columns found")
            return pd.Series([False] * len(self.df), index=self.df.index)

        data = self.df[valid_columns].copy()

        if data.empty:
            print("Isolation Forest skipped: data is empty")
            return pd.Series([False] * len(self.df), index=self.df.index)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(0)

        try:
            iso = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            preds = iso.fit_predict(data)

            self.anomaly_flags["isolation_forest"] = preds == -1

        except Exception as e:
            print("Isolation Forest failed:", str(e))
            self.anomaly_flags["isolation_forest"] = False

        return self.anomaly_flags["isolation_forest"]

    # --------------------------------------------------
    # AGGREGATE ANOMALY SCORE
    # --------------------------------------------------
    def aggregate_anomalies(self):
        """
        Combine all anomaly flags into a normalized anomaly score.
        """

        if self.anomaly_flags.empty:
            print("No anomaly methods were executed.")
            return pd.Series([0] * len(self.df), index=self.df.index)

        score = self.anomaly_flags.sum(axis=1) / len(self.anomaly_flags.columns)

        self.anomaly_flags["anomaly_score"] = score

        return self.anomaly_flags["anomaly_score"]


# --------------------------------------------------
# Example Usage
# --------------------------------------------------
if __name__ == "__main__":

    try:
        combined_df = pd.read_csv("data/combined_sample.csv")

        ad = AnomalyDetector(combined_df)

        ad.detect_zscore("revenue")
        ad.detect_rolling("profit", window=3)
        ad.detect_isolation_forest(["AAPL_Close", "MSFT_Close"])

        print("Aggregate Anomaly Score:\n", ad.aggregate_anomalies().head())

    except Exception as e:
        print("Example failed:", str(e))