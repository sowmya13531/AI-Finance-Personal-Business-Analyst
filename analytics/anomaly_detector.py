from sklearn.ensemble import IsolationForest
import pandas as pd


def detect_cost_anomaly(df, contamination=0.1):

    # copy to avoid modifying original data
    data = df.copy()

    # validation
    if "amount" not in data.columns:
        return None, None

    # reshape values
    X = data[["amount"]]

    # model
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    # predictions
    preds = model.fit_predict(X)

    # add labels
    data["anomaly"] = preds

    # anomalies = -1
    anomalies = data[data["anomaly"] == -1]

    return data, anomalies