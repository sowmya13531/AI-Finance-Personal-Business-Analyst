import pandas as pd
from prophet import Prophet


def forecast_revenue(df, periods=3):

    # copy dataframe to avoid modifying original
    data = df.copy()

    # ensure correct column names
    if "date" not in data.columns or "revenue" not in data.columns:
        return None

    # convert date
    data["date"] = pd.to_datetime(data["date"])

    # rename for prophet
    data = data.rename(columns={
        "date": "ds",
        "revenue": "y"
    })

    # sort values
    data = data.sort_values("ds")

    # build model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    # train
    model.fit(data)

    # create future dataframe
    future = model.make_future_dataframe(
        periods=periods,
        freq="M"
    )

    # predict
    forecast = model.predict(future)

    # clean output
    result = forecast[["ds", "yhat"]].tail(periods)

    result["yhat"] = result["yhat"].round(2)

    return result