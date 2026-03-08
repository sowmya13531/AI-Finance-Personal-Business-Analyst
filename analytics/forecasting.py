from prophet import Prophet

def forecast_revenue(df):

    df = df.rename(columns={"date":"ds","revenue":"y"})

    model = Prophet()

    model.fit(df)

    future = model.make_future_dataframe(periods=3,freq="M")

    forecast = model.predict(future)

    return forecast[["ds","yhat"]].tail(3)