import pandas as pd
from prophet import Prophet
from app.utils.cleaner import clean_sales_data

def forecast_from_csv(file, sku_id):
    df = clean_sales_data(file)

    # Filter by SKU
    sku_df = df[df["sku_id"] == int(sku_id)]

    if sku_df.empty:
        raise ValueError(f"SKU {sku_id} not found in the dataset.")

    # Weekly aggregated data
    weekly_df = sku_df.groupby("week").agg({"units_sold": "sum"}).reset_index()
    weekly_df.rename(columns={"week": "ds", "units_sold": "y"}, inplace=True)

    model = Prophet()
    model.fit(weekly_df)

    future = model.make_future_dataframe(periods=4, freq='W')
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].tail(10)
