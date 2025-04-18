import pandas as pd

def clean_sales_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Basic preprocessing
    df["week"] = pd.to_datetime(df["week"])
    df = df.dropna(subset=["week", "sku_id", "units_sold"])

    # Ensure correct types
    df["sku_id"] = df["sku_id"].astype(int)
    df["units_sold"] = df["units_sold"].astype(float)

    return df
