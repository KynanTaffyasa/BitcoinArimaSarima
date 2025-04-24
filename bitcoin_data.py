import requests
import pandas as pd
from datetime import datetime

def fetch_bitcoin_prices(start_date: datetime, end_date: datetime):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    params = {
        "vs_currency": "usd",
        "from": start_timestamp,
        "to": end_timestamp
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "prices" in data:
            prices = data["prices"]
            df = pd.DataFrame(prices, columns=["Timestamp", "Price"])
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.date
            daily_close_df = df.groupby("Date").last().reset_index()
            return daily_close_df
        else:
            return None
    else:
        raise Exception(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
