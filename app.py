import streamlit as st
from bitcoin_data import fetch_bitcoin_prices
from datetime import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Bitcoin Forecast with ARIMA/SARIMA", layout="centered")

st.title("📈 Bitcoin Price Forecast (ARIMA / SARIMA) + Sentiment Viewer")

# Input date range
start_date = st.date_input("Start Date for Bitcoin", datetime.now().date())
end_date = st.date_input("End Date for Bitcoin", datetime.now().date())

start_datetime = datetime.combine(start_date, time.min)
end_datetime = datetime.combine(end_date, time.max)

# Forecasting options BEFORE fetch
model_choice = st.selectbox("Choose Model for Forecasting", ["ARIMA", "SARIMA"])
forecast_days = st.number_input("📅 Number of days to forecast", min_value=1, max_value=30, value=7)

# -------------------------------
# 📌 Sentiment Data Section
# -------------------------------
st.subheader("💬 Sentiment Data Input")

sentiment_start_date = st.date_input("Sentiment Start Date", start_date)
sentiment_end_date = st.date_input("Sentiment End Date", end_date)

sentiment_dates = pd.date_range(sentiment_start_date, sentiment_end_date).strftime('%Y-%m-%d').tolist()
sentiment_values = st.text_area("Enter Sentiment Values (-1, 0, or 1, comma-separated)", 
                                value=",".join(["0" for _ in range(len(sentiment_dates))]))

try:
    sentiment_list = list(map(int, sentiment_values.split(',')))
    if len(sentiment_list) != len(sentiment_dates):
        st.warning("⚠️ Number of sentiment values must match number of dates.")
    else:
        sentiment_df = pd.DataFrame({
            "Date": pd.to_datetime(sentiment_dates),
            "Sentiment": sentiment_list
        })
        st.write("🧠 Sentiment Data:", sentiment_df)

        # Plot sentiment
        fig_s, ax_s = plt.subplots(figsize=(10, 4))
        ax_s.plot(sentiment_df["Date"], sentiment_df["Sentiment"], marker='o', color='purple')
        ax_s.set_title("Sentiment Over Time")
        ax_s.set_xlabel("Date")
        ax_s.set_ylabel("Sentiment")
        ax_s.grid(True)
        st.pyplot(fig_s)
except Exception as e:
    st.error(f"❌ Error in sentiment input: {e}")

# -------------------------------
# 📈 Bitcoin Data Fetch + Forecast
# -------------------------------
if st.button("Fetch Bitcoin Price Data"):
    if end_datetime < start_datetime:
        st.error("⚠️ End date must be after start date.")
    else:
        with st.spinner("Fetching Bitcoin data..."):
            try:
                df = fetch_bitcoin_prices(start_datetime, end_datetime)

                if df is not None and not df.empty:
                    st.success("✅ Data fetched successfully!")

                    # Normalize column names
                    if 'timestamp' in df.columns:
                        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
                    elif 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])

                    if 'price' in df.columns:
                        df['Price'] = df['price']
                    elif 'Price' not in df.columns:
                        # Try to infer
                        possible_price_col = [col for col in df.columns if 'price' in col.lower()]
                        if possible_price_col:
                            df['Price'] = df[possible_price_col[0]]
                        else:
                            st.error("❌ No valid 'Price' column found.")
                            st.stop()

                    df = df[['Date', 'Price']]
                    df = df.set_index('Date').resample('D').mean().interpolate()

                    st.line_chart(df)

                    # Forecasting
                    st.subheader(f"📊 Forecasting with {model_choice} for {forecast_days} days")

                    if model_choice == "ARIMA":
                        model = ARIMA(df['Price'], order=(5, 0, 1))
                    else:
                        model = SARIMAX(df['Price'], order=(5, 0, 1), seasonal_order=(1, 1, 1, 7))

                    results = model.fit()
                    forecast = results.forecast(steps=forecast_days)
                    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

                    # Plotting
                    fig, ax = plt.subplots(figsize=(24, 12))
                    ax.plot(df.index, df['Price'], label='Actual Price', color='blue', marker='o')
                    ax.plot(forecast_dates, forecast, label='Forecasted Price', color='orange', linestyle='--', marker='o')

                    for i, val in enumerate(forecast):
                        ax.annotate(f"${val:.2f}", (forecast_dates[i], val),
                                    textcoords="offset points", xytext=(0, 12),
                                    ha='center', fontsize=9, color='orange', weight='bold')

                    ax.set_title(f"Bitcoin Price Forecast ({model_choice})", fontsize=16)
                    ax.set_xlabel("Date", fontsize=12)
                    ax.set_ylabel("Price", fontsize=12)
                    ax.tick_params(axis='x', rotation=30)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)

                    ax.set_title(f"Bitcoin Price Forecast ({model_choice})")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Price': forecast})
                    st.write("🔮 Forecasted Prices:", forecast_df)

                    # Download CSV
                    csv = df.reset_index().to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Bitcoin Data as CSV",
                        data=csv,
                        file_name=f"bitcoin_prices_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No data available.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
