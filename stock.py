import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="ARIMA Stock Forecasting",
    layout="wide"
)

st.title("📈 Indian Stock Forecasting using ARIMA")

st.write(
    "Forecast stock prices using the last 5 years of Yahoo Finance data."
)

ticker = st.text_input(
    "Enter NSE Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:

        # -----------------------------
        # Download Data
        # -----------------------------
        with st.spinner("Downloading stock data..."):

            data = yf.download(
                ticker,
                period="5y",
                progress=False,
                auto_adjust=True
            )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        data = data[["Close"]].dropna()

        st.subheader("Last 5 Years Historical Data")

        st.dataframe(data.tail())

        # -----------------------------
        # Monthly Resampling
        # -----------------------------
        monthly_data = data.resample("ME").last()

        st.subheader("Monthly Closing Prices")

        st.dataframe(monthly_data.tail())

        # -----------------------------
        # Auto ARIMA
        # -----------------------------
        with st.spinner("Finding best ARIMA model..."):

            auto_model = auto_arima(
                monthly_data["Close"],
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
                max_p=5,
                max_q=5
            )

        order = auto_model.order

        st.success(f"Selected ARIMA Order: {order}")

        # -----------------------------
        # Train Model
        # -----------------------------
        model = ARIMA(
            monthly_data["Close"],
            order=order
        )

        model_fit = model.fit()

        # -----------------------------
        # Forecast Till June 2027
        # -----------------------------
        forecast_end = pd.Timestamp("2027-06-30")

        last_date = monthly_data.index[-1]

        months = (
            (forecast_end.year - last_date.year) * 12
            + forecast_end.month
            - last_date.month
        )

        if months <= 0:
            months = 12

        forecast_result = model_fit.get_forecast(
            steps=months
        )

        forecast_values = forecast_result.predicted_mean

        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=months,
            freq="ME"
        )

        forecast_df = pd.DataFrame(
            {
                "Forecast Price": forecast_values.values
            },
            index=future_dates
        )

        # -----------------------------
        # June 2027 Forecast
        # -----------------------------
        june_2027 = forecast_df[
            (forecast_df.index.year == 2027)
            & (forecast_df.index.month == 6)
        ]

        st.subheader("🎯 June 2027 Forecast")

        if not june_2027.empty:

            predicted_price = round(
                june_2027["Forecast Price"].iloc[-1],
                2
            )

            st.metric(
                "Expected Stock Price",
                f"₹ {predicted_price}"
            )

            st.dataframe(june_2027)

        # -----------------------------
        # Forecast Table
        # -----------------------------
        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df
        )

        # -----------------------------
        # Graph
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data["Close"],
                mode="lines+markers",
                name="Historical Prices"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["Forecast Price"],
                mode="lines+markers",
                name="Forecast Prices"
            )
        )

        fig.update_layout(
            title=f"{ticker} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=650
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------
        # Download CSV
        # -----------------------------
        csv = forecast_df.to_csv().encode("utf-8")

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
