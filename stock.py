import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="ARIMA Stock Forecasting",
    layout="wide"
)

st.title("📈 Indian Stock Forecasting using ARIMA")

ticker = st.text_input(
    "Enter NSE Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:

        # Download 5 years data
        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        # Fix yfinance MultiIndex issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Close"]].dropna()

        st.subheader("Historical Data")
        st.dataframe(data.tail())

        # Monthly closing prices
        monthly_data = data.resample("M").last()

        # ARIMA Model
        model = ARIMA(
            monthly_data["Close"],
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        st.subheader("ARIMA Model Details")

        st.write("ARIMA Order: (5,1,0)")
        st.write(f"AIC: {round(model_fit.aic,2)}")
        st.write(f"BIC: {round(model_fit.bic,2)}")

        # Forecast till June 2027
        forecast_end = pd.Timestamp("2027-06-30")
        last_date = monthly_data.index[-1]

        months = (
            (forecast_end.year - last_date.year) * 12
            + forecast_end.month
            - last_date.month
        )

        if months < 1:
            months = 12

        forecast_result = model_fit.get_forecast(
            steps=months
        )

        forecast_values = forecast_result.predicted_mean

        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=months,
            freq="M"
        )

        forecast_df = pd.DataFrame(
            {
                "Forecast Price": forecast_values.values
            },
            index=future_dates
        )

        st.subheader("Forecast Data")
        st.dataframe(forecast_df)

        # June 2027 Forecast
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
                "Expected Price in June 2027",
                f"₹ {predicted_price}"
            )

            st.dataframe(june_2027)

        # Graph
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data["Close"],
                mode="lines",
                name="Historical Prices"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["Forecast Price"],
                mode="lines+markers",
                name="ARIMA Forecast"
            )
        )

        fig.add_vline(
            x=monthly_data.index[-1],
            line_dash="dash",
            annotation_text="Forecast Start"
        )

        fig.update_layout(
            title=f"{ticker} ARIMA Forecast till June 2027",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=650
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # CSV Download
        csv = forecast_df.to_csv().encode("utf-8")

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            f"{ticker}_forecast.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")
