import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Indian Stock Forecasting using ARIMA",
    layout="wide"
)

st.title("📈 Indian Stock Forecasting using ARIMA")

ticker = st.text_input(
    "Enter NSE Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:

        # --------------------------
        # Download Last 5 Years Data
        # --------------------------
        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        # Handle MultiIndex issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Close"]].dropna()

        st.subheader("Historical Data")

        st.dataframe(data.tail())

        # --------------------------
        # Monthly Data
        # --------------------------
        monthly_data = data.resample("ME").last()

        monthly_data.index = pd.DatetimeIndex(
            monthly_data.index
        )

        # --------------------------
        # ARIMA Model
        # --------------------------
        model = ARIMA(
            monthly_data["Close"],
            order=(5, 1, 0)
        )

        model_fit = model.fit()

        st.subheader("ARIMA Model")

        st.write("Model Used: ARIMA(5,1,0)")
        st.write(f"AIC: {round(model_fit.aic,2)}")
        st.write(f"BIC: {round(model_fit.bic,2)}")

        # --------------------------
        # Forecast Till June 2027
        # --------------------------
        forecast_end = pd.Timestamp(
            "2027-06-30"
        )

        last_date = monthly_data.index[-1]

        months = (
            (forecast_end.year - last_date.year) * 12
            + forecast_end.month
            - last_date.month
        )

        if months <= 0:
            months = 12

        forecast = model_fit.forecast(
            steps=months
        )

        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=months,
            freq="ME"
        )

        forecast_df = pd.DataFrame(
            {
                "Forecast Price": forecast.values
            },
            index=future_dates
        )

        # --------------------------
        # June 2027 Prediction
        # --------------------------
        st.subheader("🎯 June 2027 Forecast")

        june_2027 = forecast_df[
            (forecast_df.index.year == 2027)
            &
            (forecast_df.index.month == 6)
        ]

        if not june_2027.empty:

            june_price = round(
                june_2027.iloc[-1]["Forecast Price"],
                2
            )

            st.metric(
                "Expected Price",
                f"₹ {june_price}"
            )

            st.dataframe(june_2027)

        # --------------------------
        # Forecast Table
        # --------------------------
        st.subheader("Forecast Data")

        st.dataframe(forecast_df)

        # --------------------------
        # Graph
        # --------------------------
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
            line_dash="dash"
        )

        fig.update_layout(
            title=f"{ticker} Forecast till June 2027",
            xaxis_title="Date",
            yaxis_title="Price",
            height=650
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------
        # CSV Download
        # --------------------------
        csv = forecast_df.to_csv().encode(
            "utf-8"
        )

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            f"{ticker}_forecast.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
