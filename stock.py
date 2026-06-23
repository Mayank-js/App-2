import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Stock Forecasting App", layout="wide")

st.title("📈 Indian Stock Forecasting using ARIMA")
st.write("Forecast Indian stock prices using last 5 years Yahoo Finance data.")

ticker = st.text_input(
    "Enter NSE Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Generate Forecast"):

    try:

        # Download last 5 years data
        data = yf.download(
            ticker,
            period="5y",
            progress=False
        )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        data = data[['Close']].dropna()

        st.subheader("Historical Data")
        st.dataframe(data.tail())

        # Auto ARIMA
        st.subheader("Finding Best ARIMA Model...")

        auto_model = auto_arima(
            data['Close'],
            seasonal=False,
            suppress_warnings=True,
            stepwise=True
        )

        order = auto_model.order

        st.success(f"Best ARIMA Order: {order}")

        # Train model
        model = ARIMA(
            data['Close'],
            order=order
        )

        model_fit = model.fit()

        # Forecast till June 2027
        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            end="2027-06-30",
            freq="B"
        )

        forecast_steps = len(future_dates)

        forecast_result = model_fit.get_forecast(
            steps=forecast_steps
        )

        forecast = forecast_result.predicted_mean
        confidence = forecast_result.conf_int()

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast Price": forecast.values
        })

        forecast_df.set_index("Date", inplace=True)

        st.subheader("Forecasted Prices")
        st.dataframe(forecast_df.head(20))

        # June 2027 forecast only
        june2027 = forecast_df[
            (forecast_df.index.month == 6) &
            (forecast_df.index.year == 2027)
        ]

        st.subheader("June 2027 Forecast")

        st.dataframe(june2027)

        # Graph
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Prices'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast Price'],
                mode='lines',
                name='Forecast Prices'
            )
        )

        fig.update_layout(
            title=f"{ticker} Stock Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Download CSV
        csv = forecast_df.to_csv().encode("utf-8")

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
