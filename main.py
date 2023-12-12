import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constraints for data range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Web app title
st.title("Stock Prediction")

# Stock options
stocks = ("AAPL", "MSFT", "GOOG", "IBM")
selected_stock = st.selectbox("Select a stock", stocks)

# Slider for selecting number of years for prediction
num_years = st.slider("Years of predicition", 1, 4)
period = num_years * 365


# Function to load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


# Display loading messages
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

# Display raw data
st.subheader("Raw data")
st.write(data.tail())


# Function to plot raw stock data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Display raw stock data plot
plot_raw_data()

# Prepare data for Prophet model
dataframe_train = data[["Date","Close"]]
dataframe_train = dataframe_train.rename(columns={"Date": "ds", "Close": "y"})

# Create and fit Prophet model
m = Prophet()
m.fit(dataframe_train)

# Create future dataframe for predictions
future = m.make_future_dataframe(periods=period)

# Generate forecast
forecast = m.predict(future)

# Display forecast data
st.subheader("Forecast data")
st.write(forecast.tail())

# Display forecast plot
st.write("forecast data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components plot
st.write("forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)