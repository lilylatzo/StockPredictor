Stock Prediction App using Streamlit and Prophet

This Python program leverages the Streamlit framework to create an interactive web application for predicting stock prices. The application integrates financial data from Yahoo Finance (via the yfinance library) and employs the time-series forecasting capabilities of the Prophet library to generate predictions.

User-Friendly Interface:
The web app is accessible through a user-friendly interface created with Streamlit.
Users can select a stock of interest (Apple, Microsoft, Google, or IBM) from a dropdown menu.

Customizable Prediction Period:
A slider allows users to specify the number of years into the future they want predictions for.

Data Loading and Display:
Stock data is loaded using the yfinance library, with a specified start date and the current date.
Loading messages inform users about the data retrieval process, ensuring a smooth experience.

Visualization of Raw Data:
The app displays the raw stock data, including the closing and opening prices, using an interactive Plotly chart.

Prophet Time Series Forecasting:
The program uses the Prophet library to create and train a time-series forecasting model on the closing prices of the selected stock.
The future dataframe is generated to make predictions for the specified period.

Forecast Data Presentation:
The forecasted data, including predicted closing prices, is presented in a table format.

Interactive Forecast Plots:
Users can visualize the forecasted closing prices through an interactive Plotly chart, providing a clear representation of predicted trends.

Component Analysis:
The app displays component analysis plots generated by Prophet, breaking down the forecast into trend, seasonality, and holidays.
