\# Time Series Forecasting – Bitcoin (BTC)



This project applies recurrent neural networks (RNNs) to forecast Bitcoin closing prices one hour ahead using raw Coinbase and Bitstamp minute-by-minute data.



\## Objectives

\- Understand time series forecasting and stationarity

\- Engineer a sliding window (24-hour lookback → 1-hour prediction)

\- Build a TensorFlow data pipeline using `tf.data.Dataset`

\- Train an LSTM/GRU RNN on preprocessed financial data

\- Evaluate its performance and visualize predictions



\## Files

\- `preprocess\_data.py` – Cleans, filters, scales, and prepares BTC data

\- `forecast\_btc.py` – Builds, trains, and evaluates the RNN model

\- `README.md` – This documentation



\## Data

Each row represents a 60-second window containing:

Unix start time, open, high, low, close, BTC volume, USD volume, and VWAP.



\## Model

Uses past 24 hours (1440 minutes) to predict the next hour closing price.



\## Requirements

\- Python 3.9

\- numpy 1.25.2

\- pandas 2.2.2

\- tensorflow 2.15

\- pycodestyle 2.11.1





