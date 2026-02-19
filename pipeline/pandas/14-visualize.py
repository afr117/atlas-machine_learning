#!/usr/bin/env python3
"""Module to transform and visualize Coinbase data."""

import matplotlib.pyplot as plt
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column if exists
if 'Weighted_Price' in df.columns:
    df = df.drop(columns=['Weighted_Price'])

# Rename Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert Unix timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set Date as index
df = df.set_index('Date')

# Fill missing Close values forward
df['Close'] = df['Close'].fillna(method='ffill')

# Fill High, Low, Open with corresponding Close values if missing
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

# Fill missing volumes with 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 onwards
df = df[df.index >= '2017-01-01']

# Resample to daily frequency and aggregate
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot daily data
df_daily.plot(figsize=(12, 6), title='Daily OHLC and Volume')
plt.xlabel('Date')
plt.ylabel('Price / Volume')
plt.tight_layout()
plt.savefig('14visualizepandas.png')
plt.show()

# Return the transformed DataFrame before plotting
df_daily
