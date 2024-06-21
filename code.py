import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def input_parsing(file):
    data = pd.read_csv(file, delimiter='\t', names=['Time', 'Price'], skiprows=1, parse_dates=['Time'], dayfirst=True)
    data.set_index('Time', inplace=True)
    return data

def normalize(data):
    return (data['Price'] - data['Price'].min()) / (data['Price'].max() - data['Price'].min())


def formatting(btc_data, eth_data):
    start_date = max(btc_data.index.min(), eth_data.index.min())
    end_date = min(btc_data.index.max(), eth_data.index.max())
    btc_aligned = btc_data.loc[start_date:end_date].copy()
    eth_aligned = eth_data.loc[start_date:end_date].copy()
    btc_aligned['Normalized Price'] = normalize(btc_aligned)
    eth_aligned['Normalized Price'] = normalize(eth_aligned)
    return btc_aligned, eth_aligned


'''Engle-Granger 2step cointegration test
1. regress projection ETH to BTC
2. test residuals by ADF test '''
def cointegration_test(btc_prices, eth_prices):
    y = eth_prices.values
    x = sm.add_constant(btc_prices.values)
    model = sm.OLS(y, x).fit()
    residuals = model.resid

    adf_result = adfuller(residuals)
    return adf_result


def graphing(btc_data, eth_data):
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data.index, btc_data['Normalized Price'], label='BTC Normalized', color='orange')
    plt.plot(eth_data.index, eth_data['Normalized Price'], label='ETH Normalized', color='darkblue')
    plt.title('Normalized Price of BTC and ETH')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.grid(True)
    plt.legend()
    plt.show()


btc, eth = formatting(input_parsing('btc_prices.tsv'), input_parsing('eth_prices.tsv'))
res = cointegration_test(btc['Normalized Price'], eth['Normalized Price'])

print('ADF:', res[0])
print('p-value:', res[1])
print('critical values:', res[4])
graphing(btc, eth)
