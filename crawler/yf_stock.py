import yfinance as yf
import pandas as pd 
from datetime import datetime
from pandas_datareader import data
import numpy as np
yf.pdr_override()
import os

def extract_stock_price(stock, data_start='2022-05-01', data_end=datetime.today().strftime("%Y-%m-%d")):
  stock_code = stock.upper()
  start = datetime.strptime(data_start,'%Y-%m-%d')
  end = datetime.strptime(data_end,'%Y-%m-%d')

  df = data.DataReader(stock_code,start=start, end=end)

  # simple moving average - using the close stock price in each of the past 5 days
  df['SMA'] = df['Close'].shift().rolling(10).mean()

  df['next_0d'] = df['Close']; df['next_0d_sma'] = df['SMA']
  # extract next 5 days data
  for i in range(5):
    df['next_'+ str(i+1) + 'd'] = df['Close'].shift(-i-1)
    df['next_'+ str(i+1) + 'd_sma'] = df['SMA'].shift(-i-1)
  
  df['next_0d_trend'] = None
  # next 5 days data compare with past 5 days SMA
  for i in range(6):
    df.loc[(df['next_'+ str(i) + 'd'] > df['next_'+ str(i) + 'd_sma']), 'next_' + str(i) + 'd_trend'] = 'uptrend'   
    df.loc[(df['next_'+ str(i) + 'd'] < df['next_'+ str(i) + 'd_sma']), 'next_' + str(i) + 'd_trend'] = 'downtrend'  
    df.loc[(df['next_'+ str(i) + 'd'] == df['next_'+ str(i) + 'd_sma']), 'next_' + str(i) + 'd_trend'] = 'nochange'   

  return df