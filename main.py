import numpy as np
import matplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Loading data
company = 'NVDA' #NVIDIA NASDAQ ticker symbol

start = dt.datetime(2020, 1, 1)
end = dt.datetime(2023, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)