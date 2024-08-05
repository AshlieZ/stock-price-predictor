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

start = dt.datetime(2012, 1, 1) #Pre-covid, post Crpyto mining boom
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)

#Preparing Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 90 #Amount of days used to predict future price

x_trainData = []
y_trainData = []

for x in range(prediction_days, len(scaled_data)):
    x_trainData.append(scaled_data[x-prediction_days:x, 0])
    y_trainData.append(scaled_data[x,0])

x_trainData, y_trainData = np.array(x_trainData), np.array(y_trainData) #Create into numpy array
x_trainData = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building The Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_trainData.shape[1], 1)))
model.add(Dropout(0.2))