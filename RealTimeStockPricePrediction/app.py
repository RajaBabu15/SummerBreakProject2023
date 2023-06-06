import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model 
import streamlit as st
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler

today = date.today()
end_date = today.strftime("%Y-%m-%d")
d1 = date.today() - timedelta(days=360*10) 
start_date = d1.strftime("%Y-%m-%d")

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker','GOOGL')

df = yf.download(tickers = user_input,start = start_date,end = end_date)
df= pd.DataFrame(df)
df = df.reset_index()
# Descibing the data
st.subheader('Past 10 year Data')
st.write(df.describe())

# Visualizng 
st.subheader('Closing Price vs Time Chart')
movin_avg_100 = df.Close.rolling(100).mean()
movin_avg_200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,label = 'Stock Data')
plt.plot(df.Date,movin_avg_100,'r',label='100 Days Moving Average')
plt.plot(df.Date,movin_avg_200,'g',label='200 Days Moving Average')
plt.legend()
plt.xlabel ('Time')
plt.ylabel ('Price')
plt.title(str(user_input)+' Stock Price')
st.pyplot(fig)

input_data = []
scaler = MinMaxScaler(feature_range=(0,1))
data=pd.DataFrame(df['Close'])
data = scaler.fit_transform(data)
input_data = []
for i in range(100,data.shape[0]):
    input_data.append(data[i-100:i])
input_data=np.array(input_data)
model = load_model('keras_model.h5')

predicted_data = model.predict(input_data)
scale_factor = 1/(scaler.scale_[0])
y_predicted = predicted_data*scale_factor

fig = plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,'b',label='Original')
plt.plot(df.Date[100:],pd.DataFrame(y_predicted)[0],'g',label = 'Predicted')
plt.legend()
plt.xlabel ('Time')
plt.ylabel ('Price')
plt.title (str(user_input)+'Stock Price')
st.pyplot(fig)