
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.fft import fft,fftfreq

from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
model = load_model('fft_model.h5')
import yfinance as yf
today = date.today()
end_date = today.strftime("%Y-%m-%d")
d1 = date.today() - timedelta(days=360*10)
start_date = d1.strftime("%Y-%m-%d")

df = yf.download(tickers = 'GOOGL',start = start_date,end = end_date)
df= pd.DataFrame(df)
df = df.reset_index()
import matplotlib.pyplot as plt
def transformation(fft_data,isPlot=False):
    scaler = MinMaxScaler(feature_range=(-1,1))
    batch1 = scaler.fit_transform(np.array(fft_data).reshape(-1,1)).reshape(-1)
    yf = fft(batch1)
    yf1 = 2.0/len(batch1) * np.abs(yf[0:len(batch1)//2])
    xf = fftfreq(len(batch1),1)[:len(batch1)//2]
    # if isPlot:
    #     plt.plot(batch1)
    #     plt.show()
    #     plt.plot(xf,yf1)
    #     plt.grid()
    #     plt.show()
    return np.array((xf,yf[0:len(batch1)//2].real,yf[0:len(batch1)//2].imag)).T

fig = plt.figure(figsize=(12,6))
plt.plot(df.Date,df.Close,'r',label = 'Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('GOOGLE Stock Price')
plt.legend()
plt.plot()
plt.show()
batch = []
data = df.Close
for i in range(128,len(df)):
    batch.append(data[i-128:i])
batch = np.array(batch)
transformed_test_data = [transformation(i) for i in batch]
y = []
for i in range(128,len(df)):
    y.append(data[i])
y=np.array(y)
transformed_data = np.array(transformed_test_data)
predicted_y = model.predict(transformed_test_data)