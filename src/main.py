import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import FastICA, PCA


import Indexes as idx
import technicalIndicators as ti

start_date = '2010-08-03'
end_date = '2015-05-23'

figclose, ax = plt.subplots(figsize=(16,9))

data =idx.Index('FB', start_date, end_date)


df = pd.DataFrame()
df['EMA'] = ti.ema(data.Close, 10)
df['%K'] = ti.STOK(data.Close, data.Low, data.High, 10)
df['ROC']=ti.ROC(data.Close,10)
df['RSI']=ti.RSI(data.Close)
df['AccDo']=ti.AccDO(data.Close,data.High,data.Low)
df['MACD']=ti.MACD(data.Close)
df['WilliamsR']=ti.WilliamsR(data.Close,data.High,data.Low)
df['High Price accelerations']=ti.HPA(data.High,14)
df['Disparity 5']=ti.Disparity(data.Close,5)
df['Disparity 10']=ti.Disparity(data.Close,10)

ax.plot(data.High.loc[start_date:end_date], label = 'Span 20-days EMA')
df.tail()
