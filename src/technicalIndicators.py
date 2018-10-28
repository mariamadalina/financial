
import pandas as pd
import numpy as np

def STOK(close, low, high, n):  
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def ema(close, span):
    e = close.ewm(span=span, adjust=False).mean()
    return e

def ROC(close, n):  
    M = close.diff(n - 1)  
    N = close.shift(n - 1)  
    ROC = pd.Series(((M / N) * 100), name = 'ROC_' + str(n))   
    return ROC

def RSI(series, period=14):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(span=period-1, adjust=False).mean() / d.ewm(span=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def AccDO(close, high, low):
    ad = np.zeros(len(close))
    for idx in range(1, len(close)):
         ad[idx] =  (high[idx] - close[idx-1])/(high[idx] - low[idx]) 

    return ad

def MACD(series):
    return ema(series,12)-ema(series,26)

def WilliamsR(close,  high,low, n=14):  
    STOK =  (high.rolling(n).max() - close)/(high.rolling(n).max()-low.rolling(n).min()) * 100
    return STOK

def HPA(high, n):  
    STOK =  (high - high.rolling(n).max())/high.shift(n-1) * 100
    return STOK

def Disparity(close, range):
    return close*100/close.rolling(range).mean()