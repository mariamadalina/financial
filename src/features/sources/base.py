import abc
from datetime import datetime
from dateutil.parser import parse
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


class _AbstractDataSource(object):

    __acceptedPrimeIndicators = ['Close', 'Open', 'High', 'Low', 'Volume']

    __acceptedDerivedIndicators = ['AvgExp', '%K', 'ROC', 'RSI','ACCUMULATION_DISTRIBUTION', 'MACD', 'WILLIAMS_R', 'HIGH_PRICE_ACCELERATION', 'DISPARITY_5','DISPARITY_10']

    __acceptedHeaders = set(__acceptedPrimeIndicators) | set(__acceptedDerivedIndicators)

    def __init__(self):
        self._dataFrame = self._readData()

    def _curateDataframe(self,dataFrame: pd.DataFrame):
        notFoundColums = set(dataFrame.columns) - set (self.__acceptedHeaders)
        for column in notFoundColums:
            dataFrame.drop(column, axis=1, inplace=True)
        return dataFrame

    def __computeDerivedIndicators(self,dataFrame: pd.DataFrame):
        intersection = set(dataFrame.columns) & set(self.__acceptedHeaders)

        if (len(intersection)!=len(self.__acceptedHeaders)):
            dataFrame['AvgExp'] = ema(dataFrame.Close, 10)
            dataFrame['%K'] = STOK(dataFrame.Close, dataFrame.Low, dataFrame.High, 10)
            dataFrame['ROC'] = ROC(dataFrame.Close,10)
            dataFrame['RSI'] = RSI(dataFrame.Close)
            dataFrame['ACCUMULATION_DISTRIBUTION'] = AccDO(dataFrame.Close,dataFrame.High,dataFrame.Low)
            dataFrame['MACD'] = MACD(dataFrame.Close)
            dataFrame['WILLIAMS'] = WilliamsR(dataFrame.Close,dataFrame.High,dataFrame.Low)
            dataFrame['HIGH_PRICE_ACCELERATION'] = HPA(dataFrame.High,14)
            dataFrame['DISPARITY_5'] = Disparity(dataFrame.Close,5)
            dataFrame['DISPARITY_10'] = Disparity(dataFrame.Close,10)

        return dataFrame

    @property
    def dataFrame(self):
        return self._dataFrame

    @dataFrame.setter
    def dataFrame(self,value):
        self._dataFrame=value

    def Data(self):
        df = self._curateDataframe(self._dataFrame)
        df = self.__computeDerivedIndicators(self._dataFrame)
        return df

    def GetData(self, df:pd.DataFrame, numberOfFeatures = 5,):
        X = df.iloc[:,0:numberOfFeatures].values
        print("Value of X \n {0}".format(X))
        return X
   

    def GetMissingValueInfo(self, df:pd.DataFrame):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    @abc.abstractmethod
    def _readData(self): pass


    


    
