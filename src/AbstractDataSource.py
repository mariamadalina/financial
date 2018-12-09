import abc
from Indexes import Index as idx
import technicalIndicators as ti
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import numpy as np

class AbstractDataSource(object):

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
            dataFrame['AvgExp'] = ti.ema(dataFrame.Close, 10)
            dataFrame['%K'] = ti.STOK(dataFrame.Close, dataFrame.Low, dataFrame.High, 10)
            dataFrame['ROC']=ti.ROC(dataFrame.Close,10)
            dataFrame['RSI']=ti.RSI(dataFrame.Close)
            dataFrame['ACCUMULATION_DISTRIBUTION']=ti.AccDO(dataFrame.Close,dataFrame.High,dataFrame.Low)
            dataFrame['MACD']=ti.MACD(dataFrame.Close)
            dataFrame['WILLIAMS']=ti.WilliamsR(dataFrame.Close,dataFrame.High,dataFrame.Low)
            dataFrame['HIGH_PRICE_ACCELERATION']=ti.HPA(dataFrame.High,14)
            dataFrame['DISPARITY_5']=ti.Disparity(dataFrame.Close,5)
            dataFrame['DISPARITY_10']=ti.Disparity(dataFrame.Close,10)

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


    


    
