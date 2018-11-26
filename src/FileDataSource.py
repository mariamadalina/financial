from Indexes import Index as idx
import technicalIndicators as ti
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import os

class FileDataSource:
    def __init__(self, filePath, start_date, end_date):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir, filePath)
        filename = os.path.abspath(os.path.realpath(filename))
        df = pd.read_csv(filename, sep=',',parse_dates=True)
        df= self._curateDataframe(df)
       
        df.fillna(0,inplace=True)
        self.Data=df
    
    def _curateDataframe(self,dataFrame: pd.DataFrame):
        dataFrame.columns = dataFrame.columns.str.replace(" ","")

        dataFrame['Date'] = dataFrame[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1).apply(lambda x: pd.to_datetime(x))
        dataFrame.set_index('Date')
        notFoundColums = set(dataFrame.columns) - set (['Close', 'Open', 'High', 'Low', 'Volume', 'AvgExp', '%K', 'ROC', 'RSI','ACCUMULATION_DISTRIBUTION', 'MACD', 'WILLIAMS_R', 'HIGH_PRICE_ACCELERATION', 'DISPARITY_5','DISPARITY_10'])
        for column in notFoundColums:
          dataFrame.drop(column, axis=1, inplace=True)
        #dataFrame.columns=foundColums
        return dataFrame




   