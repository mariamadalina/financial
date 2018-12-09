
import os
import pandas as pd
import technicalIndicators as ti
from Indexes import Index as idx
from AbstractDataSource import AbstractDataSource


class FileDataSource(AbstractDataSource):
    def __init__(self, filePath):
        self._location = filePath
        self._dataFrame = self._readData()
        super().__init__()

    def _curateDataframe(self,dataFrame: pd.DataFrame):
        dataFrame.columns = dataFrame.columns.str.replace(" ","")
        dataFrame['Date'] = dataFrame[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1).apply(lambda x: pd.to_datetime(x))
        dataFrame.set_index('Date')
        return super()._curateDataframe(dataFrame)

    def _readData(self):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir, self._location)
        filename = os.path.abspath(os.path.realpath(filename))
        df = pd.read_csv(filename, sep=',',parse_dates=True)
        return df   


class WebDataSource(AbstractDataSource):
    def __init__(self,start_date, end_date, ticker):
        self._ticker = ticker
        self._startDate = start_date
        self._end_date = end_date
        self._dataFrame = self._readData()
        super().__init__()

    def _readData(self):
        data =idx(self._ticker,self._startDate,self._end_date)
        df = pd.DataFrame()
        df['Close']=data.Close
        df['Open']=data.Open
        df['High']=data.High
        df['Low']=data.Low
        df['Volume']=data.Volume
        return df