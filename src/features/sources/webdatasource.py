
import pandas as pd
import pandas_datareader.data as web
import features.sources.base as s


class WebDataSource(s._AbstractDataSource):
    def __init__(self,start_date, end_date, ticker):
        self._ticker = ticker
        self._startDate = start_date
        self._end_date = end_date
        super().__init__()

    def _readData(self):

        reader = web.DataReader(self._ticker,'yahoo',self._startDate,self._end_date)
        df = pd.DataFrame()
        df['Close']=reader.Close
        df['Open']=reader.Open
        df['High']=reader.High
        df['Low']=reader.Low
        df['Volume']=reader.Volume
        return df