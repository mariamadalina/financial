import sys
import os
import pandas as pd
import features.sources.base as s


class FileDataSource(s._AbstractDataSource):
    def __init__(self, filePath):
        self._location = filePath
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


