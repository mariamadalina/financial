import pandas_datareader.data as web
from datetime import datetime
class Index:
    def __init__(self, ticker, start, end):
        self.ticker = ticker

        reader = web.DataReader(ticker,'yahoo',start,end)

        self.Open=reader['Open']
        self.Close= reader['Close']
        self.High=reader['High']
        self.Low=reader['Low']
        self.Volume=reader['Volume']
