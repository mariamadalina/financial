import pandas_datareader.data as web

class Index:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.Close= _get_stock(ticker,start,end)
        self.High=_get_high(ticker,start,end)
        self.Low=_get_low(ticker,start,end)


def _get_stock(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['Close']

def _get_high(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['High']

def _get_low(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['Low']