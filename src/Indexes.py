import pandas_datareader.data as web

def get_stock(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['Close']

def get_high(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['High']

def get_low(stock,start,end):
     return web.DataReader(stock,'yahoo',start,end)['Low']