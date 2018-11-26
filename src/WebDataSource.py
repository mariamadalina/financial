from Indexes import Index as idx
import technicalIndicators as ti
import pandas as pd

class WebDataSource:
    def __init__(self,ticker, start_date, end_date):
        data =idx(ticker, start_date, end_date)
        df = pd.DataFrame()
        df['Close']=data.Close
        df['Open']=data.Open
        df['High']=data.High
        df['Low']=data.Low
        df['Volume']=data.Volume
        df['AvgExp'] = ti.ema(data.Close, 10)
        df['%K'] = ti.STOK(data.Close, data.Low, data.High, 10)
        df['ROC']=ti.ROC(data.Close,10)
        df['RSI']=ti.RSI(data.Close)
        df['ACCUMULATION_DISTRIBUTION']=ti.AccDO(data.Close,data.High,data.Low)
        df['MACD']=ti.MACD(data.Close)
        df['WILLIAMS']=ti.WilliamsR(data.Close,data.High,data.Low)
        df['HIGH_PRICE_ACCELERATION']=ti.HPA(data.High,14)
        df['DISPARITY_5']=ti.Disparity(data.Close,5)
        df['DISPARITY_10']=ti.Disparity(data.Close,10)
        df.fillna(0,inplace=True)
        self.Data=df