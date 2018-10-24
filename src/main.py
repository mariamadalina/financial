import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.decomposition import FastICA, PCA


import Indexes as idx
import technicalIndicators as ti

tickers = ['AAPL', 'MSFT', '^GSPC']

start_date = '2010-01-01'
end_date = '2016-12-31'

# panel_data = web.DataReader(tickers, 'yahoo', start_date, end_date)
# panel_data.head(9)

# close = panel_data['Close']
# all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
# close = close.reindex(all_weekdays)
# close = close.fillna(method='ffill')
# close.head(10)
# msft = close.loc[:, 'MSFT']
# short_rolling_msft = msft.rolling(window=20).mean()
# long_rolling_msft = msft.rolling(window=100).mean()
# figclose, ax = plt.subplots(figsize=(16,9))

# ema_short = close.ewm(span=20, adjust=False).mean()
# ax.plot(ema_short.loc[start_date:end_date, :].index, ema_short.loc[start_date:end_date, 'MSFT'], label = 'Span 20-days EMA')

df = pd.DataFrame(idx.get_stock('FB', '1/1/2016', '12/31/2016'))
df['High'] = idx.get_high('FB', '1/1/2016', '12/31/2016')
df['Low'] = idx.get_low('FB', '1/1/2016', '12/31/2016')
df['%K'] = ti.STOK(df['Close'], df['Low'], df['High'], 14)
df['%D'] = ti.ema(df['Close'], 14)
df.tail()
