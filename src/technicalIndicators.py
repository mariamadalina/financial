
import pandas as pd
import numpy as np

def STOK(close, low, high, n):  
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def ema(close, span):
    e = close.ewm(span=span, adjust=False).mean()
    return e    