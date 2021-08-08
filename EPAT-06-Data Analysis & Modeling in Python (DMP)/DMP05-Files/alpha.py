import numpy as np

def ma_crossover(data, short_lookback, long_lookback):
    data['ma_long'] = data['Adj Close'].ewm(span=long_lookback).mean()
    data['ma_short'] = data['Adj Close'].ewm(span=short_lookback).mean()  

    data['ma_signal'] = np.where(data.ma_short > data.ma_long, 1, -1)    
    data.iloc[:long_lookback,-1] = np.nan
    return data

