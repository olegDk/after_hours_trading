import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
import mplfinance as mpl

cwd = os.getcwd()
print(cwd)

if not sys.gettrace():
    ticker_minute_data_path = f'{cwd}/analytics/modeling/training/ticker_minute_data'
else:
    ticker_minute_data_path = f'{cwd}/../ticker_minute_data'

ticker = 'C'
file_path = f'{ticker_minute_data_path}/ticker_minute_{ticker}.csv'

ticker_minute_data = pd.read_csv(file_path, index_col=0)
ticker_minute_data.index = pd.to_datetime(ticker_minute_data.index)
mpl.plot(ticker_minute_data.loc['2021-12-01 09:30':'2021-12-02 13:13'], type='candle')
