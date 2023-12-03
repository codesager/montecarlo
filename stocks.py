import pandas as pd
import numpy as np
from openbb import obb
import streamlit as st
import plotly.figure_factory as ff
import plotly as pt

def buildData(ticker):
    stock_data = obb.equity.price.historical(symbol=ticker, provider='yfinance')
    stock_data.to_df()
    print(stock_data.to_df())
    
ticker = 'GOOGL'
stockdata = buildData(ticker)

returns = stockdata['close'].pct_change()
print(returns)



# import plotly.figure_factory as ff
# import numpy as np

# # Add histogram data
# x1 = np.random.randn(200) - 2
# x2 = np.random.randn(200)
# x3 = np.random.randn(200) + 2
# x4 = np.random.randn(200) + 4

# # Group data together
# hist_data = [x1, x2, x3, x4]

# group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

# # Create distplot with custom bin_size
# fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
# fig.show()