import pandas as pd
import numpy as np
from openbb import obb
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm

# # Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#     )
    
def get_data(ticker):
    stock_data = obb.equity.price.historical(symbol=ticker, provider='yfinance', adjusted=True)
    stocks_data = stock_data.to_df()
    return stocks_data
    
ticker = st.text_input('Enter Ticker', max_chars=6)
if st.button('Get Data'):
    colors = ['#A56CC1', '#A6ACEC']
    df = get_data(ticker)
    
    # st.dataframe(df)
    fig1 = px.histogram(df, x='adj_close', title='Distribution')
    fig2 = px.line(df, x=df.index, y='adj_close',title='Closing price')
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    
    log_returns = np.log(df['adj_close'] / df['adj_close'].shift(1))
    fig3 = ff.create_distplot([log_returns.iloc[1:]], group_labels=['Log Returns'], bin_size=0.1, show_rug=False, show_hist=False)
    st.plotly_chart(fig3)
    
    mean = log_returns.mean()
    var = log_returns.var()
    drift = mean - (0.5 * var)
    
    stdev = log_returns.std()
    days = 50
    trials = 10000
    Z = norm.ppf(np.random.rand(days, trials)) #days, trials
    daily_returns = np.exp(drift + stdev * Z)
    
    price_lists = np.zeros_like(daily_returns)
    price_lists[0] = df.iloc[-1]['adj_close']
    for t in range(1, days):
        price_lists[t] = price_lists[t-1]*daily_returns[t]
    
    predicted_prices = pd.DataFrame(price_lists)
    # st.dataframe(predicted_prices)
    fig4 = ff.create_distplot([predicted_prices.iloc[-1]], group_labels=['Price Distribution'], show_rug=False, show_hist=False)
    st.plotly_chart(fig4)
    


