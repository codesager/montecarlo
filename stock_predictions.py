import pandas as pd
import numpy as np
from openbb import obb
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm

st.set_page_config(page_title="Stock Predictions", page_icon= ":chart_with_upwards_trend", layout="wide")
def get_data(ticker, from_date, to_date):
    stock_data = obb.equity.price.historical(symbol=ticker, provider='yfinance', adjusted=True, start_date=from_date, end_date=to_date)
    stocks_data = stock_data.to_df()
    return stocks_data

st.sidebar.header("Ticker selection")
ticker = st.sidebar.text_input(label = "Ticker", placeholder="Enter Ticker", max_chars=6)

col1, col2 = st.sidebar.columns((2))
with col1:
    from_date = st.sidebar.date_input('From')

with col2:
    to_date = st.sidebar.date_input('To')
    
button = st.sidebar.button('Get Data')

if button:
    colors = ['#A56CC1', '#A6ACEC']
    df = get_data(ticker, from_date, to_date)
    st.title(f'Historic stock data for {ticker}')
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
        
    # st.dataframe(df)
    fig1 = px.histogram(df, x='adj_close', title='Price Distribution')
    fig2 = px.line(df, x=df.index, y='adj_close',title='Closing price')
    
    log_returns = np.log(df['adj_close'] / df['adj_close'].shift(1))
    fig3 = ff.create_distplot([log_returns.iloc[1:]], group_labels=['Log Returns'], bin_size=0.1, show_rug=False, show_hist=False)
    
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
    fig4 = ff.create_distplot([predicted_prices.iloc[-1]], group_labels=['Price Prediction'], show_rug=False, show_hist=False)
    
    col21, col22 = st.columns((2))    
    with col21:
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        
    with col22:
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)



