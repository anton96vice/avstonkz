import yfinance as yf
import streamlit as st
import datetime 
import talib 
import ta
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen, Request
from yahoo_fin import stock_info as si
from pandas_datareader import DataReader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
yf.pdr_override()

st.write("""
# Technical Analysis Web Application
Shown below are the **Moving Average Crossovers**, **Bollinger Bands**, **MACD's**, **Commodity Channel Indexes**, and **Relative Strength Indexes** of any stock!
""")

st.sidebar.header('User Input Parameters')

today = datetime.date.today()
def user_input_features():
    ticker = st.sidebar.text_input("Ticker", 'ASRT')
    start_date = st.sidebar.text_input("Start Date", '2020-02-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker, start_date, end_date

symbol, start, end = user_input_features()

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']
company_name = get_symbol(symbol.upper())

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# Read data 
data = yf.download(symbol,start,end)

# Adjusted Close Price
st.header(f"Adjusted Close Price\n {company_name}")
st.line_chart(data['Adj Close'])

# ## SMA and EMA
#Simple Moving Average
data['SMA'] = talib.SMA(data['Adj Close'], timeperiod = 20)

# Exponential Moving Average
data['EMA'] = talib.EMA(data['Adj Close'], timeperiod = 20)

# Plot
st.header(f"Simple Moving Average vs. Exponential Moving Average\n {company_name}")
st.line_chart(data[['Adj Close','SMA','EMA']])

# Bollinger Bands
data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)

# Plot
st.header(f"Bollinger Bands\n {company_name}")
st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

# ## MACD (Moving Average Convergence Divergence)
# MACD
data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Plot
st.header(f"Moving Average Convergence Divergence\n {company_name}")
st.line_chart(data[['macd','macdsignal']])

## CCI (Commodity Channel Index)
# CCI
cci = ta.trend.cci(data['High'], data['Low'], data['Close'], window=31, constant=0.015)

# Plot
st.header(f"Commodity Channel Index\n {company_name}")
st.line_chart(cci)

# ## RSI (Relative Strength Index)
# RSI
data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)

# Plot
st.header(f"Relative Strength Index\n {company_name}")
st.line_chart(data['RSI'])

# ## OBV (On Balance Volume)
# OBV
data['OBV'] = talib.OBV(data['Adj Close'], data['Volume'])/10**6

# Plot
st.header(f"On Balance Volume\n {company_name}")
st.line_chart(data['OBV'])


def getData(list_of_stocks):
    for stock in list_of_stocks:
        df = DataReader(stock, 'yahoo', start, end)
        print (stock)
        
        # Current Price 
        cprice = si.get_live_price('{}'.format(stock))
        cprice = round(cprice, 2)
        
        # Sharpe Ratio
        x = 5000
        y = (x)
            
        stock_df = df
        stock_df['Norm return'] = stock_df['Adj Close'] / stock_df.iloc[0]['Adj Close']
         
        allocation = float(x/y)
        stock_df['Allocation'] = stock_df['Norm return'] * allocation
            
        stock_df['Position'] = stock_df['Allocation'] * x
        pos = [df['Position']]
        val = pd.concat(pos, axis=1)
        val.columns = ['WMT Pos']
        val['Total Pos'] = val.sum(axis=1)
            
        val.tail(1)            
        val['Daily Return'] = val['Total Pos'].pct_change(1)            
        Sharpe_Ratio = val['Daily Return'].mean() / val['Daily Return'].std()            
        A_Sharpe_Ratio = (252**0.5) * Sharpe_Ratio        
        A_Sharpe_Ratio = round(A_Sharpe_Ratio, 2)
        
        # News Sentiment 
        finwiz_url = 'https://finviz.com/quote.ashx?t='
        news_tables = {}
        
        url = finwiz_url + stock
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        html = BeautifulSoup(response, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[stock] = news_table
        
        parsed_news = []
        
        # Iterate through the news
        for file_name, news_table in news_tables.items():
            for x in news_table.findAll('tr'):
                text = x.a.get_text() 
                date_scrape = x.td.text.split()
        
                if len(date_scrape) == 1:
                    time = date_scrape[0]
                    
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]
        
                ticker = file_name.split('_')[0]
                
                parsed_news.append([ticker, date, time, text])
                
        vader = SentimentIntensityAnalyzer()
        
        columns = ['ticker', 'date', 'time', 'headline']
        dataframe = pd.DataFrame(parsed_news, columns=columns)
        scores = dataframe['headline'].apply(vader.polarity_scores).tolist()
        
        scores_df = pd.DataFrame(scores)
        dataframe = dataframe.join(scores_df, rsuffix='_right')
        
        dataframe['date'] = pd.to_datetime(dataframe.date).dt.date
        dataframe = dataframe.set_index('ticker')
        
        sentiment = round(dataframe['compound'].mean(), 2)
        
        # Beta
        df = DataReader(stock,'yahoo',start, end)
        dfb = DataReader('^GSPC','yahoo',start, end)
        
        rts = df.resample('M').last()
        rbts = dfb.resample('M').last()
        dfsm = pd.DataFrame({'s_adjclose' : rts['Adj Close'],
                                'b_adjclose' : rbts['Adj Close']},
                                index=rts.index)
        
        
        dfsm[['s_returns','b_returns']] = dfsm[['s_adjclose','b_adjclose']]/\
            dfsm[['s_adjclose','b_adjclose']].shift(1) -1
        dfsm = dfsm.dropna()
        covmat = np.cov(dfsm["s_returns"],dfsm["b_returns"])
                
        beta = covmat[0,1]/covmat[1,1] 
        beta = round(beta, 2)
        
        # Relative Strength Index
        df["rsi"] = talib.RSI(df["Close"])
        values = df["rsi"].tail(14)
        value = values.mean()
        rsi = round(value, 2)

        output = ("\nTicker: " + str(stock) + "\nCurrent Price : " + str(cprice) + "\nSharpe Ratio: " + str(A_Sharpe_Ratio) + "\nNews Sentiment: " + str(sentiment) + "\nRelative Strength Index: " + str(rsi) + "\nBeta Value for 1 Year: " + str(beta))
        print(output)

        return output, stock, cprice, A_Sharpe_Ratio, rsi, beta, sentiment

stock = []
stock.append(symbol)
output, stock, cprice, A_Sharpe_Ratio, rsi, beta, sentiment = getData(stock)

dictt = [{'Ticker': str(stock), 'Current_Price': str(cprice), 'Sharpe Ratio':str(A_Sharpe_Ratio),
          'News Sentiment': str(sentiment), 'Relative Strength Index': str(rsi), 'Beta Value for 1 Year': str(beta) }]


output2 = pd.DataFrame(dictt)
st.write("""
# TA + Vader sentimental analysis for: """,stock, """\n(neg = bad, 0 - neutral, pos = good)
""")
st.dataframe(data=output2.style.highlight_max(axis=0))
