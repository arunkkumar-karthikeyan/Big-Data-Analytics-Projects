# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 01:19:26 2022

@author: akarthikeyan1
"""

#==============================================================================

# Initiating required libraries

#==============================================================================

import pandas as pd                                    # for creating dataframe
import numpy as np # for replacing null values, finding percentile, std deviation
import yfinance as yf                             # getting ticker informations
from datetime import datetime, timedelta                      # for time period
import streamlit as st                               # data web app development
import matplotlib.pyplot as plt             # for visualizing graphs and charts
import plotly.graph_objects as go           # for visualizing graphs and charts
from plotly.subplots import make_subplots                 # for making subplots
from numerize import numerize              # for numerize the numeric variables
import plotly.express as px
from PIL import Image                                    # for inserting Images
import yahoo_fin.stock_info as si 
              # for getting Key Executive Informations, Ticker Analysis details

#==============================================================================

# **Tab1 - Summary:**

#==============================================================================

def Tab1():
    
    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab1 - Summary 📈')
    
    
    # Get Ticker information from yahoo finance :

    def GetStockPrice(ticker, start_date, end_date):
        global stock_price
        # Loop through the selected tickers
        stock_price = pd.DataFrame()
        for tick in ticker:
            stock_df = yf.Ticker(tick).history(start=start_date, end=end_date)
            stock_price = pd.concat([stock_price, stock_df], axis=0)                      # Combine results
        return stock_price
    
    @st.cache
    def GetSummaryData(ticker):
        table = yf.Ticker(ticker).info
        return table
        
    summary = GetSummaryData(ticker)
    
    current_price = summary['currentPrice']                                                                  # Getting the current price of the stock
    change = str(round(((summary['currentPrice'] / summary['previousClose']) -1), 2)) + "% Today"            # Percentage Change of Current Price with the Previous Close
    
    Col1,Col2,Col3 = st.columns([1,1,1])
    Col1.metric(label="Current Stock Price", value =current_price, delta=change, help="The percentage value denotes the percentage change of current stock price of the selected ticker compared to the previous Close of the selected ticker")
    Col2.metric(label='Volume', value=summary['volume'])
    Col3.metric(label='Day High',value=summary['dayHigh'])
    
    column1, column2 = st.columns(2)
    with column1:
        keys=['previousClose', 'open', 'bid', 'ask', 'dayHigh', 'dayLow','volume','averageVolume','fiftyTwoWeekHigh','fiftyTwoWeekLow']
        company_stats = {}
        for key in keys:
            company_stats.update({key:summary[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)}).rename(index={'previousClose':'Previous Close', 'open':'Open', 'bid':'Bid', 'ask':'Ask', 'dayHigh':'Day High', 'dayLow':'Day Low', 'volume':'Volume', 'averageVolume':'Avg. Volume', 'fiftyTwoWeekHigh':'52 Week High', 'fiftyTwoWeekLow':'52 Week Low'})                # Convert to DataFrame
        company_stats.iloc[:,0] = company_stats.iloc[:,0].replace(np.nan,0)                                   # Replacing all null values as zero
        company_stats.iloc[:,0] = [numerize.numerize(y) for y in company_stats.iloc[:,0]]                     # Numerize the values
        column1.dataframe(company_stats, width=300)
    
    summary = GetSummaryData(ticker)
    with column2:
        keys=['marketCap','beta','trailingPE','trailingEps','totalCashPerShare','revenuePerShare','dividendYield','returnOnAssets','debtToEquity','returnOnEquity']
        company_stats = {}
        for key in keys:
            company_stats.update({key:summary[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)}).rename(index={'marketCap':'Market Cap', 'beta':'Beta (5Y Monthly)', 'trailingPE':'PE Ratio (TTM)', 'trailingEps':'EPS (TTM)', 'totalCashPerShare':'Total Cash per Share', 'revenuePerShare':'Revenue per Share', 'dividendYield':'Dividend Yield', 'debtToEquity':'Debt to Equity', 'returnOnEquity':'Return on Equity'})                  # Convert to DataFrame
        company_stats.iloc[:,0] = company_stats.iloc[:,0].replace(np.nan,0)                                    # Replacing all null values as zero
        company_stats.iloc[:,0] = [numerize.numerize(y) for y in company_stats.iloc[:,0]]                      # Numerize the values 
        column2.dataframe(company_stats, width=300)    
      
    # Plotting chart using Plotly library and subplots. Adding Scatter chart for Ticker Close price and Bar chart for Ticker Volume   
    # https://stackoverflow.com/questions/67291178/how-to-create-subplots-using-plotly-express

    
    chart_data=yf.Ticker(ticker).history(period='MAX', interval='1d')
    # Dropping na values from the downloaded ticker data 
    chart_data_01 = chart_data.dropna()
    
    
    # Creating Range-slider charts using Plotly            
    # https://plotly.com/python/range-slider/#basic-range-slider-and-range-selectors
    # https://plotly.com/python/axes/
    
    
    fig = make_subplots(specs=[[{"secondary_y": True}]]) 
    fig.add_trace(go.Scatter(x=chart_data_01.index,y=chart_data_01['Close'],name='Close Price', mode='lines', line=dict(color="crimson")),secondary_y=False)
    fig.add_trace(go.Bar(x=chart_data_01.index,y=chart_data_01['Volume'],name='Volume'),secondary_y=True)
    fig.update_xaxes(
        rangeselector=dict(
            buttons= list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(label = "MAX", step="all")
                ])), rangeselector_bgcolor="darkblue", rangeslider = dict(visible = True), title_text = 'Date')
    
    fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])                                     # Hiding Weekends where stock market is Closed
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")      # To show unified spikes as per yahoo finance website
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",spikedash="dot")       # To show unified spikes as per yahoo finance website
    fig.update_yaxes(title_text = 'Close Price (USD)')
    fig.update_yaxes(visible=False, secondary_y=True)                                               # Aligning the axis to one y axis 
    fig.update_traces(hovertemplate="<br>".join(["Date: %{x}","Close: %{y}"]))                      # Customizing hover text (wrapped)
    fig.update_layout(height=500, width=900, showlegend = False,                                    # Customizing the plot layout
                         title = {
                             'text': 'Historical Close price',
                              'y': 1.0,
                              'x': 0.5, 
                              'xanchor': 'center',
                              'yanchor': 'top'})
    
    st.plotly_chart(fig)
#==============================================================================

# **Tab2 - Chart:**

#==============================================================================

def Tab2():
    
    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 2 - Chart 📊')
    
    Indicator = st.selectbox("Select indicator:", ['Moving Average','Bollinger Bands'])
    
    @st.cache
    def BollingerBands(ticker):                                                                     # Code for finding the Bollinger Upper and Lower Bands
        global stockdata1
        stockdata = yf.Ticker(ticker).history(start=start_date1, end=end_date1)
        typical_price = (stockdata['High'] + stockdata['Low'] + stockdata['Close']) / 3
        B_MA = pd.Series((typical_price.rolling(50).mean()), name='B_MA')
        sigma = typical_price.rolling(50).std()
        BU = pd.Series((B_MA + 2 * sigma), name='BU')
        BL = pd.Series((B_MA - 2 * sigma), name='BL')
        stockdata = stockdata.join(B_MA)
        stockdata = stockdata.join(BU)
        stockdata = stockdata.join(BL)
        stockdata1 = stockdata.dropna()
        return stockdata
    
    if ticker != '':
        # creating start date and end date buttons to select date range to plot charts
        Col1, Col2, Col3, Col4 = st.columns(4)
        global start_date1, end_date1
        start_date1 = Col1.date_input("Start date", datetime.today().date() - timedelta(days=365), key='start_date')
        end_date1 = Col2.date_input("End date", datetime.today().date(), key='end_date')
        
        # creating drop down to select interval to show data
        interval_pd = Col3.selectbox("Select an interval:", ['Day','Week','Month'])
        if interval_pd == 'Month':
            interval = '1mo'
        elif interval_pd == 'Week':
            interval = '1wk'
        else:
            interval = '1d'
            
        st.write("Select date range or duration:")
        
        # creating columns to create buttons to select the duration of data
        Col5,Col6,Col7,Col8,Col9,Col10,Col11,Col12= st.columns(8)
        with Col5: Dur_1M = st.button('1M')
        with Col6: Dur_3M = st.button('3M')
        with Col7: Dur_6M = st.button('6M')
        with Col8: Dur_YTD = st.button('YTD')
        with Col9: Dur_1Y = st.button('1Y')
        with Col10: Dur_3Y = st.button('3Y')
        with Col11: Dur_5Y = st.button('5Y')
        with Col12: Dur_MAX = st.button('MAX')
        
        # Getting ticker data for the specified time period
        if Dur_1M:
            chart_data=yf.Ticker(ticker).history(period = "1mo", interval=interval)
        elif Dur_3M:
            chart_data=yf.Ticker(ticker).history(period = "3mo", interval=interval)
        elif Dur_6M:
            chart_data=yf.Ticker(ticker).history(period = "6mo", interval=interval)
        elif Dur_YTD:
            chart_data=yf.Ticker(ticker).history(period = "ytd", interval=interval)
        elif Dur_1Y:
            chart_data=yf.Ticker(ticker).history(period = "1y", interval=interval)
        elif Dur_3Y:
            chart_data=yf.Ticker(ticker).history(period = "3y", interval=interval)
        elif Dur_5Y:
            chart_data=yf.Ticker(ticker).history(period = "5y", interval=interval)
        elif Dur_MAX:
            chart_data=yf.Ticker(ticker).history(period= "MAX", interval=interval)
        else:
            chart_data = yf.Ticker(ticker).history(start=start_date1, end=end_date1)
        # Dropping na values from the downloaded ticker data 
        chart_data_01 = chart_data.dropna()
    
        # https://plotly.com/python/multiple-axes/   
        # https://plotly.com/python/candlestick-charts/  
        # https://plotly.com/python/line-and-scatter/
        # https://community.plotly.com/t/any-way-to-have-both-x-unified-and-y-unified-for-hovermode-like-crosshair/63340/2
        # https://stackoverflow.com/questions/59057881/python-plotly-how-to-customize-hover-template-on-with-what-information-to-show
        # https://www.learnpythonwithrune.org/simple-and-exponential-moving-average-with-python-and-pandas/
        # https://tcoil.info/compute-bollinger-bands-for-stocks-with-python-and-pandas/
        # https://albertum.medium.com/plotting-bollinger-bands-with-plotly-graph-objects-1c7172899542
        
        
        
        graph_type = Col4.selectbox("Choose Chart Type", ['Line','Candle','Area'])
        if graph_type == 'Line' and Indicator == 'Moving Average':                                        # Added plots for Simple Moving Average and Exponential Moving Average
        
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chart_data_01.index, y=chart_data_01['Close'], mode='lines', name = 'Close', line=dict(color="violet"), text=['Date','Volume']), secondary_y = False)
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'red', name = '50 day-SMA'))
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].ewm(span=50, adjust=False).mean(), marker_color = 'yellow', name = '50 day-EMA'))
            fig.add_trace(go.Bar(x = chart_data_01.index, y = chart_data_01['Volume'], marker_color='green', name = 'Volume'), secondary_y = True)
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_traces(hovertemplate="<br>".join(["Date: %{x}","Close: %{y}"]))                    # To customize hover text (wrapped)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)
            
        if graph_type == 'Line' and Indicator == 'Bollinger Bands':                                       # Added plots for Bollinger Bands
            
            bollinger_band = BollingerBands(ticker)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chart_data_01.index, y=chart_data_01['Close'], mode='lines', name = 'Close', line=dict(color="blue"), text=['Date','Volume']), secondary_y = False)
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BU'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.5))         # Plot for Bollinger Upper Band
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BL'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Lower Band', opacity = 0.5))         # Plot for Bollinger Lower Band
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'red', name = '50 day-SMA'))
            fig.add_trace(go.Bar(x = chart_data_01.index, y = chart_data_01['Volume'], marker_color='violet', name = 'Volume'), secondary_y = True)
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_traces(hovertemplate="<br>".join(["Date: %{x}","Close: %{y}"]))                    # To customize hover text (wrapped)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)
    
        if graph_type == 'Candle' and Indicator == 'Moving Average': 
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Candlestick(x=chart_data_01.index, open=chart_data_01['Open'], high=chart_data_01['High'], low=chart_data_01['Low'], close=chart_data_01['Close'], name = 'Stock'))
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'blue', name = '50 day-SMA'))
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].ewm(span=50, adjust=False).mean(), marker_color = 'yellow', name = '50 day-EMA'))
            fig.add_trace(go.Bar(x = chart_data_01.index, y = chart_data_01['Volume'], marker_color='green', name = 'Volume'), secondary_y = True)
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)
            
        if graph_type == 'Candle' and Indicator == 'Bollinger Bands':                                     # Added plots for Bollinger Bands
            
            bollinger_band = BollingerBands(ticker)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Candlestick(x=chart_data_01.index, open=chart_data_01['Open'], high=chart_data_01['High'], low=chart_data_01['Low'], close=chart_data_01['Close'], name = 'Stock'))
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BU'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.5))
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BL'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Lower Band', opacity = 0.5))
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'red', name = '50 day-SMA'))
            fig.add_trace(go.Bar(x = chart_data_01.index, y = chart_data_01['Volume'], marker_color='green', name = 'Volume'), secondary_y = True)
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)

        if graph_type == 'Area' and Indicator == 'Moving Average':
        
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig = px.area(chart_data_01, chart_data_01.index, chart_data_01['Close'])
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'red', name = '50 day-SMA'))
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(visible=False, secondary_y=True)
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_traces(hovertemplate="<br>".join(["Date: %{x}","Close: %{y}"]))                    # To customize hover text (wrapped)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)  
            
        if graph_type == 'Area' and Indicator == 'Bollinger Bands':
            
            bollinger_band = BollingerBands(ticker)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig = px.area(chart_data_01, chart_data_01.index, chart_data_01['Close'])
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BU'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Upper Band', opacity = 0.5))
            fig.add_trace(go.Scatter(x = bollinger_band.index, y=bollinger_band['BL'], marker_color = 'yellow', line = {'dash': 'dash'}, name = 'Lower Band', opacity = 0.5))
            fig.add_trace(go.Scatter(x = chart_data_01.index, y=chart_data_01['Close'].rolling(window=50).mean(), marker_color = 'red', name = '50 day-SMA'))
            fig.update_xaxes(rangebreaks= [dict(bounds=['sat','mon'])])
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dot")    # To show unified spikes as per yahoo finance website
            fig.update_yaxes(visible=False, secondary_y=True)
            fig.update_yaxes(range=[0,chart_data_01['Volume'].max()*3],secondary_y=True,visible=False)
            fig.update_traces(hovertemplate="<br>".join(["Date: %{x}","Close: %{y}"]))                    # To customize hover text (wrapped)
            fig.update_layout(height=500, width=900,xaxis_showgrid=False, yaxis_showgrid=True)
            st.plotly_chart(fig)     
            
#==============================================================================

# **Tab3 - Financials:**

#==============================================================================
      

def Tab3():
    
# Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 3 - Financials 💸')
    
# Adding Yearly Income table to show stock data
    @st.cache
    def GetyearlyIncome(ticker):
        yearlyincome = yf.Ticker(ticker).financials
        return yearlyincome
    
# Adding Quarterly Income table to show stock data
    @st.cache
    def GetQuarterlyIncome(ticker):
        quarterlyincome = yf.Ticker(ticker).quarterly_financials
        return quarterlyincome
    
# Adding Yearly Balance table to show stock data
    @st.cache
    def GetYearlyBalance(ticker):
        yearlybalance = yf.Ticker(ticker).balance_sheet
        return yearlybalance
    
# Adding Quarterly Balance table to show stock data
    @st.cache
    def GetQuarterlyBalance(ticker):
        quarterlybalance = yf.Ticker(ticker).quarterly_balance_sheet
        return quarterlybalance
    
# Adding Yearly Cash Flow table to show stock data
    @st.cache
    def GetYearlyCash(ticker):
        yearlycashflow = yf.Ticker(ticker).cashflow
        return yearlycashflow
    
# Add Quarterly Cash Flow table to show stock data
    @st.cache
    def GetQuarterlyCash(ticker):
        quarterlycashflow = yf.Ticker(ticker).quarterly_cashflow
        return quarterlycashflow
    
    
# Adding a select box for Type of Financial and duration
    Column1, Column2 = st.columns([1,1])    
    Financial = Column1.selectbox("Type", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
    Duration = Column2.selectbox("Select tab", ['Annual', 'Quarterly'])
    
    # Show the selected tab
    if Financial == 'Income Statement' and Duration == 'Annual':
        # Run Option 1 & Option 1
        df = GetyearlyIncome(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Annual Income Statement for " + str(ticker))
        st.dataframe(df,width=1000,height=500)

        
    elif Financial == 'Income Statement' and Duration == 'Quarterly':
        # Run Option 1 & Option 2
        df = GetQuarterlyIncome(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Quarterly Income Statement for " + str(ticker))
        st.dataframe(df,width=1000,height=800)
        
    elif Financial == 'Balance Sheet' and Duration == 'Annual':
        # Run Option 2 & Option 1
        df = GetYearlyBalance(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Annual Balance Sheet for " + str(ticker))
        st.dataframe(df,width=1000,height=800)
        
    elif Financial == 'Balance Sheet' and Duration == 'Quarterly':
        # Run Option 2 & Option 2
        df = GetQuarterlyBalance(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Quarterly Balance Sheet for " + str(ticker))
        st.dataframe(df,width=1000,height=800)
    
    elif Financial == 'Cash Flow' and Duration == 'Annual':
        # Run Option 3 & Option 1
        df = GetYearlyCash(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Annual Cash Flow for " + str(ticker))
        st.dataframe(df,width=1000,height=800)
    
    elif Financial == 'Cash Flow' and Duration == 'Quarterly':
        # Run Option 3 & Option 2
        df = GetQuarterlyCash(ticker)
        df = df.style.format(na_rep=0, thousands=",",formatter=('{:0,.0f}'))
        st.caption("All numbers in thousands")
        st.subheader(" Quarterly Income Statement for " + str(ticker))
        st.dataframe(df,width=1000,height=800)

 #=============================================================================
 
# **Tab4 - Statistics:**
 
 #=============================================================================               

def Tab4():

    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 4 - Statistics 🎯') 
    Col1, Col2 = st.columns(2) 
             
    with Col1:
        @st.cache
        def GetValuation(ticker):
            table = yf.Ticker(ticker).info
            #table['exDividendDate'] = datetime.fromtimestamp(table['exDividendDate']).strftime('%Y%m%d')
            return table
        valuation = GetValuation(ticker)
        
        st.subheader("Valuation Measures")
        
        keys=['marketCap','enterpriseValue','trailingPE', 'forwardPE', 'pegRatio','priceToSalesTrailing12Months', 'priceToBook', 'enterpriseToRevenue','enterpriseToEbitda']
        company_stats = {}
        for key in keys:
            company_stats.update({key:valuation[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)}).rename(index={'marketCap':'Market Cap (intraday)','enterpriseValue':'Enterprise Value','trailingPE':'Trailing P/E','forwardPE':'Forward P/E','pegRatio':'PEG Ratio (5 yr expected)','priceToSalesTrailing12Months':'Price/Sales (ttm)','priceToBook':'Price/Book (mrq)','enterpriseToRevenue':'Enterprise Value/Revenue','enterpriseToEbitda':'Enterprise Value/EBITDA'})              # Convert to DataFrame  
        company_stats.iloc[:,0] = company_stats.iloc[:,0].replace(np.nan,0)                # Replacing all null values as zero
        company_stats.iloc[:,0] = [numerize.numerize(y) for y in company_stats.iloc[:,0]]  # Numerize the values
        Col1.table(company_stats)
        
        st.subheader("Financial Highlights")
            
        st.markdown("Fiscal Year")
        keys=['nextFiscalYearEnd','lastFiscalYearEnd','mostRecentQuarter']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'nextFiscalYearEnd':'Next Fiscal Year End','lastFiscalYearEnd':'Last Fiscal Year End','mostRecentQuarter':'Most Recent Quarter (mrq)'})          # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
            
        st.markdown("Profitability")
        keys=['profitMargins','operatingMargins']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'profitMargins':'Profit Margin','operatingMargins':'Operating Margin (ttm)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
            
        st.markdown("Management Effectiveness")
        keys=['returnOnAssets','returnOnEquity']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'returnOnAssets':'Return on Assets (ttm)','returnOnEquity':'Return on Equity (ttm)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
            
        st.markdown("Income Statement")
        keys=['totalRevenue','revenuePerShare','revenueGrowth','grossProfits','ebitda','netIncomeToCommon','trailingEps','earningsGrowth']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'totalRevenue':'Revenue (ttm)','revenuePerShare':'Revenue Per Share (ttm)','revenueGrowth':'Quarterly Revenue Growth (yoy)','grossProfits':'Gross Profit (ttm)','ebitda':'EBITDA','netIncomeToCommon':'Net Income Avi to Common (ttm)','trailingEps':'Diluted EPS (ttm)','earningsGrowth':'Quarterly Earnings Growth (yoy)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
            
        st.markdown("Balance Sheet")
        keys=['totalCash','totalCashPerShare','totalDebt','debtToEquity','currentRatio','bookValue']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'totalCash':'Total Cash (mrq)','totalCashPerShare':'Total Cash Per Share (mrq)','totalDebt':'Total Debt (mrq)','debtToEquity':'Total Debt/Equity (mrq)','currentRatio':'Current Ratio (mrq)','bookValue':'Book Value Per Share (mrq)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
            
        st.markdown("Cash Flow Statement")
        keys=['operatingCashflow','freeCashflow']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'operatingCashflow':'Operating Cash Flow (ttm)','freeCashflow':'Levered Free Cash Flow (ttm)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col1.table(stats)
       
    with Col2:
        st.subheader("Trading Information")
            
        st.markdown("Stock Price History")
        keys=['beta','52WeekChange','SandP52WeekChange','fiftyTwoWeekHigh','fiftyTwoWeekLow','fiftyDayAverage','twoHundredDayAverage']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'beta':'Beta (5Y Monthly)','52WeekChange':'52-Week Change','SandP52WeekChange':'S&P500 52-Week Change','fiftyTwoWeekHigh':'52 Week High','fiftyTwoWeekLow':'52 Week Low','fiftyDayAverage':'50-Day Moving Average','twoHundredDayAverage':'200-Day Moving Average'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col2.table(stats)
            
        st.markdown("Share Statistics")
        keys=['averageVolume','averageVolume10days','sharesOutstanding','impliedSharesOutstanding','floatShares','heldPercentInsiders','heldPercentInstitutions','sharesShort','shortRatio','shortPercentOfFloat','sharesShortPriorMonth']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'averageVolume':'Avg Vol (3 month)','averageVolume10days':'Avg Vol (10 day)','sharesOutstanding':'Shares Outstanding','impliedSharesOutstanding':'Implied Shares Outstanding','floatShares':'Float','heldPercentInsiders':'% Held by Insiders','heldPercentInstitutions':'% Held by Institutions','sharesShort':'Shares Short','shortRatio':'Short Ratio','shortPercentOfFloat':'Short % of Float','sharesShortPriorMonth':'Shares Short (prior Month)'})                                    # Convert to DataFrame
        stats.iloc[:,0] = stats.iloc[:,0].replace(np.nan,0)                                 # Replacing all null values as zero
        stats.iloc[:,0] = [numerize.numerize(y) for y in stats.iloc[:,0]]                   # Numerize the values
        Col2.table(stats)
       
        st.markdown("Dividends & Splits")
        keys=['dividendRate','dividendYield','trailingAnnualDividendYield','fiveYearAvgDividendYield','payoutRatio','lastDividendDate','exDividendDate','lastSplitDate']
        stats = {}
        for key in keys:
            stats.update({key:valuation[key]})
        stats = pd.DataFrame({'Value':pd.Series(stats)}).rename(index={'dividendRate':'Forward Annual Dividend Rate','dividendYield':'Forward Annual Dividend Yield','trailingAnnualDividendYield':'Trailing Annual Dividend Rate','fiveYearAvgDividendYield':'5 Year Average Dividend Yield','payoutRatio':'Payout Ratio','lastDividendDate':'Dividend Date','exDividendDate':'Ex-Dividend Date','lastSplitDate':'Last Split Date'})                                    # Convert to DataFrame                                                 
        stats = stats.astype(str)
        Col2.table(stats)
        
#=============================================================================
 
# **Tab5 - Company Profile and Description:**
 
 #=============================================================================               

def Tab5():

    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 5 - Company Profile and Description 📢')
    
    Col1, Col2 = st.columns([4,4])
    with Col1:
        st.subheader(yf.Ticker(ticker).info['longName'])
    with Col2:
        st.image(yf.Ticker(ticker).info['logo_url'])
    
    Col3, Col4 = st.columns(2)   
    with Col3:
        @st.cache
        def GetCompanyInfo(ticker):
                return yf.Ticker(ticker).info
   
        info = GetCompanyInfo(ticker)
        st.write(info['address1'])
        st.write(info['city'],info['state'],info['zip'])
        st.write(info['country'])
        st.write(info['phone'])
        st.write(info['website'])
        
    with Col4:
        @st.cache
        def GetCompanyInfo(ticker):
                return yf.Ticker(ticker).info
   
        info = GetCompanyInfo(ticker)
        st.write("Sector(s):",info['sector'])
        st.write("Industry:",info['industry'])
        st.write("Full Time Employees:",info['fullTimeEmployees'])
        
    st.subheader('Key Executives')
    @st.cache(allow_output_mutation=True)
    def GetExecutives(ticker):
        company_off = si.get_company_officers(ticker)                                                                   # Getting Key Executive Informations (Used si - yahoo_fin.stock_info to get entire details of Analysis as per Yahoo finance website)
        company_off[['totalPay','yearBorn','age']] = company_off[['totalPay','yearBorn','age']].replace(np.nan,0)       # Replacing na values with zero
        company_off.loc[:,'totalPay'] = [numerize.numerize(y) for y in company_off.loc[:,'totalPay']]                   # Numerize Total pay variable
        return company_off
    
    executives = GetExecutives(ticker)
    executives = executives.rename(columns={'title':'Title','totalPay':'Pay','exercisedValue':'Exercised','yearBorn':'Year Born','age':'Age'})     # Renaming Column variable names
    executives['Year Born'] = executives['Year Born'].map('{:,.0f}'.format)                                                                        # Formatting the Year Born variable to zero decimal places
    executives['Age'] = executives['Age'].astype(int)                                                                                              # Converting Age variable to int data type
    st.dataframe(executives.loc[:,['Title','Pay','Exercised','Year Born','Age']],width=800)
    
    
    st.subheader('Description')
    st.write(info['longBusinessSummary'])
    
    st.subheader('Holders')
    @st.cache
    def GetShareHolders(ticker):
        return yf.Ticker(ticker).major_holders
    
    # Get the information of major share holders
    share_holders = GetShareHolders(ticker)
    st.dataframe(share_holders)
    
    st.subheader('Top Institutional Holders')
    @st.cache(allow_output_mutation=True)
    def GetInstituteHolders(ticker):
        return yf.Ticker(ticker).institutional_holders
    
    # Get the information of major share holders
    ins_holders = GetInstituteHolders(ticker)
    ins_holders['Date Reported'] = pd.to_datetime(ins_holders['Date Reported']).dt.date
    ins_holders = ins_holders.style.format(na_rep=0, thousands=",")
    st.dataframe(ins_holders, width=1000)
                      
#==============================================================================

# **Tab6 - Monte Carlo Simulation:**

#==============================================================================

def Tab6():
    
    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 6 - Monte Carlo Simulation 🚀')
    
    #Dropdown for selecting simulation and horizon
    col1,col2 = st.columns(2)
    simulations = col1.selectbox("Select number of simulations (n)", [200, 500, 1000])
    time_horizon = col2.selectbox("Select a time horizon (t)", [30, 60, 90])
    
    @st.cache
    def montecarlo(ticker, time_horizon, simulations):
        
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=30)
        stockprice = yf.Ticker(ticker).history(start=start_date, end=end_date)
        close_price = stockprice['Close']
        
        daily_return = stockprice['Close'].pct_change()
        daily_volatility = np.std(daily_return)

        # Run the simulation
        simulation_df = pd.DataFrame()

        for i in range(simulations):
    
            # The list to store the next stock price
            next_price = []
    
            # Create the next stock price
            last_price = stockprice['Close'].iloc[-1]
    
            for j in range(time_horizon):
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price
    
            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
        
        return simulation_df
    
    mc = montecarlo(ticker, time_horizon, simulations)
        
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30)
    stockprice = yf.Ticker(ticker).history(start=start_date, end=end_date)
    close_price = stockprice['Close']
    
    # Plot the simulation stock price in the future
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)
        
    ax.plot(mc)
    plt.title('Monte Carlo simulation for ' + str(ticker) + ' stock price in next ' + str(time_horizon) + ' days')
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    plt.axhline(y= close_price[-1], color ='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
        
    st.pyplot(fig)

    # Get the ending price of the 200th day
    ending_price = mc.iloc[-1:, :].values[0, ]
    
    fig1, ax = plt.subplots()
    fig1.set_size_inches(15, 10, forward=True)
        
    ax.hist(ending_price, bins=50)
    plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
    plt.legend(['Current Stock Price is: ' + str(np.round(np.percentile(ending_price, 5), 2))+ ' USD '])
    plt.title('Distribution of the Ending Price')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    st.pyplot(fig1)
        
    # Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)
        
    # Value at Risk
    VaR = close_price[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

#=============================================================================
 
# **Tab7 - Analysis:**
 
 #=============================================================================               

def Tab7():

    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 7 - Analysis 👨🏻‍💻') 
    
    @st.cache
    def GetAnalysisdetails(ticker):                                                   # Used si - yahoo_fin.stock_info to get entire details of Analysis as per Yahoo finance website
        return si.get_analysts_info(ticker)
    
    # Getting Analysis details
    info = GetAnalysisdetails(ticker)                                                 # Getting Ticker Analysis details
    for key,values in info.items():                                                   # Looping through each key and value field from Analysis data
        analysis = pd.DataFrame(info[key])                                            # Storing each key and values in dataframe
        st.dataframe(analysis, width=1000)
        
#=============================================================================
 
# **Tab8 - Sustainability:**
 
 #=============================================================================               

def Tab8():

    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 8 - Sustainability 🌱') 
    
    Col1, Col2, Col3, Col4, Col5 = st.columns(5)
    
    @st.cache
    def GetSustainability(ticker):
        return yf.Ticker(ticker).sustainability
    
    sus = GetSustainability(ticker)
    Col1.metric(label="Total ESG Risk score", value = sus.loc['totalEsg'])
    Col2.metric(label="Percentile",value = sus.loc['percentile'])
    Col3.metric(label="Environment Risk Score", value = sus.loc['environmentScore'])
    Col4.metric(label="Social Risk Score", value = sus.loc['socialScore'])
    Col5.metric(label="Governance Risk Score", value = sus.loc['governanceScore'])

    st.metric(label="Controversy Level", value = sus.loc['highestControversy'], help ="Sustainalytics’ Controversies Research identifies companies involved in incidents and events that may negatively impact stakeholders, the environment or the company’s operations. Controversies are rated on a scale from one to five with five denoting the most serious controversies with the largest potential impact." )

#=============================================================================
 
# **Tab9 - Reccomendations:**
 
 #=============================================================================


def Tab9():

    # Add dashboard title and description
    st.write("Data Source - Yahoo Finance (https://finance.yahoo.com/)")
    st.subheader('Tab 9 - Recommendations 📔') 
    
    @st.cache
    def GetAnalystRecommendation(ticker):
        return yf.Ticker(ticker).recommendations
    
    recco = GetAnalystRecommendation(ticker)
    def green_font (series):
        color = 'background-color: green'
        default = ''
        return [color if (series["To Grade"] == 'Buy') else default] 
    
    recco.style.apply(green_font,axis=1)
    st.dataframe(recco, width=1000,height=800)
      
#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar :
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    #Inserting Image
    st.sidebar.image("https://www.businessinsider.in/thumb/msid-94084438,width-700,resizemode-4,imgsize-1797166/stock-market/news/bajaj-finance-is-the-favourite-stock-of-mutual-funds-itc-is-the-most-sold/stock-11.jpg", width=300, use_column_width=False, caption='Stock market')
    
    # Add selection box :
    global ticker
    st.sidebar.subheader("""Stock Prediction Web Application :dart:""")
    st.sidebar.caption("""Build by Arunkkumar Karthikeyan""")
    ticker = st.sidebar.selectbox("Select a ticker from the below S&P 500 list 👇", ticker_list)
    
    # Add start date and end date :
    # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/                  # Adding emojis in streamlit
    global start_date, end_date
    column1, column2 = st.sidebar.columns([1,1])
    start_date = column1.date_input("Start date 📅 :", datetime.today().date() - timedelta(days=30))
    end_date = column2.date_input("End date 📅 :", datetime.today().date())
      

    tab01, tab02, tab03, tab04, tab05, tab06, tab07, tab08, tab09 = st.tabs(['Summary', 'Chart', 'Financials', 'Statistics', 'Profile', 'Monte Carlo Simulation', 'Analysis', 'Sustainability', 'Recommendations'])
    
    # defining an update button:
    update_button = st.sidebar.button('Update Data 👈')
    if update_button:
        st.experimental_rerun()
    
    # Show the selected tab
    with tab01:
        # Run tab 1
        Tab1()
    with tab02:
        # Run tab 2
        Tab2()
    with tab03:
        # Run tab 3
        Tab3()
    with tab04:
        # Run tab 4
        Tab4()
    with tab05:
         # Run tab 5
         Tab5() 
    with tab06:
        # Run tab 6
         Tab6()
    with tab07:
         # Run tab 7
          Tab7()
    with tab08:
          # Run tab 8
           Tab8()
    with tab09:
          # Run tab 9
           Tab9()
             
if __name__ == "__main__":
    
    # Customizing the Page icon and Title
    # https://emojipedia.org/chart-increasing/
    st.set_page_config(layout="wide", page_title='Stock Analysis Dashboard', page_icon='📈', initial_sidebar_state = 'auto')
    st.markdown(""" <p style="text-align: center;"><span style="font-family: Helvetica; color: rgb(0, 131, 184); font-size: 30px;">
                S&P 500 Stock Analysis Dashboard</span></p>""", unsafe_allow_html=True)                                              # Adding Header Style
    EXPANDER_TEXT = """
    ```python
    [theme]
    primaryColor = "#E694FF"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#DCDCDC"
    font = "sans-serif"
    ```
    """
    tabs_font_css = """ <style> button[data-baseweb="tab"] { font-size: 16px;} </style> """
    st.write(tabs_font_css, unsafe_allow_html=True)
    run()
    
###############################################################################
# END
###############################################################################
