import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from io import BytesIO
import xlsxwriter
import base64
import datetime

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

from plotly import graph_objs as go

import requests
import json

# Adding Logo and Page Name
st.set_page_config(page_title='The Stocks Prediction App', page_icon='📈')

###########
# Sidebar #
###########
# create an empty container in the sidebar
window_selection_c = st.sidebar.container()
window_selection_c.markdown("## Initial Selections")  # add a title to the sidebar container
sub_columns = window_selection_c.columns(2)  # Split the container into two columns for start and end date

option = st.sidebar.selectbox('Select Stock Name', ('FB', 'AMZN', 'AAPL', 'MSFT', 'GOOG'))

# todays represents present day
today = datetime.date.today()

# if nothing selected then this will show the past 3000 days previous date
before = today - datetime.timedelta(days=3000)

start_date = sub_columns[0].date_input('Start Date', before)

# if nothing selected then this will select present data
end_date = sub_columns[1].date_input('End Date', today)

if start_date < end_date:
    start_date_mod = start_date.strftime('%d %B %Y')
    end_date_mod = end_date.strftime('%d %B %Y')
    st.sidebar.success('Start Date: `%s`\n\nEnd Date: `%s`' % (start_date_mod, end_date_mod))

else:
    st.sidebar.error('Error: End Date must fall after Start Date.')

##############
# Stock data #
##############

# Download data
df = yf.download(option, start=start_date, end=end_date)


# Code for downloading excel sheet
def to_excel(df):
    output = BytesIO()
    # BytesIO manipulates string and bytes data in memory.
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">🔽 Download Excel File</a>'  # decode b'abc' => abc
    # Base64 is a binary to ASCII encoding scheme.


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    # #Plots the stock_open feature
    # fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name="stock_open"))
    # # Plots the stock_close feature
    # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="stock_close"))

    st.subheader('Plotting The Raw Data 👨🏻‍🔧')
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.layout.update(xaxis_title="Date", yaxis_title="Price in $", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


# Bollinger Bands
indicator_bb = BollingerBands(df['Close'])
bb = df
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_mavg'] = indicator_bb.bollinger_mavg()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close', 'bb_h', 'bb_mavg', 'bb_l']]

# Moving Average Convergence Divergence
macd = df
macd['MACD'] = MACD(df['Close']).macd()
macd['Signal'] = MACD(df['Close']).macd_signal()
macd = macd[['MACD', 'Signal']]

# macd_hist = MACD(df['Close']).macd_diff()


# Resistence Strength Indicator
rsi = RSIIndicator(df['Close']).rsi()

###################
# Main app #
###################

# Setting the title of the App
st.title("The Stock Prediction App 📈")
st.write(
    "This app let's you to generate both short-term and long-term **live time-series** forecast of FAAMG stocks and **predict** the future price of them (upto 7 years).")
st.markdown("![Gif](https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif)")

# What are stocks
st.subheader("Some Key Terms 📌")

with st.expander("Defining Stock Market"):
    st.markdown(
        "The stock market is a market that enables the seamless exchange of buying and selling of company stocks. Every Stock Exchange has its own Stock Index value. The index is the average value that is calculated by combining several stocks. This helps in representing the entire stock market and predicting the market’s movement over time. The stock market can have a huge impact on people and the country’s economy as a whole. Therefore, predicting the stock trends in an efficient manner can minimize the risk of loss and maximize profit.")

with st.expander("Time-Series Forecasting in Stock Market"):
    st.markdown(
        "Stock and financial markets tend to be unpredictable and even illogical. Due to these characteristics, financial data should be necessarily possessing a rather turbulent structure which often makes it hard to find reliable patterns. Modeling turbulent structures requires machine learning algorithms capable of finding hidden structures within the data and predict how they will affect them in the future. The most efficient methodology to achieve this is Machine Learning.")
    st.markdown(
        "**Time-Series Forecasting with the help of machine learning has the potential to ease the whole process by analyzing large chunks of data, spotting significant patterns and generating a single output that navigates traders towards a particular decision based on predicted asset prices.**")

with st.expander("Understanding FAAMG Stocks"):
    st.markdown(
        "In finance, “FAAMG” is an acronym that refers to the stocks of five prominent American technology companies: Facebook (FB), Amazon (AMZN), Apple (AAPL), Microsoft (MSFT); and Alphabet's Google (GOOG)")
    st.markdown(
        "FAAMG are termed growth stocks, mostly due to their year-over-year (YOY) steady and consistent increase in the earnings they generate, which translates into increasing stock prices. Retail and institutional investors buy into these stocks directly or indirectly through mutual funds, hedge funds, or exchange traded funds (ETFs) in a bid to make a profit when the share prices of the tech firms go up.")

# Shows the text in the app
data_load_state = st.text('Loading data...')

# Data of recent days
st.subheader('Raw data 🛠')
st.dataframe(df[["Open", "High", "Low", "Adj Close", "Volume"]])

# When loading done then shows this text
data_load_state.text('Loading data... Done ✅!')

# Download excel
DownDf = df[["Open", "Close", "High", "Low", "Volume"]]
st.markdown(get_table_download_link(DownDf), unsafe_allow_html=True)

# Recent open and close price
st.markdown("**Current Open Price in USD:**")
st.write(round(df.iloc[-1, 1]))
st.markdown("**Current Close Price in USD:**")
st.write(round(df.iloc[-1, 4]))

# will be using this at the end for comparing
current = round(df.iloc[-1, 4])

# Plotting close price
plot_raw_data()

st.subheader("Stock Indicators 📍")

# Plot the prices and the bolinger bands
st.markdown("**Bollinger Bands**")
st.line_chart(bb)

# with st.expander("What are Bollinger Bands ?"):
#     st.markdown("Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price. Bollinger Bands use 2 parameters, Period and Standard Deviations, StdDev. The default values are 20 for period, and 2 for standard deviations, although you may customize the combinations. Bollinger bands help determine whether prices are high or low on a relative basis. They are used in pairs, both upper and lower bands and in conjunction with a moving average.")

with st.expander("⚙️ - How to use Bollinger Bands"):
    st.write(
        """    
        - Bollinger Bands are composed of three lines. One of the more common calculations uses a 20-day simple moving average (SMA) for the middle band. The upper band is calculated by taking the middle band and adding twice the daily standard deviation to that amount. The lower band is calculated by taking the middle band minus two times the daily standard deviation.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/Boolinger.png?raw=true")

    st.write(
        """    
        - To calculate a simple moving average, the number of prices within a time period is divided by the number of total periods.
        """
    )
    st.image(
        "https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/calculate%20moving%20avg.JPG?raw=true")
    st.image(
        "https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/calculate%20moving%20avg%202.JPG?raw=true")

    st.write(
        """    
        - If the stock prices crosses the lower band, starting from below the lower band then it indicates that the stock is pushing higher and traders can exploit the opportunity to make a buy decision.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/boolinger%20entry.JPG?raw=true")

    st.write(
        """    
        - If the stock prices crosses the upper band, starting from above the upper band then it indicates that the stock prices are falling and traders should make a sell decision.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/boolinger%20exit.JPG?raw=true")

# Plot RSI
st.markdown("**Relative Strength Index(RSI)**")
st.line_chart(rsi)

# with st.expander("What is RSI ?"):
#     st.markdown(' The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset. The RSI is displayed as an oscillator (a line graph that moves between two extremes) and can have a reading from 0 to 100. Traditional interpretation and usage of the RSI are that values of 70 or above indicate that a security is becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. An RSI reading of 30 or below indicates an oversold or undervalued condition. ')

with st.expander("⚙️ - How to use RSI"):
    st.write(
        """    
        - The RSI is calculated using average price gains and losses over a 14 days time period, with values bounded from 0 to 100.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/RSI%20AVG%20Gains.JPG?raw=true")
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/rsi%20avg%20loss.JPG?raw=true")

    st.write(
        """    
        - The RSI is said to be overbought when over the 70 zone and oversold when under the 30 level. Now one idea to generate a buy signal is to wait for the RSI to dip below 30 and then buy when the RSI breaks above the 30 line. This means that price was significantly oversold and is likely to bounce as the RSI sees strength returning to the market in the form of a move above the 30 area.
        """
    )
    st.image(
        "https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/rsi%20buy%20and%20sell.JPG?raw=true")

# Plot MACD
st.markdown("**Moving Average Convergence Divergence (MACD)**")
st.line_chart(macd)
# with st.expander("What is MACD ?"):
#     st.markdown('Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line," is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals. Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.')

with st.expander("⚙️ - How to use MACD"):
    st.write(
        """    
        - The MACD is calculated by subtracting the 26-period moving average from the 12-period moving average.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/macd%201.JPG?raw=true")

    st.write(
        """    
        - The result of that calculation is the MACD line. A nine-day moving average of the MACD called the "signal line" is then plotted on top of the MACD line, which can function as a trigger for buy and sell signals.
        """
    )
    st.image(
        "https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/macd%20and%20signal%20v2.JPG?raw=true")

    st.write(
        """    
        - Traders may buy the security when the MACD crosses above its signal line and sell—or short—the security when the MACD crosses below the signal line.
        """
    )
    st.image("https://github.com/Kens3i/Stocks-Daily-2.0/blob/main/Web%20App%20Images/macd%203.JPG?raw=true")

###################
# Prediciton Part #
###################
st.markdown("""---""")
df.reset_index(inplace=True)

st.subheader("Future Prediction ⏳")

# A slider in streamlit which is used to set how many years you want to predict
st.markdown("**Select How Many Years You Want To Predict:**")
n_years = st.slider("", 1, 7)

# 1 year=365 days so if n_years then n_years*365 days.
period = n_years * 365

# This is a button to run the prediction
start_execution = st.button('Click Here To Predict 🚀')
st.markdown(
    "Note : Could take some time(30 sec - 1 min) to predict as the model is training and fitting the data in real time. Thanks for having patience.")

if start_execution:
    # Displays GIF when loading
    gif_runner = st.image(
        'https://aws1.discourse-cdn.com/business7/uploads/streamlit/original/2X/2/247a8220ebe0d7e99dbbd31a2c227dde7767fbe1.gif')

    # Predict forecast with Prophet.
    # The features are "Data" and "Close" price.
    df_train = df[['Date', 'Close']]

    # Renaming the features as per Prophet needs.
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # changepoint_prior_scale eplains how much flexible the trend should be, should it overfit opr underfit
    m = Prophet(weekly_seasonality=False, yearly_seasonality=False, daily_seasonality=False, growth="linear",
                changepoint_prior_scale=0.02)
    # fitting the data
    m.fit(df_train)
    # Make dataframe with future dates for forecasting
    # periods:Int number of periods to forecast forward.
    future = m.make_future_dataframe(periods=period)

    # setting a minimum price of 50 dollars
    future['floor'] = 50.00

    # forecast is the predicted dataset
    forecast = m.predict(future)

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = forecast[col].clip(lower=50.0)

    # Adding subheader
    st.subheader('Forecast Data 🔮')
    # Shows the last 5 values
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    with st.expander("Explaining The Features"):
        st.markdown("`ds` -> Dates")
        st.markdown("`yhat` -> Forecast Data")
        st.markdown("`yhat_lower` and `yhat_upper` -> Lowermost and uppermost uncertainty intervals")

    # Plotting the forecast data
    if n_years == 1:
        st.subheader(f'Result of the Forecast upto {n_years} year ⚙️')
    else:
        st.subheader(f'Result of the Forecast upto {n_years} years ⚙️')

    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1, use_container_width=True)
    fig1.layout.update(xaxis_title="Date", yaxis_title="Price in $", xaxis_rangeslider_visible=True)
    st.markdown("The date column is now labeled as **ds** and the values columns as **y**")

    # Showing the latest predicted price
    st.markdown("**Lastest Predicted Price in USD:**")
    latest = round(forecast.iloc[-1, -1])
    st.write(latest)

    # Suggesting the user if they should buy or not
    st.markdown("**Should You Buy The Stock?**")
    if latest > current:
        st.markdown("**YES**")
    else:
        st.markdown("**NO**")

    # Removes The GIF after loading
    gif_runner.empty()

# This is the part of converting the currency
st.markdown("""---""")
st.subheader("Convert Your Currency ⚖️")
st.image("https://cdn.dribbble.com/users/22930/screenshots/1923847/money.gif", width=340)
st.markdown("**Input Option**")


def currency_converter():
    url = "https://free.currconv.com/api/v7/currencies?apiKey=73373685ba13d7df49a0"
    # Make a request to a web page, and return the status code
    response = requests.get(url)

    # response.content returns the content of the response, in bytesl
    status_get = response.content

    # decoding the bytes from utf-8 to string
    status_get = status_get.decode('utf-8')

    # parsing and stored the json string to python dictionary
    status_json = json.loads(status_get)

    # making an array for storing currency
    all_currencies = []
    for i in status_json["results"]:
        all_currencies.append(i)
    number = st.number_input('Input the Value')
    # rounding the input to 2 digit place
    number = round(number, 2)
    # selecting the currency to convert
    option = st.selectbox('', all_currencies, key=1)
    # filling values
    value = st.empty()
    # Selecting the currency to convert to
    option2 = st.selectbox('', all_currencies, key=2)

    left_column, right_column = st.columns(2)
    pressed = left_column.button('Convert 💱')
    if pressed:
        # This is used for conversion
        sta = str(option) + "_" + str(option2)

        url = "https://free.currconv.com/api/v7/convert?q=" + sta + "&compact=ultra&apiKey=73373685ba13d7df49a0"
        response = requests.get(url)
        status_get = response.content
        status_get = status_get.decode('utf-8')
        status_json = json.loads(status_get)

        # if the value is valid then -> number * 1 unit of input currency = output currency.
        # here we display 1 unit of input currency = how much unit of output currency
        converter = "1 " + option + " = " + str(status_json[sta]) + " " + option2
        st.write(converter)
        st.markdown("The Converted Value Is:")
        st.write(status_json[sta] * number)


currency_converter()

with st.expander("Credits💎"):
    st.subheader("Developer:")
    st.image("https://avatars.githubusercontent.com/u/52373756?s=400&u=f3b4f3403656c3f61c6b378f1028803bd9e81031&v=4")
    st.markdown(""" App Made By **[Koustav Banerjee](https://www.linkedin.com/in/koding-senpai/)**""")
    st.markdown("""**[Source code](https://github.com/Kens3i/Stocks-Daily)**""")