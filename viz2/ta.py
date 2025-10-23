from scipy.stats import linregress

def moving_average(data,num:int):
        
    return data['Close'].rolling(num).mean()
        
def rsi(data,num=14):

    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    RSI = 100-(100 / (1 +rs))
    # Skip first 14 days to have real values
    RSI = RSI.iloc[14:]
    return RSI

        
def lrc(data):
        
        data0 = data.copy()
        data0['date_id'] = ((data0.index.date - data0.index.date.min())).astype('timedelta64[D]')
        data0['date_id'] = data0['date_id'].dt.days + 1

        # high trend line

        data1 = data0.copy()

        while len(data1)>3:

            reg = linregress(
                            x=data1['date_id'],
                            y=data1['High'],
                            )
            data1 = data1.loc[data1['High'] > reg[0] * data1['date_id'] + reg[1]]

        reg = linregress(
                            x=data1['date_id'],
                            y=data1['High'],
                            )

        data0['high_trend'] = reg[0] * data0['date_id'] + reg[1]

        # low trend line

        data1 = data0.copy()

        while len(data1)>3:

            reg = linregress(
                            x=data1['date_id'],
                            y=data1['Low'],
                            )
            data1 = data1.loc[data1['Low'] < reg[0] * data1['date_id'] + reg[1]]

        reg = linregress(
                            x=data1['date_id'],
                            y=data1['Low'],
                            )

        data0['low_trend'] = reg[0] * data0['date_id'] + reg[1]

        return data0

def candelistick(ax,data):

    up = data[data.Close >= data.Open]
  
    # "down" dataframe will store the stock_prices
    # when the closing stock price is
    # lesser than the opening stock prices
    down = data[data.Close < data.Open]
    
    # When the stock prices have decreased, then it
    # will be represented by blue color candlestick
    col1 = 'green'
    
    # When the stock prices have increased, then it 
    # will be represented by green color candlestick
    col2 = 'red'
    
    # Setting width of candlestick elements
    width = .8
    width2 = .08
    
    # Plotting up prices of the stock
    ax.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
    ax.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
    ax.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
    
    # Plotting down prices of the stock
    ax.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
    ax.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
    ax.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)
    
    # rotating the x-axis tick labels at 30degree 
    # towards right
    #ax.xticks(rotation=30, ha='right')
