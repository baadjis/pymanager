
import matplotlib.pyplot as plt
import pandas as pd
from .stock import Stock
from wordcloud import WordCloud, STOPWORDS
from .ta import moving_average
   
def plot_stock(ticker,period="4y"):
    st=Stock(ticker)
    st.get_data(period)
    pdf=pd.DataFrame()
    pdf["close"]=st.data["Close"]
    pdf["ma_50"]=moving_average(st.data,num=50)
    pdf["ma_200"]=moving_average(st.data,num=200)
    pdf.plot()
    plt.show()


def word_cloud(text):
    
    stopwords = set(STOPWORDS)
    allWords = ' '.join([nws for nws in text])
    
    wordCloud = WordCloud(background_color='black',width = 1600, height = 800,stopwords = stopwords,min_font_size = 20,max_font_size=150,colormap='prism').generate(allWords)
    fig, ax = plt.subplots(figsize=(20,10), facecolor='k')
    plt.imshow(wordCloud)
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()
if __name__=="__main__":
    print("viz")
