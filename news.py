#from GoogleNews import GoogleNews
from gnews import GNews
from newspaper import Article,Config
import datetime as dt
import pandas as pd
import nltk


#nltk.download('punkt')
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 20

now = dt.date.today()
now = now.strftime('%m-%d-%Y')
yesterday = dt.date.today() - dt.timedelta(days = 2)
yesterday = yesterday.strftime('%m-%d-%Y')

def search_company_news(comp_name):
    #print(comp_name)
    #googlenews = GoogleNews(start=yesterday,end=now)
    '''googlenews=GoogleNews(start='05/01/2020',end='05/31/2020')
    #googlenews=GoogleNews(lang='en', period='1d')
    googlenews.search('Coronavirus')
    result = googlenews.results(sort=True)

    googlenews.clear()
    '''
    google_news = GNews(period='7d')
    result=google_news.get_news("cryptos")
    print("news",result)
    #store the results
    df = pd.DataFrame(result)
    
    
    return df

def get_news_articles(comp_name):
    df=search_company_news(comp_name)
    if not(df.empty):
        try:
            list =[] #creating an empty list 
            for i in df.index:
                dict = {} #creating an empty dictionary to append an article in every single iteration
                article:Article = Article(df['url'][i],config=config) #providing the link
                try:
                    article.download() #downloading the article 
                    article.parse() #parsing the article
                    article.nlp() #performing natural language processing (nlp)
                except Exception as e:
                    print("exception occurred:" + str(e))
                #storing results in our empty dictionary
                dict['Date']=df['date'][i] 
                dict['Media']=df['media'][i]
                dict['Title']=article.title
                dict['Article']=article.text
                dict['Summary']=article.summary
                dict['Key_words']=article.keywords
                list.append(dict)

            check_empty = not any(list)
            # print(check_empty)
            if check_empty == False:
                news_df=pd.DataFrame(list) #creating dataframe
                return news_df

        except Exception as e:
            #exception handling
            print("exception occurred:" + str(e))
      
def get_news_summary(comp_name):
    news_df=get_news_articles(comp_name)
    return news_df["Summary"].values
    
