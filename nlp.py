
import nltk
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sa
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import spacy

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
#from .news import get_news_summary
nlp = spacy.load("en_core_web_sm")

#nltk.download('vader_lexicon')

def get_sentiments(new_text):
    analyzer = sa().polarity_scores(new_text)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']
    return neg ,neu,pos,comp

def get_sentiments_stock(sumarized_text:str):
    #stock_news_sumarized=get_news_summary(comp_name=stock)

    sent_list_count=[]
    for news in sumarized_text:
        sents_dict={}
        sents=get_sentiments(news)
        sents_dict["negative"]=1 if (sents[0]>sents[2]) else 0
        sents_dict["positive"]=1 if (sents[0]<sents[2]) else 0
        sents_dict["neutral"]=1 if (sents[0]==sents[1]) else 0
        sent_list_count.append(sents_dict)

    return pd.DataFrame(sent_list_count)


def compute_cosine_similarity(text1, text2,stop_words=en_stop):
    
    # stores text in a list
    list_text = [text1, text2]

    # converts text into vectors with the TF-IDF 
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    vectorizer.fit_transform(list_text)
    tfidf_text1, tfidf_text2 = vectorizer.transform([list_text[0]]), vectorizer.transform([list_text[1]])
    
    # computes the cosine similarity
    cs_score = cosine_similarity(tfidf_text1, tfidf_text2)
    score=np.round(cs_score[0][0],2)
    if score > 0.5:
      print(text1,text2)
      print(score)
    return score
    
def get_named_entities(sent:str):
    doc = nlp(sent)
    return [(X.text, X.label_) for X in doc.ents]
def get_lemmatize(sent:str):
    sent_nlp=nlp(str(sent))
    #drop stopword and punctuations
    lemmatizables=[y for y in sent_nlp if not y.is_stop and y.pos_ != 'PUNCT']
    return [(x.orth_,x.pos_, x.lemma_) for x in lemmatizables]

if __name__=="__main__":
    sent = "main financials sectors in France are going well"
    print(get_named_entities(sent))
    print(get_sentiments(sent))
    print(compute_cosine_similarity("ukraine","Ukraine"))
    print(get_lemmatize("code of Ukraine"))