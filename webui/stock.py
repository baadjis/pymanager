from curses.ascii import isdigit
from functools import lru_cache
from time import time
from unicodedata import numeric
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import matplotlib.dates as mpdates
from mpl_finance import candlestick2_ohlc,candlestick_ochl
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import dask.dataframe as dd

from stocks import Stock
from dataprovider import news 
from viz import ta 
from dataprovider import yahoo
from index import Index
from ml import nlp 
st.set_page_config(page_title="stock view",layout="wide")
st.title("stock view")

sectors=[
    "Energy","Industrials","Utilities","Financials Services",
    "Consumer Defensives","Consumer Staples","Consumer Cyclables","Consumer Staples",
    "Basic Materials","Consumer Discretionary","Communication Services","Real Estate",
    "Information Technology"
]
funcs=["infos","stats","dividends",
"analysis","financials","quote","news","chart"]
graphs=["plot","candlestick","returns","compare"]
periods=["1d","5d","1mo","3mo","6mo","1y","2y","5y"]


def get_indicator(stock:Stock,indicator):
    data=stock.data
    if indicator=="ma_50":
        return ta.moving_average(data,50)
    if indicator=="ma_200":
        return ta.moving_average(data,200)
    if indicator=="rsi":
    
        return ta.rsi(data)
    if indicator=="volume":
        return data["Volume"]
        
    if indicator=="lrc":
        return ta.lrc(data)

def execute_func(func:str,ticker,stock:Stock):

    st.title(func)
    if func=="stats":

        stats=stock.get_stats()
        st.dataframe(stats.astype(str))

    if func=="dividends":

        div=stock.get_dividends()
        st.dataframe(div.astype(str))

    if func=="infos":
        infos=stock.infos
        longbus=infos.Value.pop('longBusinessSummary')
        st.write(longbus)
        infosval=infos.astype(str)
        st.dataframe(infosval)

    if func=="news":
        news=news.search_company_news(ticker)
        st.dataframe(news)

    if func=="chart":
        graph_select=st.selectbox(label="graph",options=graphs)
        period = option_menu(None,periods,icons=None,default_index=0, orientation="horizontal")
        stock.get_data(period)
        data=stock.data
        close=data["Close"]
        open=data["Open"]
        if graph_select=="plot":
            indicators_options=["ma_50","ma_200","rsi","volume","lrc"]
            plots=["line","candlestick","area"]
            cols = st.columns(2)
            
            indicators=cols[0].multiselect("indicators",indicators_options)
            plot_type=cols[1].selectbox("line type",plots)
            plt.rc('figure', figsize=(20, 15))
            #volume=data["Volume"]
            
            fig = plt.figure()
            #pdf=pd.DataFrame()
            
            
            nb_plots=1
            gs = fig.add_gridspec(nb_plots, 1)
    
            axes = []
            plot_price = fig.add_subplot(gs[0])
            if plot_type=="line":
                plot_price.plot(close,linewidth=2, label='close')
                
                
            if plot_type=="candlestick":
                
                plt.style.use('ggplot')
                ta.candelistick(ax=plot_price,data=data)
                #date_format = mpdates.DateFormatter('%d-%m-%Y')
                #plot_price.xaxis.set_major_formatter(date_format)
                #plot_price.set_xticks(np.arange(len(data)))
                plot_price.set_xticklabels(data.index, fontsize=6, rotation=45)
            if plot_type=="area":
                plot_price.fill_between(close.index,close.values)
            plot_price.set_facecolor("black")
            plot_price.grid(True)
            axes.append(plot_price)
            #fig.tight_layout(pad=4.0)
            n=len(indicators)
            if n>0:
                
                for ind in indicators:
                    ind_data=get_indicator(stock,ind)
                    if ind in ["ma_50","ma_200"]:
        
                       axes[0].plot(ind_data,linewidth=1, label=ind)
                    elif ind=="lrc":
                         axes[0].plot(ind_data["high_trend"],linewidth=2, label="high_trend")
                         axes[0].plot(ind_data["low_trend"],linewidth=2, label="low_trend")
                    else:
                        gs = fig.add_gridspec(nb_plots + 1, 1)
                        for j in range(nb_plots):
                           axes[j].set_position(gs[j].get_position(fig))
                           axes[j].set_subplotspec(gs[j])
    
                            # And add the new one...
                        ax = fig.add_subplot(gs[-1])
                        if ind=="volume":
                            diff=close-open
                            color=(diff > 0).apply(lambda x: 'g' if x else 'r')
                            ax.bar(ind_data.index,ind_data.values,color=color.values)
                    
                        else:
                           ax.plot(ind_data, linewidth=2, label=ind)
                        
                        ax.set_ylabel(ind)
                        ax.set_facecolor("black")
                        ax.grid(True)
                        axes.append(ax)
                        nb_plots+=1

            plot_price.legend(loc='upper left', 
        bbox_to_anchor= (-0.005, 0.95), fontsize=16)
            
            st.pyplot(fig)

        
        if graph_select=="returns":
    
            returns=yahoo.get_log_returns(data)

            color = (returns > 0).apply(lambda x: 'g' if x else 'r')
            fig, ax = plt.subplots()
            returns.plot.bar(ax=ax,color=color)
            st.pyplot(fig)
        if graph_select=="compare":
            to_comp=st.text_input("enter ticker")
            if to_comp:
                stock2=Stock(to_comp)
                stock2.get_data(period)
                #returns=pd.DataFrame()
                returns=yahoo.get_log_returns(stock.data)
                returns2=yahoo.get_log_returns(stock2.data)
                fig, ax = plt.subplots()
                
                ax.plot(returns,linewidth=2, label=ticker)
                ax.plot(returns2,linewidth=2, label=to_comp)
                ax.legend(loc='upper left', 
        bbox_to_anchor= (-0.005, 0.95), fontsize=16)
                st.pyplot(fig)


        
    if func=="financials":
        financials=stock.get_financials()
        
        for dt  in financials:
           st.write(dt)
           st.table(financials[dt])
    if func=="analysis":

        an=stock.get_analysises()
        for dt  in an:
           st.write(dt)
           st.table(an[dt])
    if func=="quote":
        quote=stock.get_quote_table()
        data={"keys":list(quote.keys()),"values":list(quote.values())}
        dt=pd.DataFrame(data)
        
        st.table(dt.astype(str))
        
def explore_stock():
   
    ticker=st.text_input("ticker")
    rt=st.empty()
   
    if ticker:
        infos,stats,dividends,analysis,financials,quote,news,chart= st.tabs(funcs)
        
        stock=Stock(ticker)
        stock.get_data("2d")
        close=stock.data["Close"].values
        value=round(close[-1],2)
        pch=round((close[-1]-close[-2])/close[-2],2)
        delta=round(close[-1]-close[-2],2)
        rt.metric(label=ticker,value=f"{value}({pch}%)",delta=delta)

        with infos:
            execute_func("infos",ticker,stock)
        with stats:
            execute_func("stats",ticker,stock)
        with financials:
            execute_func("financials",ticker,stock)
        with dividends:
            execute_func("dividends",ticker,stock)
        with quote:
            execute_func("quote",ticker,stock)
        with chart:
            execute_func("chart",ticker,stock)
        with news:
            execute_func("news",ticker,stock)
        with analysis:
            execute_func("analysis",ticker,stock)

def dask_read_json(file):
    return dd.read_json(file, blocksize=None, orient="records", 
    lines=False).compute()
def get_countries_indexes(data,countries):

    return data[data.country.isin(countries)].to_dict('records')
def get_regions_indexes(data,regions):
    return data[data.region.isin(regions)].name

def screen_stocks():

    json_data=dask_read_json("major_indices.json")

    wiki_indexes=dask_read_json("wiki_indexes.json")
    regions=wiki_indexes["region"].unique()
    countries=wiki_indexes["country"].unique()
    indexes=json_data["Symbol"].unique()
    sectors_list=["consumer","telecom"]
    industries_list=["consumer","telecom"]
    global_indexes=wiki_indexes[wiki_indexes.theme=="global"].name
    st.session_state.filters_dicts=[]
    if "index_str" not in st.session_state:

        st.session_state.index_str=""

    if "tab_data" not in st.session_state:

        st.session_state.tab_data=None

    if "my_regions" not in st.session_state:
        st.session_state.my_regions=[]
    if "my_countries" not in st.session_state:
        st.session_state.my_countries=[]
    st.session_state.filter_container=st.empty()
    st.session_state.theme=st.radio("theme",options=("major","global","regional","national"))
    if st.session_state.theme =="national":
             st.session_state.my_countries=st.multiselect("countries",countries)
    if st.session_state.theme =="regional":
        st.session_state.my_regions=st.multiselect("regions",regions)
    with st.form("index_form"):
         
         form_cols=st.columns(2)
         if st.session_state.theme =="major":
            form_cols[0].write("major indexes")
            st.session_state.index_str=form_cols[1].selectbox("indexes",indexes)
         if st.session_state.theme =="global":
            form_cols[0].write("global indexes")
            st.session_state.index_str=form_cols[1].selectbox("indexes",global_indexes)
         
         if len(st.session_state.my_regions)>0 and st.session_state.theme =="regional":
                indexes=get_regions_indexes(wiki_indexes,st.session_state.my_regions)
                if len(indexes):
                    st.session_state.index_str=form_cols[1].selectbox("indexes",indexes)
         if len(st.session_state.my_countries)>0 and st.session_state.theme =="national":
                    indexes=get_countries_indexes(wiki_indexes,st.session_state.my_countries)
                    
                    index_selected=form_cols[1].selectbox("indexes",indexes, format_func=lambda x:f"{x['name']}({x['symbol']})")
                    st.session_state.index_str=index_selected["symbol"]
         
         choose_button=st.form_submit_button("choose")
         
    def apply_filter(filt_dict):
        op=filt_dict["operation"]
        val=filt_dict["value"]
        filter_name=filt_dict["name"]

        if filter_name in st.session_state.tab_data.columns:
            fn_filter=st.session_state.tab_data.apply(lambda row: float(str(row[filter_name]).strip("%"))>val,axis=1)
            if op=="less":
                fn_filter=st.session_state.tab_data.apply(lambda row: float(str(row[filter_name]).strip("%"))<val,axis=1)
            
                
            st.session_state.tab_data=st.session_state.tab_data[fn_filter]
                

    def trigger_filter_value():
        
            
            
            with st.form("filter_form"):   
                    for el in st.session_state.filters:
                        cols=st.columns(2)
                        p={}
                        p["name"]=el
                        p["operation"]=cols[0].selectbox(label="operation",options=operations,key=el+"op")
                        p["value"]=cols[1].number_input(el)   
                        st.session_state.filters_dicts.append(p)
                    add_filter_buttons=st.form_submit_button("add filters")
            
            if add_filter_buttons:
                    
                    for el in st.session_state.filters_dicts:
                        apply_filter(el)
                    
                         
    
    filters_column=st.columns(1)
    m_filters=["Price (End of Day)","52 Week Price % Change","Volume","% Change"]
    operations=["greatter","less"]
    filters_column[0].multiselect(label="filters",key="filters",options=m_filters)
    if len(st.session_state.filters):
        trigger_filter_value()
       
    
    if choose_button:
        #print(st.session_state.index_str)
        st.session_state.index=Index(st.session_state.index_str)

        st.session_state.tab_data=st.session_state.index.lood_data()
        
    if not(st.session_state.tab_data is None):   
            with st.form("form"):
                choose_button=True
                sectors_list=st.session_state.tab_data['sectors'].unique()
                industries_list=st.session_state.tab_data['industries'].unique()
                cols = st.columns(3)
    
                sectors= cols[0].multiselect("sectors",sectors_list)
                industries=cols[1].multiselect("industries",industries_list)
                marketcap=cols[2].multiselect("market cap",["Small","Mid","Large","Mega"])
                filter_button=st.form_submit_button(label='apply filters')
    
            if filter_button:
                    
                    if len(sectors)>0:
                        st.session_state.tab_data=st.session_state.tab_data[st.session_state.tab_data.sectors.isin(list(sectors))]
                    if len(industries)>0:
                         st.session_state.tab_data=st.session_state.tab_data[st.session_state.tab_data.sectors.isin(list(industries))]
            st.dataframe(st.session_state.tab_data)

tabs=["explore","screen"]

#select_tabs = option_menu(None,tabs,icons=None,default_index=0, orientation="horizontal")

#if select_tabs=="explore":
    #explore_stock()
#if select_tabs=="screen":
    #screen_stocks()
def deepbot():
   
    qualifiers=["most","main","major","top","best","flop","worst"]
    paramets=["ticker","symbol","index","stock","sector","industry","region"]
    comparatifs=["greater","less","superior","inferior","between","equal"] 
    metrics=["volume","change","performance",""]
    named_qualifiers=["gainers","loosers"]
    named_functions=["screener"]
    actions=["build","create","show","screen"]
    questions=[
        "build screener",
        "major indexes",
        "main sectors",
        "ticker",
        "index"
    ]
    if "quest_dict" not in st.session_state:
        st.session_state.quest_dict={}

    if "questions" not in st.session_state:
        st.session_state.questions=[]
    def is_number(val:str):
        if val.isdigit():
            return True
        try:
            float(val)
            return True
        except:
            return False
        
        return val.isnumeric()
    def get_most_similar(l_words,question):
        return sorted(l_words,key=lambda x:nlp.compute_cosine_similarity(question,x),reverse=True)[0]
    def get_nlp(q):
        quest_dict={}
        tockens=q.split()
        for t in tockens:
            if t in  qualifiers:
                quest_dict["qualif"]=t
            if t in  actions:
                quest_dict["action"]=t
            if t in  paramets:
                quest_dict["paramets"]=t
            if t in  metrics:
                if quest_dict.get("metrics"):
                    quest_dict["metrics"].append(t)  
                else :
                    quest_dict["metrics"]=[t]
            if t in comparatifs:
                 if quest_dict.get("comparatifs"):
                    quest_dict["comparatifs"].append(t)
                 else:
                    quest_dict["comparatifs"]=[t]
            if is_number(t):
                if quest_dict.get("refs"):
                    quest_dict["refs"].append(t)
                else:
                    quest_dict["refs"]=[t]

        st.session_state.quest_dict=quest_dict
    question = st.text_input("question")
    
    st.session_state.questions.append(question)

    for q in st.session_state.questions:
            rep=get_most_similar(questions,q)
            named_entities=nlp.get_named_entities(q)
            get_nlp(q)
            for ne in named_entities:
                if ne[1]=="GPE":
                    rep=rep+" in " + ne[0]
                if ne[1]=="ORG":
                    rep=rep+" of " + ne[0]
            #st.write("do you mean:",rep)
            #st.write(get_lemmatize(q))
            if st.session_state.quest_dict.get("metrics"):
                l=len(st.session_state.quest_dict["metrics"])
                st.subheader("your screener is:")
            
                for i in range(l):
                    metrcs=st.session_state.quest_dict["metrics"]
                    comps=st.session_state.quest_dict["comparatifs"]
                    refs=st.session_state.quest_dict["refs"]
                    sf=f"{metrcs[i]} {comps[i]} {refs[i]}"
                    st.write(sf)
        

    
   
            
with st.sidebar:
    page=st.radio("choose",options=("explorer","screener","deepbot"))
    
if page =="screener":
    screen_stocks()
if page=="explorer":
    explore_stock()
if page=="deepbot":
    deepbot()