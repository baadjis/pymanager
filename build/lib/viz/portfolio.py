import time
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from matplotlib import rcParams
import numpy as np

from ml import ica, pca
from portfolios import Portfolio,factory
import pandas as pd
from dataprovider import yahoo



matplotlib.use('TkAgg')
rcParams['figure.figsize'] = 12, 9


def corr_heatmap(corr):
  
  mask = np.triu(np.ones_like(corr, dtype=np.bool))
  sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,mask=mask, annot=True)
  plt.show()

def plot_risk_return(portfolio,col:str,label:str):
    plt.plot(portfolio.stdev, portfolio.expected_return, col, markeredgewidth=5, markersize=20, label=label)

def pair_plot(a:str,b:str,period='2y'):
    
  data= yahoo.retrieve_data((a,b),period=period)
  portf=Portfolio([a,b],data=data)
  pds=portf.assets_log_returns
  pds.plot()
  plt.show()




def plot_pie(stocks,weights):
    #data=yahoo.retrieve_data(tuple(stocks),period="8y")
    #portfolio=create_portfolio_by_name(stocks,p_name,data=data)
    
    
    """plt.pie(portfolio.weights, labels=portfolio.assets, 
    autopct='%2.2f%%', pctdistance=0.8, startangle=90)
    """
    sectors_weights=yahoo.get_sectors_weights(stocks,weights)
    plt.pie(sectors_weights.values(), labels=sectors_weights.keys(), 
    autopct='%2.2f%%', pctdistance=0.8, startangle=90)
    plt.axis('equal')
    plt.show()

def plot(stocks,name,period=factory.back_period,show=True): 
  
  benchmark=factory.create_benchmark("SPY",period=period)
  portfolio_data=yahoo.retrieve_data(tuple(stocks),period=period)
  portfolio=factory.create_portfolio_by_name(stocks,name,data=portfolio_data)
  pds=pd.DataFrame()
  pds["portfolio"]=portfolio.daily_returns
  pds["benchmark"]=benchmark.daily_returns
  pcd=pd.DataFrame(portfolio.assets_log_returns)
  pds["pca_5"]=pca(pcd,5)
  pds["pca_1"]=pca(pcd,1)
  pds["ica"]=ica(pcd,1)
  pds.plot()
  if show:
    plt.show()
  else:
      st.pyplot(fig=plt)
  
def do_all(stocks,portname,col:str,label:str,data):

   portfolio=factory.create_portfolio_by_name(stocks,portname,data)
   plot_risk_return(portfolio,col,label)

def draw_efficient_frontier(stocks,data):
    # Drawing the efficient frontier
    X ,y = factory.get_unsafe_portfolios(stocks,data)
    plt.plot(X, y, 'k', linewidth=3, label='Efficient frontier')

def draw_optimized_porfolios(stocks,data):
    do_all(stocks,"risk0", 'm+', label='optimize_with_risk_tolerance(0)',data=data)
    do_all(stocks,"risk", 'r+', label='optimize_with_risk_tolerance(20)',data=data)
    do_all(stocks,"return", 'g+', label='optimize_with_expected_return(0.25)',data=data)
    do_all(stocks,"sharp", 'y+', label='optimize_sharpe_ratio()',data=data)


def plot_random_efficient(stocks,data,random_porfolios,show=True):
    X,y=random_porfolios
    col=[y1/x for x ,y1 in zip(X,y) ]

    plt.scatter(X,y, label='Random portfolios',c=col)
    plt.colorbar(label="sharpe ratio")

    # Drawing the efficient frontier
    draw_efficient_frontier(stocks,data)

    # Drawing optimized portfolios
    draw_optimized_porfolios(stocks,data=data)

    plt.xlabel('Portfolio standard deviation')
    plt.ylabel('Portfolio expected (logarithmic) return')
    plt.legend(loc='lower right')
    if show==True:
       plt.show()


def plot_markowitz_curve(stocks,n,data,show=True):
    
    #draw random portfolios
    #data=retrieve_data(tuple(stocks),period=period)
    random_portfolios=factory.generate_random_portpolios(stocks,n,data=data)
    plot_random_efficient(stocks,data,random_portfolios,show)
    


    


if __name__=="__main__":
    print("viz")