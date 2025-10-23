import numpy as np

from dask import delayed

#from yahoo import get_ticker_data, retrieve_data
from functools import lru_cache
from typing import List

import pandas as pd
from portfolio import Portfolio, random_weights
from dataprovider import yahoo 
back_period="8y"

def create_portfolio_with_assets(assets,weights,period=back_period):
    assert len(assets)==len(weights)
    assert sum(weights)==1
    data= yahoo.retrieve_data(tuple(assets),period=period)
    portfolio=Portfolio(assets=assets,data=data)
    portfolio.set_weights(weights)
    return portfolio
  #print(asset_name,portfolio.data)

def one_asset(asset_name,period='1y'):

  portfolio_data=yahoo.get_ticker_data(asset_name,period=period)
  portfolio=Portfolio([asset_name],data=portfolio_data)
  #print(asset_name,portfolio.data)
  portfolio.set_weights([1])
  return portfolio


def create_benchmark(name,period=back_period):
    benchmark=create_portfolio_with_assets([name],[1.0],period)
    return benchmark

def create_random_portfolio(stocks,data):
    portfolio = Portfolio(stocks,data)
    w=random_weights(len(stocks))
          
    portfolio.set_weights(w)
    
    return portfolio.stdev,portfolio.expected_return
def generate_random_portpolios(stocks,n,data):
    
      r=[]
      for i in range(n):
          v=delayed(create_random_portfolio)(stocks,data)
          r.append(v)
      
      Xy=delayed(zip)(*r)
      return Xy.compute()
      """
      # Drawing random portfolios
      for i in prange(n):
          
          portfolio = Portfolio(stocks,data)
          w=random_weights(len(stocks))
          
          portfolio.set_weights(w)
          
          X[i]=portfolio.stdev
          y[i]=portfolio.expected_return
      
      return X,y
      """




def create_portfolio_by_name(stocks,port,data,risk_tolerance=20.00,expected_return=0.25):

  p:Portfolio=None
  if port=="unsafe":
    p=Portfolio.unsafe_optimize_with_risk_tolerance(stocks,data,200)
  if  port=="sharp":
    p=Portfolio.optimize_sharpe_ratio(stocks,data)
  if port=="return":
    p=Portfolio.optimize_with_expected_return(stocks,data,expected_return)
  if port=="risk":
    p=Portfolio.optimize_with_risk_tolerance(stocks,data,risk_tolerance)
  if port=="risk0":
    p=Portfolio.optimize_with_risk_tolerance(stocks,data,0)
  return p
  


def add_unsafe_portfolio(rt,stocks,data):
    p=Portfolio.unsafe_optimize_with_risk_tolerance(stocks,data,risk_tolerance=rt)
    return p.stdev,p.expected_return


def get_unsafe_portfolios(stocks,data):
  
    l=np.linspace(-300, 200, 1000)
    delays=[]
    for rt in l:
      delay_portfolios=delayed(add_unsafe_portfolio)(rt,stocks,data)
      delays.append(delay_portfolios)
    portpolios_tuple= delayed(zip)(*delays)
    return portpolios_tuple.compute()
if __name__=="__main__":
    print("factory")
