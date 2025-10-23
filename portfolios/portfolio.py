
from functools import cached_property, lru_cache
from typing import List, Tuple,Callable,NewType,Union
import pandas as pd
from dask import compute,delayed
#from yahoo import get_ticker_data, retrieve_data ,get_log_returns
from scipy.optimize import minimize
from scipy.stats import norm,linregress,skew,kurtosis
import numpy as np
from consts import TRADING_DAYS_PER_YEAR, TREASURY_BILL_RATE
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#from assets import Asset
#Optimizer=NewType("Optimizer",callable[Union[float,None],None])
def random_weights(weight_count):
   alphas = .05*np.ones(weight_count)

   ws = np.random.dirichlet(alphas, size=1)
   
   return  list(ws[0])
   
def get_log_returns(price_history: pd.DataFrame):
  prices = price_history['Adj Close']
  v=np.log(prices/prices.shift(1)).dropna()
  return v
    
class Portfolio:
  def __init__(self, assets: List[str],data,initial:float=100.0):
    self.assets = assets
    self.data=data
    self.assets_log_returns = get_log_returns(data)
    
    self.initial=initial
    self.weights=[]

  def set_weights(self,w):
    self.weights = w

  @classmethod
  def unsafe_optimize_with_risk_tolerance(cls,assets,data, risk_tolerance: float,initial=100.0):
    p=cls(assets,data,initial)
    wet=random_weights(len(assets))
    
    p.set_weights(wet)
    
    res = minimize(
      lambda w: p._variance(w) - risk_tolerance * p._expected_return(w),
      random_weights(len(p.assets)),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
      ],
      bounds=[(0., 1.) for i in range(len(p.assets))]
    )

    assert res.success, f'Optimization failed: {res.message}'
    v=list(res.x.reshape(-1, 1))
    p.weights=[list(t)[0] for t in v]
    
    return p
    
  @classmethod
  def optimize_with_risk_tolerance(cls, assets: List[str], data, 
                                 risk_tolerance: float, initial: float = 100.0):
    """
    Optimize portfolio for given risk tolerance.
    
    Args:
        assets: List of asset tickers
        data: Price data DataFrame
        risk_tolerance: Risk tolerance parameter (>= 0)
        initial: Initial investment amount
        
    Returns:
        Optimized Portfolio object
        
    Raises:
        ValueError: If risk_tolerance < 0
    """
    if risk_tolerance < 0:
        raise ValueError(f"Risk tolerance must be >= 0, got {risk_tolerance}")
    
    logger.info(f"Optimizing with risk tolerance: {risk_tolerance}")
    return Portfolio.unsafe_optimize_with_risk_tolerance(
        assets, data, risk_tolerance, initial
    )
    
  @classmethod
  def optimize_with_expected_return(cls,assets,data, expected_portfolio_return: float,initial=100.0):
    p=cls(assets,data,initial)
    wet=random_weights(len(assets))
    p.set_weights(wet)
    res = minimize(
      lambda w: p._variance(w),
      random_weights(len(assets)),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        {'type': 'eq', 'fun': lambda w: p._expected_return(w) - expected_portfolio_return},
      ],
      bounds=[(0., 1.) for i in range(len(p.assets))]
    )

    assert res.success, f'Optimization failed: {res.message}'
    v=list(res.x.reshape(-1, 1))
    p.weights=[list(t)[0] for t in v]
    
    return p
  @classmethod
  def optimize_sharpe_ratio(cls,assets,data,initial=100.0):
    # Maximize Sharpe ratio = minimize minus Sharpe ratio
    p=cls(assets,data,initial)
    wet=random_weights(len(assets))
    p.set_weights(wet)
    res = minimize(
      lambda w: -(p._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(p._variance(w)),
      random_weights(len(assets)),
      constraints=[
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
      ],
      bounds=[(0., 1.) for i in range(len(p.assets))]
    )

    assert res.success, f'Optimization failed: {res.message}'
    v=list(res.x.reshape(-1, 1))
    p.weights=[list(t)[0] for t in v]
    return p
    

  def _expected_return(self, w):
  
  
    ex=self._daily_returns(w).mean()*TRADING_DAYS_PER_YEAR
    
    return  ex
  @cached_property
  def covariance_matrix(self):
    return self.assets_log_returns.cov()*(TRADING_DAYS_PER_YEAR)**2
  @cached_property
  def correlation_matrix(self):
        return self.assets_log_returns.corr()


  def _variance(self, w):
    return np.transpose(w)@self.covariance_matrix@w
  

  def _sharp_ratio(self):

    return(self.expected_return - TREASURY_BILL_RATE / 100) / self.stdev
  @cached_property
  def skewness(self):
    return skew(self.daily_returns)
  
  @property
  def kurtosis(self):
    return kurtosis(self.daily_returns)
  @property
  def sharp_ratio(self):
    return self._sharp_ratio()
  @property
  def expected_return(self):
    return self._expected_return(self.weights)
  
  @cached_property
  def variance(self):
    return self._variance(self.weights)
  @cached_property
  def stdev(self):
    return np.sqrt(self.variance)
  
  def __repr__(self):
    return f'<Portfolio assets={[asset for asset in self.assets]}, expected return={self.expected_return}, variance={self.variance}>'
  
  @cached_property
  def stdev_mean(self):
    dd=delayed(self.daily_returns)
    stdev,mean_inv = compute(dd.std(),dd.mean())
    return stdev,mean_inv

  def VAR(self,conf_level:float = 0.05,num_days :int=1):
      stdev,mean_inv=self.stdev_mean
      zvalue =  norm.ppf(abs(conf_level))
      var_1d = -self.initial*(zvalue*stdev+mean_inv)
      
      return var_1d * np.sqrt(num_days)
  def Cornish_Fisher_var(self,conf_level:float = 0.05,num_days :int=1):
       stdev,mean_inv=self.stdev_mean
       z= norm.ppf(abs(conf_level))
       s=self.skewness
       k=self.kurtosis
       Z=z +(z**2-1)*s/6 + (z**3-z**3)*k/24-((2*z**3 -5**z)*s**2)/36
       return -self.initial*(Z*stdev+mean_inv)
  def _daily_returns(self,w):
    
    if len(w)==1:
      return self.assets_log_returns
    p=self.assets_log_returns.dot(w)
    return p

  @cached_property
  def daily_returns(self):
    return self._daily_returns(self.weights)

  @lru_cache
  def _regress(self,benchmark):
      
      return(linregress(benchmark.daily_returns,self.daily_returns)[0:2])
  
  def alpha(self,benchmark):
    return self._regress(benchmark)[1] 


  def beta(self,benchmark):
    return self._regress(benchmark)[0]  

  def treynor_ratio(self,benchmark):
    return(self.expected_return-TREASURY_BILL_RATE)/self.beta(benchmark)


