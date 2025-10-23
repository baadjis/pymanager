from functools import cached_property, lru_cache
from typing import List, Tuple,Callable,NewType,Union
import pandas as pd
from dask import compute,delayed
from scipy.optimize import minimize
from scipy.stats import norm,linregress,skew,kurtosis
import numpy as np
from consts import TRADING_DAYS_PER_YEAR, TREASURY_BILL_RATE
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
      return self.assets_log_returns.iloc[:, 0]
    p=self.assets_log_returns.dot(w)
    return p

  @cached_property
  def daily_returns(self):
    return self._daily_returns(self.weights)

  @lru_cache
  def _regress(self,benchmark):
    bench_returns = benchmark.daily_returns
    port_returns = self.daily_returns
    
    common_index = bench_returns.index.intersection(port_returns.index)
    bench_aligned = bench_returns.loc[common_index]
    port_aligned = port_returns.loc[common_index]
    
    x = np.array(bench_aligned).flatten()
    y = np.array(port_aligned).flatten()
    
    logger.info(f"Regression: benchmark shape={x.shape}, portfolio shape={y.shape}")
    
    lin_reg = linregress(x, y)
    return (lin_reg.slope, lin_reg.intercept)
  
  def alpha(self,benchmark):
    return self._regress(benchmark)[1] 

  def beta(self,benchmark):
    return self._regress(benchmark)[0]  

  def treynor_ratio(self,benchmark):
    return(self.expected_return-TREASURY_BILL_RATE/100)/self.beta(benchmark)

  # ============= NOUVEAUX RATIOS =============
  
  def downside_deviation(self, target_return: float = 0.0):
    """
    Calcule la déviation à la baisse (downside deviation)
    Mesure la volatilité des rendements négatifs
    """
    returns = self.daily_returns
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return 0.0
    downside_variance = np.mean((downside_returns - target_return) ** 2)
    return np.sqrt(downside_variance * TRADING_DAYS_PER_YEAR)
  
  def sortino_ratio(self, target_return: float = 0.0):
    """
    Ratio de Sortino: mesure le rendement ajusté du risque de baisse
    Similaire au Sharpe mais utilise seulement la volatilité négative
    """
    downside_dev = self.downside_deviation(target_return)
    if downside_dev == 0:
        return 0.0
    return (self.expected_return - TREASURY_BILL_RATE / 100) / downside_dev
  
  @cached_property
  def max_drawdown(self):
    """
    Calcule le drawdown maximum (perte maximale depuis un pic)
    """
    cumulative_returns = (1 + self.daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()
  
  def calmar_ratio(self):
    """
    Ratio de Calmar: rendement annualisé / drawdown maximum
    Mesure le rendement par unité de risque extrême
    """
    max_dd = abs(self.max_drawdown)
    if max_dd == 0:
        return 0.0
    return self.expected_return / max_dd
  
  def information_ratio(self, benchmark):
    """
    Ratio d'information: alpha / tracking error
    Mesure l'excès de rendement par unité de risque actif
    """
    tracking_error = self.tracking_error(benchmark)
    if tracking_error == 0:
        return 0.0
    return self.alpha(benchmark) * TRADING_DAYS_PER_YEAR / tracking_error
  
  def tracking_error(self, benchmark):
    """
    Tracking error: écart-type des différences de rendements
    Mesure à quel point le portfolio suit le benchmark
    """
    port_returns = self.daily_returns
    bench_returns = benchmark.daily_returns
    
    common_index = port_returns.index.intersection(bench_returns.index)
    port_aligned = port_returns.loc[common_index]
    bench_aligned = bench_returns.loc[common_index]
    
    active_returns = port_aligned - bench_aligned
    return active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
  
  def omega_ratio(self, threshold: float = 0.0):
    """
    Ratio Omega: probabilité de gains / probabilité de pertes
    Mesure tous les moments de la distribution
    """
    returns = self.daily_returns
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    if losses.sum() == 0:
        return float('inf')
    
    return gains.sum() / losses.sum()
  
  @cached_property
  def tail_ratio(self):
    """
    Tail Ratio: 95e percentile / 5e percentile (absolu)
    Mesure l'asymétrie des queues de distribution
    """
    returns = self.daily_returns
    percentile_95 = returns.quantile(0.95)
    percentile_5 = abs(returns.quantile(0.05))
    
    if percentile_5 == 0:
        return 0.0
    
    return percentile_95 / percentile_5
  
  def value_at_risk_historical(self, conf_level: float = 0.05):
    """
    VaR historique (non paramétrique)
    Utilise directement les quantiles historiques
    """
    returns = self.daily_returns
    var_percentile = returns.quantile(conf_level)
    return -self.initial * var_percentile
  
  def conditional_var(self, conf_level: float = 0.05):
    """
    CVaR (Expected Shortfall): moyenne des pertes au-delà du VaR
    Mesure la perte moyenne dans le pire des scénarios
    """
    returns = self.daily_returns
    var_percentile = returns.quantile(conf_level)
    tail_losses = returns[returns <= var_percentile]
    
    if len(tail_losses) == 0:
        return 0.0
    
    return -self.initial * tail_losses.mean()
  
  @cached_property
  def annualized_volatility(self):
    """
    Volatilité annualisée (alias pour stdev)
    """
    return self.stdev
  
  def sterling_ratio(self):
    """
    Ratio de Sterling: rendement / average drawdown
    Variante du Calmar ratio utilisant le drawdown moyen
    """
    cumulative_returns = (1 + self.daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    
    avg_drawdown = abs(drawdowns[drawdowns < 0].mean())
    
    if avg_drawdown == 0:
        return 0.0
    
    return self.expected_return / avg_drawdown
  
  def burke_ratio(self):
    """
    Ratio de Burke: rendement / racine carrée de la somme des drawdowns²
    Pénalise plus fortement les drawdowns multiples
    """
    cumulative_returns = (1 + self.daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    
    negative_drawdowns = drawdowns[drawdowns < 0]
    
    if len(negative_drawdowns) == 0:
        return 0.0
    
    burke_denominator = np.sqrt((negative_drawdowns ** 2).sum())
    
    if burke_denominator == 0:
        return 0.0
    
    return self.expected_return / burke_denominator
  
  @cached_property
  def positive_periods(self):
    """
    Pourcentage de périodes avec rendements positifs
    """
    returns = self.daily_returns
    return (returns > 0).sum() / len(returns)
  
  @cached_property
  def negative_periods(self):
    """
    Pourcentage de périodes avec rendements négatifs
    """
    return 1 - self.positive_periods
  
  def gain_to_pain_ratio(self):
    """
    Ratio Gain-to-Pain: somme des gains / somme absolue des pertes
    """
    returns = self.daily_returns
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf')
    
    return gains / losses
  
  def ulcer_index(self):
    """
    Ulcer Index: mesure la profondeur et la durée des drawdowns
    Plus bas = mieux
    """
    cumulative_returns = (1 + self.daily_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    
    squared_drawdowns = drawdowns ** 2
    ulcer = np.sqrt(squared_drawdowns.mean())
    
    return ulcer
  
  def martin_ratio(self):
    """
    Ratio de Martin: rendement / Ulcer Index
    Alternative au Sharpe utilisant l'Ulcer Index
    """
    ulcer = self.ulcer_index()
    
    if ulcer == 0:
        return 0.0
    
    return (self.expected_return - TREASURY_BILL_RATE / 100) / ulcer
