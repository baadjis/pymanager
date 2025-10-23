"""
Backtesting Module - Test rigoureux de stratégies de portfolio
Supporte walk-forward optimization, import CSV, et sauvegarde DB
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import logging
from portfolio import Portfolio
from dataprovider import yahoo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Backtester pour stratégies de portfolio
    Supporte plusieurs méthodes de validation
    """
    
    def __init__(self,
                 assets: List[str],
                 data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 rebalance_frequency: str = 'monthly'):
        """
        Args:
            assets: Liste des actifs
            data: DataFrame des prix (Adj Close)
            initial_capital: Capital initial
            transaction_cost: Coût de transaction (0.1% = 0.001)
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
        """
        self.assets = assets
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        
        # Calculer les rendements
        from portfolio import get_log_returns
        self.returns = get_log_returns(data)
        
        # Préparer returns DataFrame pour multi-assets
        if len(assets) > 1:
            if isinstance(data.columns, pd.MultiIndex):
                self.returns_df = pd.DataFrame()
                for asset in assets:
                    prices = data[('Adj Close', asset)]
                    self.returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
            else:
                self.returns_df = self.returns if isinstance(self.returns, pd.DataFrame) else pd.DataFrame({assets[0]: self.returns})
        else:
            self.returns_df = pd.DataFrame({assets[0]: self.returns})
        
        logger.info(f"Backtester initialized: {len(assets)} assets, {len(self.returns_df)} periods")
    
    def train_test_split(self,
                        train_size: float = 0.7,
                        method: str = 'sequential') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split données en train/test
        
        Args:
            train_size: Proportion pour training (0.7 = 70%)
            method: 'sequential' ou 'random' (recommandé: sequential pour séries temporelles)
        
        Returns:
            train_data, test_data
        """
        n = len(self.returns_df)
        split_idx = int(n * train_size)
        
        if method == 'sequential':
            train = self.returns_df.iloc[:split_idx]
            test = self.returns_df.iloc[split_idx:]
        else:
            # Random split (non recommandé pour time series)
            indices = np.random.permutation(n)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            train = self.returns_df.iloc[train_indices]
            test = self.returns_df.iloc[test_indices]
        
        logger.info(f"Split: Train={len(train)} periods, Test={len(test)} periods")
        return train, test
    
    def walk_forward_optimization(self,
                                  strategy_func: Callable,
                                  train_window: int = 252,
                                  test_window: int = 63,
                                  step_size: int = 21,
                                  **strategy_params) -> Dict:
        """
        Walk-forward optimization (rolling window)
        
        Args:
            strategy_func: Fonction qui retourne des poids (ex: pca_portfolio)
            train_window: Taille fenêtre d'entraînement (252 = 1 an)
            test_window: Taille fenêtre de test (63 = 3 mois)
            step_size: Pas de déplacement (21 = 1 mois)
            strategy_params: Paramètres additionnels pour la stratégie
        
        Returns:
            Dictionnaire avec résultats du backtesting
        """
        logger.info("Starting walk-forward optimization...")
        
        results = {
            'periods': [],
            'train_dates': [],
            'test_dates': [],
            'weights_history': [],
            'returns': [],
            'portfolio_values': [self.initial_capital],
            'rebalance_costs': [],
            'sharpe_ratios': []
        }
        
        current_capital = self.initial_capital
        previous_weights = None
        
        # Itérer sur les fenêtres
        start_idx = 0
        while start_idx + train_window + test_window <= len(self.returns_df):
            # Fenêtre de training
            train_start = start_idx
            train_end = start_idx + train_window
            train_data = self.returns_df.iloc[train_start:train_end]
            
            # Fenêtre de test
            test_start = train_end
            test_end = min(train_end + test_window, len(self.returns_df))
            test_data = self.returns_df.iloc[test_start:test_end]
            
            logger.info(f"Period {len(results['periods'])+1}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
            
            # Entraîner la stratégie sur train
            try:
                weights, _ = strategy_func(train_data, **strategy_params)
                
                # Calculer coût de rebalancement
                if previous_weights is not None:
                    portfolio_change = np.abs(weights - previous_weights).sum()
                    rebalance_cost = portfolio_change * self.transaction_cost * current_capital
                else:
                    rebalance_cost = 0
                
                results['rebalance_costs'].append(rebalance_cost)
                
                # Tester sur test
                test_returns = (test_data.values * weights).sum(axis=1)
                period_return = np.exp(test_returns.sum()) - 1
                
                # Update capital
                current_capital = current_capital * (1 + period_return) - rebalance_cost
                
                # Sharpe du test period
                if len(test_returns) > 1:
                    sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
                else:
                    sharpe = 0
                
                # Enregistrer résultats
                results['periods'].append(len(results['periods']) + 1)
                results['train_dates'].append((train_data.index[0], train_data.index[-1]))
                results['test_dates'].append((test_data.index[0], test_data.index[-1]))
                results['weights_history'].append(weights.copy())
                results['returns'].append(period_return)
                results['portfolio_values'].append(current_capital)
                results['sharpe_ratios'].append(sharpe)
                
                previous_weights = weights
                
            except Exception as e:
                logger.error(f"Error in period {len(results['periods'])+1}: {e}")
                continue
            
            # Avancer la fenêtre
            start_idx += step_size
        
        # Calculer métriques globales
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        avg_sharpe = np.mean(results['sharpe_ratios']) if results['sharpe_ratios'] else 0
        total_costs = sum(results['rebalance_costs'])
        
        results['summary'] = {
            'total_return': total_return,
            'final_capital': current_capital,
            'avg_sharpe': avg_sharpe,
            'total_rebalance_costs': total_costs,
            'n_periods': len(results['periods'])
        }
        
        logger.info(f"Walk-forward complete: {len(results['periods'])} periods, {total_return:.2%} return")
        
        return results
    
    def simple_backtest(self,
                       weights: np.ndarray,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict:
        """
        Backtest simple avec poids fixes (buy & hold)
        
        Args:
            weights: Poids du portfolio
            start_date: Date de début (None = début des données)
            end_date: Date de fin (None = fin des données)
        
        Returns:
            Dictionnaire avec métriques de performance
        """
        # Filtrer par dates si nécessaire
        if start_date or end_date:
            returns = self.returns_df.loc[start_date:end_date]
        else:
            returns = self.returns_df
        
        # Calculer rendements du portfolio
        portfolio_returns = (returns.values * weights).sum(axis=1)
        
        # Valeur du portfolio au fil du temps
        portfolio_values = self.initial_capital * np.exp(np.cumsum(portfolio_returns))
        portfolio_values = np.insert(portfolio_values, 0, self.initial_capital)
        
        # Métriques
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(returns))
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Max Drawdown
        cumulative = portfolio_values / self.initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Sortino
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = (annual_return - 0.02) / downside_std if downside_std > 0 else 0
        
        results = {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown < 0 else 0,
            'final_capital': portfolio_values[-1],
            'dates': returns.index
        }
        
        return results
    
    def compare_strategies(self,
                          strategies: Dict[str, Tuple[Callable, Dict]],
                          method: str = 'walk_forward') -> pd.DataFrame:
        """
        Compare plusieurs stratégies
        
        Args:
            strategies: Dict {name: (strategy_func, params)}
            method: 'walk_forward' ou 'simple'
        
        Returns:
            DataFrame avec comparaison des métriques
        """
        comparison = []
        
        for name, (strategy_func, params) in strategies.items():
            logger.info(f"Testing strategy: {name}")
            
            try:
                if method == 'walk_forward':
                    results = self.walk_forward_optimization(strategy_func, **params)
                    metrics = results['summary']
                else:
                    # Pour simple backtest, on doit d'abord obtenir les poids
                    weights, _ = strategy_func(self.returns_df, **params)
                    results = self.simple_backtest(weights)
                    metrics = results
                
                comparison.append({
                    'Strategy': name,
                    'Total Return': f"{metrics.get('total_return', 0):.2%}",
                    'Sharpe Ratio': f"{metrics.get('avg_sharpe', metrics.get('sharpe_ratio', 0)):.3f}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Final Capital': f"${metrics.get('final_capital', 0):,.2f}"
                })
            
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                comparison.append({
                    'Strategy': name,
                    'Total Return': 'Error',
                    'Sharpe Ratio': 'Error',
                    'Max Drawdown': 'Error',
                    'Final Capital': 'Error'
                })
        
        return pd.DataFrame(comparison)


def load_portfolio_from_csv(csv_file) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Charge un portfolio depuis un fichier CSV
    
    Format attendu:
    Asset,Weight,Quantity (optionnel)
    AAPL,0.3,10
    GOOGL,0.4,5
    MSFT,0.3,8
    
    Returns:
        assets: Liste des tickers
        weights: Array des poids
        quantities: Array des quantités (None si non fourni)
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Vérifier colonnes requises
        required_cols = ['Asset', 'Weight']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        assets = df['Asset'].tolist()
        weights = df['Weight'].values
        
        # Normaliser les poids si nécessaire
        if abs(weights.sum() - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weights.sum():.3f}, normalizing...")
            weights = weights / weights.sum()
        
        # Quantités optionnelles
        quantities = df['Quantity'].values if 'Quantity' in df.columns else None
        
        logger.info(f"Loaded portfolio: {len(assets)} assets from CSV")
        
        return assets, weights, quantities
    
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def save_backtest_results_to_csv(results: Dict, filename: str):
    """
    Sauvegarde les résultats de backtesting en CSV
    
    Args:
        results: Résultats du backtest
        filename: Nom du fichier de sortie
    """
    # Créer DataFrame des résultats
    if 'periods' in results:  # Walk-forward results
        df = pd.DataFrame({
            'Period': results['periods'],
            'Return': results['returns'],
            'Portfolio_Value': results['portfolio_values'][1:],
            'Sharpe_Ratio': results['sharpe_ratios'],
            'Rebalance_Cost': results['rebalance_costs']
        })
    else:  # Simple backtest results
        df = pd.DataFrame({
            'Date': results['dates'],
            'Portfolio_Value': results['portfolio_values'][:-1],
            'Daily_Return': results['portfolio_returns']
        })
    
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BACKTESTING MODULE - TEST")
    print("=" * 80)
    
    # Setup
    assets = ['AAPL', 'GOOGL', 'MSFT']
    
    # Charger données
    print("\n1. Loading data...")
    data = yahoo.retrieve_data(tuple(assets), period='2y')
    
    # Créer backtester
    backtester = PortfolioBacktester(
        assets=assets,
        data=data,
        initial_capital=10000,
        transaction_cost=0.001
    )
    
    # Test 1: Simple backtest avec Equal Weight
    print("\n2. Simple Backtest (Equal Weight)")
    equal_weights = np.ones(len(assets)) / len(assets)
    results_simple = backtester.simple_backtest(equal_weights)
    
    print(f"   Total Return: {results_simple['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results_simple['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {results_simple['max_drawdown']:.2%}")
    print(f"   Final Capital: ${results_simple['final_capital']:,.2f}")
    
    # Test 2: Walk-forward avec PCA
    print("\n3. Walk-Forward Optimization (PCA)")
    from .ml_portfolio import pca_portfolio
    
    results_wf = backtester.walk_forward_optimization(
        strategy_func=pca_portfolio,
        train_window=252,
        test_window=63,
        step_size=21,
        n_components=2
    )
    
    summary = results_wf['summary']
    print(f"   Total Return: {summary['total_return']:.2%}")
    print(f"   Avg Sharpe: {summary['avg_sharpe']:.3f}")
    print(f"   Total Costs: ${summary['total_rebalance_costs']:,.2f}")
    print(f"   Periods: {summary['n_periods']}")
    
    # Test 3: Comparison
    print("\n4. Strategy Comparison")
    
    strategies = {
        'Equal Weight': (lambda df: (np.ones(len(assets)) / len(assets), {}), {}),
        'PCA (2 comp)': (pca_portfolio, {'n_components': 2}),
    }
    
    comparison = backtester.compare_strategies(strategies, method='simple')
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
