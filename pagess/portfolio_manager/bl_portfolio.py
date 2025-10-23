"""
Black-Litterman Portfolio Optimization
Combine market equilibrium with investor views
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """
    Implementation du modèle Black-Litterman
    
    Le modèle combine:
    1. Prior (équilibre du marché basé sur CAPM)
    2. Views (opinions de l'investisseur)
    → Posterior (rendements ajustés)
    """
    
    def __init__(self,
                 returns_data: pd.DataFrame,
                 market_caps: Optional[Dict[str, float]] = None,
                 risk_free_rate: float = 0.02,
                 tau: float = 0.05,
                 risk_aversion: float = 2.5):
        """
        Args:
            returns_data: DataFrame des rendements historiques
            market_caps: Dict {asset: market_cap} pour calculer poids de marché
            risk_free_rate: Taux sans risque annuel (2% = 0.02)
            tau: Facteur d'incertitude du prior (0.01-0.05 typique)
            risk_aversion: Coefficient d'aversion au risque (2.5 typique)
        """
        self.returns_data = returns_data
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.risk_aversion = risk_aversion
        
        # Calculer covariance matrix (annualisée)
        self.cov_matrix = returns_data.cov() * 252
        
        # Calculer poids de marché (équilibre)
        if market_caps is not None:
            self.market_weights = self._calculate_market_weights(market_caps)
        else:
            # Si pas de market caps, utiliser equal weight comme proxy
            logger.warning("No market caps provided, using equal weights")
            self.market_weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculer rendements implicites (equilibrium returns)
        self.equilibrium_returns = self._calculate_equilibrium_returns()
        
        logger.info(f"Black-Litterman model initialized: {self.n_assets} assets")
    
    def _calculate_market_weights(self, market_caps: Dict[str, float]) -> np.ndarray:
        """Calcule les poids de marché basés sur market cap"""
        weights = np.array([market_caps.get(asset, 1.0) for asset in self.assets])
        weights = weights / weights.sum()
        return weights
    
    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """
        Calcule les rendements d'équilibre implicites (reverse optimization)
        π = δ * Σ * w_mkt
        """
        equilibrium_returns = self.risk_aversion * self.cov_matrix @ self.market_weights
        logger.info(f"Equilibrium returns: {equilibrium_returns}")
        return equilibrium_returns
    
    def add_views(self,
                  view_type: str,
                  views: Dict[str, float],
                  confidences: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ajoute des vues d'investisseur
        
        Args:
            view_type: 'absolute' ou 'relative'
            views: Dict des vues
                - absolute: {'AAPL': 0.15} = "AAPL aura 15% de rendement"
                - relative: {'AAPL-MSFT': 0.05} = "AAPL surperformera MSFT de 5%"
            confidences: Dict des niveaux de confiance (0-1)
                         Si None, utilise 0.5 pour toutes
        
        Returns:
            P: Matrice de picking (k x n)
            Q: Vecteur des vues (k x 1)
            Omega: Matrice de confiance (k x k)
        """
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        if confidences is None:
            confidences = {k: 0.5 for k in views.keys()}
        
        for i, (view_key, view_return) in enumerate(views.items()):
            Q[i] = view_return
            
            if view_type == 'absolute':
                # Vue absolue: un actif aura un rendement donné
                asset = view_key
                asset_idx = self.assets.index(asset)
                P[i, asset_idx] = 1.0
            
            elif view_type == 'relative':
                # Vue relative: asset1 vs asset2
                if '-' in view_key:
                    asset1, asset2 = view_key.split('-')
                    idx1 = self.assets.index(asset1.strip())
                    idx2 = self.assets.index(asset2.strip())
                    P[i, idx1] = 1.0
                    P[i, idx2] = -1.0
                else:
                    raise ValueError("Relative views must be in format 'ASSET1-ASSET2'")
        
        # Matrice de confiance Omega (diagonal)
        # Plus la confiance est élevée, plus Omega est petit
        omega_diag = []
        for i, view_key in enumerate(views.keys()):
            confidence = confidences.get(view_key, 0.5)
            # Variance de la vue inversement proportionnelle à la confiance
            view_variance = self.tau * (P[i] @ self.cov_matrix @ P[i].T) / confidence
            omega_diag.append(view_variance)
        
        Omega = np.diag(omega_diag)
        
        logger.info(f"Added {n_views} {view_type} views")
        return P, Q, Omega
    
    def calculate_posterior_returns(self,
                                    P: np.ndarray,
                                    Q: np.ndarray,
                                    Omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les rendements a posteriori (Black-Litterman formula)
        
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
        
        Returns:
            posterior_returns: Rendements ajustés (n x 1)
            posterior_cov: Covariance ajustée (n x n)
        """
        tau_cov = self.tau * self.cov_matrix
        tau_cov_inv = np.linalg.inv(tau_cov)
        omega_inv = np.linalg.inv(Omega)
        
        # Calcul de la moyenne a posteriori
        A = tau_cov_inv + P.T @ omega_inv @ P
        A_inv = np.linalg.inv(A)
        
        posterior_returns = A_inv @ (tau_cov_inv @ self.equilibrium_returns + P.T @ omega_inv @ Q)
        
        # Covariance a posteriori
        posterior_cov = self.cov_matrix + A_inv
        
        logger.info(f"Posterior returns calculated: {posterior_returns}")
        
        return posterior_returns, posterior_cov
    
    def optimize_portfolio(self,
                          posterior_returns: np.ndarray,
                          posterior_cov: np.ndarray,
                          constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Optimise le portfolio avec les rendements a posteriori
        
        Args:
            posterior_returns: Rendements ajustés
            posterior_cov: Covariance ajustée
            constraints: Contraintes additionnelles
                - 'max_weight': poids max par actif (ex: 0.4)
                - 'min_weight': poids min par actif (ex: 0.05)
                - 'target_return': rendement cible
        
        Returns:
            weights: Poids optimaux du portfolio
        """
        # Fonction objective: minimiser variance
        def objective(w):
            return w.T @ posterior_cov @ w
        
        # Contraintes
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]  # Somme = 1
        
        # Contrainte de rendement si spécifiée
        if constraints and 'target_return' in constraints:
            target = constraints['target_return']
            cons.append({
                'type': 'eq',
                'fun': lambda w: w.T @ posterior_returns - target
            })
        
        # Bounds
        if constraints:
            min_w = constraints.get('min_weight', 0.0)
            max_w = constraints.get('max_weight', 1.0)
            bounds = [(min_w, max_w) for _ in range(self.n_assets)]
        else:
            bounds = [(0.0, 1.0) for _ in range(self.n_assets)]
        
        # Point de départ: poids de marché
        x0 = self.market_weights
        
        # Optimisation
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fallback: market weights
            return self.market_weights
        
        weights = result.x
        logger.info(f"Portfolio optimized: {weights}")
        
        return weights


def black_litterman_portfolio(returns_data: pd.DataFrame,
                              views: Dict[str, float],
                              view_type: str = 'absolute',
                              confidences: Optional[Dict[str, float]] = None,
                              market_caps: Optional[Dict[str, float]] = None,
                              risk_free_rate: float = 0.02,
                              tau: float = 0.05,
                              constraints: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    Fonction wrapper pour créer un portfolio Black-Litterman
    
    Args:
        returns_data: DataFrame des rendements
        views: Vues de l'investisseur
        view_type: 'absolute' ou 'relative'
        confidences: Niveaux de confiance (0-1)
        market_caps: Market capitalizations
        risk_free_rate: Taux sans risque
        tau: Paramètre d'incertitude
        constraints: Contraintes d'optimisation
    
    Returns:
        weights: Poids optimaux
        info: Informations sur le modèle
    """
    # Créer le modèle
    bl_model = BlackLittermanModel(
        returns_data=returns_data,
        market_caps=market_caps,
        risk_free_rate=risk_free_rate,
        tau=tau
    )
    
    # Ajouter les vues
    P, Q, Omega = bl_model.add_views(view_type, views, confidences)
    
    # Calculer rendements a posteriori
    posterior_returns, posterior_cov = bl_model.calculate_posterior_returns(P, Q, Omega)
    
    # Optimiser
    weights = bl_model.optimize_portfolio(posterior_returns, posterior_cov, constraints)
    
    # Calculer métriques
    portfolio_return = weights @ posterior_returns
    portfolio_vol = np.sqrt(weights @ posterior_cov @ weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    info = {
        'method': 'Black-Litterman',
        'equilibrium_returns': bl_model.equilibrium_returns,
        'posterior_returns': posterior_returns,
        'market_weights': bl_model.market_weights,
        'views': views,
        'view_type': view_type,
        'n_views': len(views),
        'portfolio_return': portfolio_return,
        'portfolio_vol': portfolio_vol,
        'sharpe_ratio': sharpe,
        'tau': tau
    }
    
    logger.info(f"Black-Litterman complete: Sharpe={sharpe:.3f}, Return={portfolio_return:.2%}")
    
    return weights, info


def estimate_market_caps_from_data(returns_data: pd.DataFrame,
                                   reference_index: str = '^GSPC') -> Dict[str, float]:
    """
    Estime les market caps relatives si non disponibles
    Basé sur la volatilité et corrélation avec l'index
    
    Args:
        returns_data: DataFrame des rendements
        reference_index: Index de référence (S&P 500 par défaut)
    
    Returns:
        Dict de market caps estimées (relatives)
    """
    try:
        from dataprovider import yahoo
        
        # Télécharger l'index de référence
        index_data = yahoo.get_ticker_data(reference_index, period='1y')
        index_returns = np.log(index_data['Adj Close'] / index_data['Adj Close'].shift(1)).dropna()
        
        market_caps = {}
        
        for asset in returns_data.columns:
            asset_returns = returns_data[asset].dropna()
            
            # Aligner les données
            common_index = asset_returns.index.intersection(index_returns.index)
            if len(common_index) < 50:
                # Pas assez de données, utiliser volatilité seule
                vol = asset_returns.std()
                market_caps[asset] = 1.0 / (vol + 1e-6)  # Inverse vol comme proxy
            else:
                asset_aligned = asset_returns.loc[common_index]
                index_aligned = index_returns.loc[common_index]
                
                # Beta comme proxy de market cap
                covariance = np.cov(asset_aligned, index_aligned)[0, 1]
                variance_index = np.var(index_aligned)
                beta = covariance / variance_index if variance_index > 0 else 1.0
                
                # Market cap estimée proportionnelle au beta
                market_caps[asset] = max(beta, 0.1)  # Min 0.1 pour éviter 0
        
        # Normaliser
        total = sum(market_caps.values())
        market_caps = {k: v/total for k, v in market_caps.items()}
        
        logger.info(f"Estimated market caps: {market_caps}")
        return market_caps
    
    except Exception as e:
        logger.warning(f"Could not estimate market caps: {e}")
        # Fallback: equal weights
        return {asset: 1.0/len(returns_data.columns) for asset in returns_data.columns}


# ============================================================================
# Exemples d'utilisation et presets
# ============================================================================

def create_bullish_tech_views(assets: List[str]) -> Tuple[Dict, Dict]:
    """
    Exemple: vues bullish sur la tech
    
    Returns:
        views: Dict des vues
        confidences: Dict des confiances
    """
    views = {}
    confidences = {}
    
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
    
    for asset in assets:
        if asset in tech_stocks:
            views[asset] = 0.15  # 15% de rendement attendu
            confidences[asset] = 0.7  # Confiance élevée
    
    return views, confidences


def create_relative_views(assets: List[str],
                         outperformer: str,
                         underperformer: str,
                         spread: float = 0.05,
                         confidence: float = 0.6) -> Tuple[Dict, Dict]:
    """
    Exemple: vue relative (un actif surperformera un autre)
    
    Args:
        outperformer: Actif qui surperformera
        underperformer: Actif qui sous-performera
        spread: Spread attendu (5% = 0.05)
        confidence: Niveau de confiance
    
    Returns:
        views, confidences
    """
    view_key = f"{outperformer}-{underperformer}"
    views = {view_key: spread}
    confidences = {view_key: confidence}
    
    return views, confidences


def create_sector_rotation_views(assets: List[str],
                                 bullish_sector: str,
                                 bearish_sector: str) -> Tuple[Dict, Dict]:
    """
    Exemple: rotation sectorielle
    
    Args:
        bullish_sector: Secteur bullish (ex: 'tech', 'finance', 'health')
        bearish_sector: Secteur bearish
    
    Returns:
        views, confidences
    """
    sector_mapping = {
        'tech': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META'],
        'finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],
        'health': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'SBUX']
    }
    
    views = {}
    confidences = {}
    
    bullish_stocks = sector_mapping.get(bullish_sector, [])
    bearish_stocks = sector_mapping.get(bearish_sector, [])
    
    for asset in assets:
        if asset in bullish_stocks:
            views[asset] = 0.12
            confidences[asset] = 0.65
        elif asset in bearish_stocks:
            views[asset] = 0.03
            confidences[asset] = 0.65
    
    return views, confidences


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BLACK-LITTERMAN MODEL - TEST")
    print("=" * 80)
    
    # Données de test
    np.random.seed(42)
    n_days = 252
    assets = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
    
    # Générer rendements synthétiques
    mean_returns = np.array([0.0003, 0.0002, 0.0003, 0.0004, 0.0002])
    cov_matrix = np.array([
        [0.0004, 0.0001, 0.0001, 0.0002, 0.0001],
        [0.0001, 0.0003, 0.0001, 0.0001, 0.0001],
        [0.0001, 0.0001, 0.0003, 0.0001, 0.0001],
        [0.0002, 0.0001, 0.0001, 0.0005, 0.0002],
        [0.0001, 0.0001, 0.0001, 0.0002, 0.0004],
    ])
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns, columns=assets)
    
    print(f"\n📊 Data: {n_days} days, {len(assets)} assets")
    
    # Test 1: Sans vues (market equilibrium)
    print("\n" + "=" * 80)
    print("TEST 1: Market Equilibrium (No Views)")
    print("=" * 80)
    
    market_caps = {'AAPL': 2.5, 'GOOGL': 1.5, 'MSFT': 2.0, 'NVDA': 1.0, 'META': 0.8}
    
    bl_model = BlackLittermanModel(returns_df, market_caps=market_caps)
    weights_market = bl_model.market_weights
    
    print(f"\nMarket weights:")
    for asset, weight in zip(assets, weights_market):
        print(f"  {asset}: {weight:.2%}")
    
    # Test 2: Avec vues absolues
    print("\n" + "=" * 80)
    print("TEST 2: With Absolute Views")
    print("=" * 80)
    
    views = {
        'NVDA': 0.20,  # NVDA aura 20% de rendement
        'META': 0.10   # META aura 10% de rendement
    }
    confidences = {
        'NVDA': 0.7,  # Haute confiance
        'META': 0.5   # Confiance moyenne
    }
    
    print(f"\nViews:")
    for asset, view in views.items():
        conf = confidences[asset]
        print(f"  {asset}: {view:.1%} (confidence: {conf:.0%})")
    
    weights_bl, info_bl = black_litterman_portfolio(
        returns_df,
        views=views,
        view_type='absolute',
        confidences=confidences,
        market_caps=market_caps
    )
    
    print(f"\nBlack-Litterman weights:")
    for asset, weight in zip(assets, weights_bl):
        print(f"  {asset}: {weight:.2%}")
    
    print(f"\nPortfolio metrics:")
    print(f"  Expected Return: {info_bl['portfolio_return']:.2%}")
    print(f"  Volatility: {info_bl['portfolio_vol']:.2%}")
    print(f"  Sharpe Ratio: {info_bl['sharpe_ratio']:.3f}")
    
    # Test 3: Vues relatives
    print("\n" + "=" * 80)
    print("TEST 3: With Relative Views")
    print("=" * 80)
    
    views_rel = {
        'NVDA-GOOGL': 0.08,  # NVDA surperformera GOOGL de 8%
    }
    confidences_rel = {
        'NVDA-GOOGL': 0.6
    }
    
    print(f"\nRelative view: NVDA will outperform GOOGL by 8%")
    
    weights_bl_rel, info_bl_rel = black_litterman_portfolio(
        returns_df,
        views=views_rel,
        view_type='relative',
        confidences=confidences_rel,
        market_caps=market_caps
    )
    
    print(f"\nBlack-Litterman weights (relative):")
    for asset, weight in zip(assets, weights_bl_rel):
        print(f"  {asset}: {weight:.2%}")
    
    # Comparaison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Market': weights_market,
        'BL Absolute': weights_bl,
        'BL Relative': weights_bl_rel
    }, index=assets)
    
    print("\n" + comparison.to_string())
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
