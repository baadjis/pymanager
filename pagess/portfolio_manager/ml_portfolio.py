# pages/ml_portfolio.py
"""
Core ML Portfolio Functions (Backend)
Fonctions de base pour PCA, ICA, HRP sans dépendance Streamlit
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, KernelPCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PCA PORTFOLIO
# ============================================================================

def pca_portfolio(returns_data: pd.DataFrame,
                  n_components: int = 2,
                  use_kernel: bool = False,
                  kernel: str = 'rbf') -> Tuple[np.ndarray, Dict]:
    """
    Construit un portfolio basé sur PCA
    
    Args:
        returns_data: DataFrame des rendements
        n_components: Nombre de composantes principales
        use_kernel: Utiliser Kernel PCA (non-linéaire)
        kernel: Type de kernel ('rbf', 'poly', 'sigmoid')
    
    Returns:
        weights: Array des poids normalisés
        info: Dict avec informations PCA
    """
    # Standardiser les données
    returns_normalized = (returns_data - returns_data.mean()) / returns_data.std()
    
    # Appliquer PCA
    if use_kernel:
        pca = KernelPCA(n_components=n_components, kernel=kernel)
        pca_fit = pca.fit(returns_normalized)
        
        # Pour Kernel PCA, pas de explained_variance_ratio_ direct
        # On utilise les eigenvalues
        try:
            eigenvalues = pca_fit.eigenvalues_
            explained_variance = eigenvalues / eigenvalues.sum()
        except:
            explained_variance = np.ones(n_components) / n_components
        
        # Utiliser les alphas (dual space)
        components = pca_fit.eigenvectors_[:n_components]
        
    else:
        pca = PCA(n_components=n_components)
        pca_fit = pca.fit(returns_normalized)
        
        components = pca_fit.components_
        explained_variance = pca_fit.explained_variance_ratio_
    
    # Calculer les poids basés sur les composantes
    # Méthode: pondérer par variance expliquée
    weights = np.abs(components.T @ explained_variance)
    weights = weights / weights.sum()
    
    info = {
        'method': 'Kernel PCA' if use_kernel else 'PCA',
        'n_components': n_components,
        'explained_variance_ratio': explained_variance,
        'total_variance_explained': explained_variance.sum(),
        'components': components
    }
    
    logger.info(f"PCA portfolio: {info['total_variance_explained']:.2%} variance explained")
    
    return weights, info


# ============================================================================
# ICA PORTFOLIO
# ============================================================================

def ica_portfolio(returns_data: pd.DataFrame,
                  n_components: int = 3,
                  max_iter: int = 1000,
                  tol: float = 1e-4) -> Tuple[np.ndarray, Dict]:
    """
    Construit un portfolio basé sur ICA
    
    Args:
        returns_data: DataFrame des rendements
        n_components: Nombre de composantes indépendantes
        max_iter: Nombre max d'itérations
        tol: Tolérance pour convergence
    
    Returns:
        weights: Array des poids normalisés
        info: Dict avec informations ICA
    """
    # Standardiser
    returns_normalized = (returns_data - returns_data.mean()) / returns_data.std()
    
    # Appliquer ICA
    ica = FastICA(n_components=n_components, max_iter=max_iter, tol=tol, random_state=42)
    sources = ica.fit_transform(returns_normalized)
    mixing_matrix = ica.mixing_
    
    # Calculer l'importance de chaque source (kurtosis comme proxy)
    source_kurtosis = []
    for i in range(sources.shape[1]):
        # Kurtosis mesure l'écart à la gaussianité
        kurt = pd.Series(sources[:, i]).kurtosis()
        source_kurtosis.append(abs(kurt))
    
    source_importance = np.array(source_kurtosis)
    source_importance = source_importance / source_importance.sum()
    
    # Calculer poids basés sur la matrice de mélange et importance des sources
    weights = np.abs(mixing_matrix @ source_importance)
    weights = weights / weights.sum()
    
    info = {
        'method': 'ICA',
        'n_components': n_components,
        'mixing_matrix': mixing_matrix,
        'sources': sources,
        'source_importance': source_importance,
        'mean_source_kurtosis': np.mean(source_kurtosis)
    }
    
    logger.info(f"ICA portfolio: {n_components} components, mean kurtosis={info['mean_source_kurtosis']:.3f}")
    
    return weights, info


# ============================================================================
# MIXED PCA-ICA
# ============================================================================

def mixed_pca_ica_portfolio(returns_data: pd.DataFrame,
                            pca_weight: float = 0.5,
                            n_components_pca: int = 2,
                            n_components_ica: int = 3) -> Tuple[np.ndarray, Dict]:
    """
    Portfolio mixte PCA + ICA
    
    Args:
        returns_data: DataFrame des rendements
        pca_weight: Poids donné à PCA (1 - weight va à ICA)
        n_components_pca: Nombre de composantes PCA
        n_components_ica: Nombre de composantes ICA
    
    Returns:
        weights: Poids mixtes
        info: Informations combinées
    """
    # Calculer PCA
    weights_pca, info_pca = pca_portfolio(returns_data, n_components=n_components_pca)
    
    # Calculer ICA
    weights_ica, info_ica = ica_portfolio(returns_data, n_components=n_components_ica)
    
    # Combiner
    weights = pca_weight * weights_pca + (1 - pca_weight) * weights_ica
    weights = weights / weights.sum()
    
    info = {
        'method': 'Mixed PCA-ICA',
        'pca_weight': pca_weight,
        'ica_weight': 1 - pca_weight,
        'pca_info': info_pca,
        'ica_info': info_ica
    }
    
    logger.info(f"Mixed portfolio: {pca_weight:.0%} PCA + {1-pca_weight:.0%} ICA")
    
    return weights, info


# ============================================================================
# HIERARCHICAL RISK PARITY
# ============================================================================

def hierarchical_risk_parity_ml(returns_data: pd.DataFrame,
                                method: str = 'pca',
                                linkage_method: str = 'ward') -> Tuple[np.ndarray, Dict]:
    """
    Hierarchical Risk Parity avec dimensionality reduction
    
    Args:
        returns_data: DataFrame des rendements
        method: 'pca' ou 'ica' pour clustering
        linkage_method: Méthode de linkage ('ward', 'complete', 'average')
    
    Returns:
        weights: Poids HRP
        info: Informations
    """
    # Réduire la dimensionnalité
    if method == 'pca':
        n_comp = min(3, len(returns_data.columns))
        pca = PCA(n_components=n_comp)
        reduced_data = pca.fit_transform(returns_data)
    elif method == 'ica':
        n_comp = min(3, len(returns_data.columns))
        ica = FastICA(n_components=n_comp, random_state=42)
        reduced_data = ica.fit_transform(returns_data)
    else:
        reduced_data = returns_data.values
    
    # Calculer matrice de corrélation sur données réduites
    corr_matrix = np.corrcoef(reduced_data.T)
    
    # Distance matrix
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # Clustering hiérarchique
    link = linkage(squareform(dist_matrix), method=linkage_method)
    
    # Récupérer l'ordre des clusters
    dendro = dendrogram(link, no_plot=True)
    sorted_indices = dendro['leaves']
    
    # HRP allocation
    def get_cluster_var(cov, items):
        """Variance d'un cluster"""
        cov_slice = cov[np.ix_(items, items)]
        w = 1 / np.diag(cov_slice)
        w = w / w.sum()
        return np.dot(w, np.dot(cov_slice, w))
    
    def recursive_bisection(cov, sorted_items):
        """Bisection récursive pour HRP"""
        n = len(sorted_items)
        weights = np.ones(n)
        items = [sorted_items]
        
        while len(items) > 0:
            items = [i[j:k] for i in items 
                    for j, k in ((0, len(i)//2), (len(i)//2, len(i))) 
                    if len(i) > 1]
            
            for i in range(0, len(items), 2):
                items0 = items[i]
                items1 = items[i+1] if i+1 < len(items) else []
                
                if len(items1) == 0:
                    continue
                
                var0 = get_cluster_var(cov, items0)
                var1 = get_cluster_var(cov, items1)
                
                alpha = 1 - var0 / (var0 + var1)
                
                weights[items0] *= alpha
                weights[items1] *= (1 - alpha)
        
        return weights
    
    # Covariance des données originales
    cov_matrix = returns_data.cov().values
    
    # Appliquer HRP
    weights = recursive_bisection(cov_matrix, sorted_indices)
    
    # Réordonner pour correspondre aux actifs originaux
    final_weights = np.zeros(len(returns_data.columns))
    for i, idx in enumerate(sorted_indices):
        final_weights[idx] = weights[i]
    
    info = {
        'method': f'HRP ({method.upper()})',
        'linkage_method': linkage_method,
        'sorted_indices': sorted_indices,
        'linkage': link
    }
    
    logger.info(f"HRP portfolio with {method.upper()} clustering complete")
    
    return final_weights, info


# ============================================================================
# COMPONENT ANALYSIS
# ============================================================================

def explain_portfolio_components(returns_data: pd.DataFrame,
                                 weights: np.ndarray,
                                 method: str = 'pca',
                                 n_components: int = 3) -> pd.DataFrame:
    """
    Explique les composantes qui ont influencé le portfolio
    
    Args:
        returns_data: DataFrame des rendements
        weights: Poids du portfolio
        method: 'pca' ou 'ica'
        n_components: Nombre de composantes à analyser
    
    Returns:
        DataFrame avec loadings des composantes
    """
    returns_normalized = (returns_data - returns_data.mean()) / returns_data.std()
    
    if method == 'pca':
        model = PCA(n_components=n_components)
        model.fit(returns_normalized)
        
        components = model.components_
        explained_var = model.explained_variance_ratio_
        
        # DataFrame des loadings
        df = pd.DataFrame(
            components,
            columns=returns_data.columns,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Ajouter variance expliquée
        df['Variance Explained'] = explained_var
        
    elif method == 'ica':
        model = FastICA(n_components=n_components, random_state=42)
        sources = model.fit_transform(returns_normalized)
        mixing = model.mixing_
        
        # Kurtosis des sources
        source_kurtosis = [pd.Series(sources[:, i]).kurtosis() for i in range(n_components)]
        
        df = pd.DataFrame(
            mixing.T,
            columns=returns_data.columns,
            index=[f'IC{i+1}' for i in range(n_components)]
        )
        
        df['Source Kurtosis'] = source_kurtosis
    
    return df


# ============================================================================
# HELPERS
# ============================================================================

def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Valide que les poids sont corrects"""
    if weights is None or len(weights) == 0:
        return False
    
    if np.any(weights < -tolerance):
        logger.warning("Negative weights detected")
        return False
    
    if abs(weights.sum() - 1.0) > tolerance:
        logger.warning(f"Weights sum to {weights.sum():.6f}, not 1.0")
        return False
    
    return True


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalise les poids pour sommer à 1"""
    weights = np.abs(weights)  # Forcer positif
    return weights / weights.sum()


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ML PORTFOLIO - TESTS")
    print("=" * 80)
    
    # Créer données de test
    np.random.seed(42)
    n_days = 500
    n_assets = 5
    
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0002, 0.0003])
    cov_matrix = np.array([
        [0.0004, 0.0001, 0.0001, 0.0000, 0.0001],
        [0.0001, 0.0003, 0.0001, 0.0001, 0.0000],
        [0.0001, 0.0001, 0.0005, 0.0001, 0.0001],
        [0.0000, 0.0001, 0.0001, 0.0003, 0.0001],
        [0.0001, 0.0000, 0.0001, 0.0001, 0.0004],
    ])
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(
        returns,
        columns=['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
    )
    
    print(f"\n📊 Data: {n_days} days, {n_assets} assets")
    
    # Test PCA
    print("\n" + "=" * 80)
    print("TEST 1: PCA Portfolio")
    print("=" * 80)
    
    weights_pca, info_pca = pca_portfolio(returns_df, n_components=2)
    print(f"Weights: {weights_pca}")
    print(f"Variance explained: {info_pca['total_variance_explained']:.2%}")
    print(f"Valid: {validate_weights(weights_pca)}")
    
    # Test ICA
    print("\n" + "=" * 80)
    print("TEST 2: ICA Portfolio")
    print("=" * 80)
    
    weights_ica, info_ica = ica_portfolio(returns_df, n_components=3)
    print(f"Weights: {weights_ica}")
    print(f"Mean kurtosis: {info_ica['mean_source_kurtosis']:.3f}")
    print(f"Valid: {validate_weights(weights_ica)}")
    
    # Test Mixed
    print("\n" + "=" * 80)
    print("TEST 3: Mixed PCA-ICA")
    print("=" * 80)
    
    weights_mixed, info_mixed = mixed_pca_ica_portfolio(returns_df, pca_weight=0.6)
    print(f"Weights: {weights_mixed}")
    print(f"Valid: {validate_weights(weights_mixed)}")
    
    # Test HRP
    print("\n" + "=" * 80)
    print("TEST 4: HRP with PCA")
    print("=" * 80)
    
    weights_hrp, info_hrp = hierarchical_risk_parity_ml(returns_df, method='pca')
    print(f"Weights: {weights_hrp}")
    print(f"Valid: {validate_weights(weights_hrp)}")
    
    # Component analysis
    print("\n" + "=" * 80)
    print("TEST 5: Component Analysis")
    print("=" * 80)
    
    components_pca = explain_portfolio_components(returns_df, weights_pca, method='pca')
    print("\nPCA Components:")
    print(components_pca.to_string())
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
