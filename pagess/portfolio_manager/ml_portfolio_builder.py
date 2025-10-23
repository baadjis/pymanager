"""
Machine Learning Portfolio Builder
Utilise PCA et ICA pour construire des portfolios basés sur l'analyse de composantes
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from portfolio import Portfolio
from factory import create_portfolio_by_name, create_benchmark
from dataprovider import yahoo
from database import get_portfolios, get_single_portfolio, save_portfolio
from utils import calculate_portfolio_current_value, format_pnl
from uiconfig import get_theme_colors
from .portfolio_helpers import (color_column , color_background_column,
cached_get_sectors_weights,METRIC_TOOLTIPS)


import numpy as np

from .ml_portfolio import (
    pca_portfolio, 
    ica_portfolio, 
    mixed_pca_ica_portfolio,
    hierarchical_risk_parity_ml,
    explain_portfolio_components
)


def build_ml_portfolio(assets, data, theme):
    """Build ML-based portfolio (PCA/ICA)"""
    st.markdown("#### Machine Learning Portfolio")
    st.info("🤖 Use dimensionality reduction to find optimal weights")
    
    # Sélection de la méthode ML
    ml_methods = ["PCA", "ICA", "Mixed PCA-ICA", "HRP with ML"]
    ml_method = st.selectbox("ML Method", ml_methods, 
                             help="Choose dimensionality reduction method")
    
    # Calculer les rendements
    from portfolio import get_log_returns
    returns_data = get_log_returns(data)
    
    # Si multi-actifs, créer un DataFrame approprié
    if len(assets) > 1:
        if isinstance(returns_data, pd.Series):
            returns_data = pd.DataFrame(returns_data)
        
        # S'assurer que les colonnes correspondent aux actifs
        if isinstance(data.columns, pd.MultiIndex):
            returns_df = pd.DataFrame()
            for asset in assets:
                try:
                    prices = data[('Adj Close', asset)]
                    returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
                except:
                    st.error(f"Could not process {asset}")
                    return
            returns_data = returns_df
        else:
            # Données à une seule colonne
            if len(assets) == 1:
                returns_data = pd.DataFrame({assets[0]: returns_data})
    
    # Vérifier les données
    if returns_data.empty or len(returns_data) < 30:
        st.error("❌ Insufficient data for ML methods (need at least 30 data points)")
        return
    
    st.success(f"✅ Using {len(returns_data)} data points for ML analysis")
    
    # Configuration selon la méthode
    weights = None
    info = None
    
    if ml_method == "PCA":
        st.markdown("##### PCA Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider(
                "Number of Components",
                min_value=1,
                max_value=min(5, len(assets)),
                value=min(2, len(assets)),
                help="Number of principal components to use"
            )
        
        with col2:
            use_kernel = st.checkbox(
                "Use Kernel PCA",
                value=False,
                help="Use non-linear kernel PCA (slower but can capture non-linear relationships)"
            )
        
        if st.button("Build PCA Portfolio", use_container_width=True):
            with st.spinner("Computing PCA..."):
                try:
                    weights, info = pca_portfolio(
                        returns_data, 
                        n_components=n_components,
                        use_kernel=use_kernel
                    )
                    
                    # Vérifier que les poids sont valides
                    if weights is None or len(weights) != len(assets):
                        st.error("❌ PCA failed to produce valid weights")
                        return
                    
                    # Afficher les résultats
                    st.success("✅ PCA Portfolio created!")
                    
                    # Métriques PCA
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Variance Explained",
                            f"{info['total_variance_explained']:.1%}"
                        )
                    with col2:
                        st.metric(
                            "Method",
                            info['method']
                        )
                    with col3:
                        st.metric(
                            "Components",
                            info['n_components']
                        )
                    
                    # Variance par composante
                    st.markdown("**Variance Explained per Component:**")
                    var_df = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(len(info['explained_variance_ratio']))],
                        'Variance': [f"{v:.2%}" for v in info['explained_variance_ratio']]
                    })
                    st.dataframe(var_df, use_container_width=True, hide_index=True)
                    
                except AttributeError as e:
                    st.error(f"❌ PCA configuration error: {e}")
                    st.info("💡 Try using standard PCA (uncheck 'Use Kernel PCA')")
                    return
                except ValueError as e:
                    st.error(f"❌ Data error: {e}")
                    st.info("💡 Check that you have enough data points and valid returns")
                    return
                except Exception as e:
                    st.error(f"❌ Error building PCA portfolio: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
                    return
    
    elif ml_method == "ICA":
        st.markdown("##### ICA Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider(
                "Number of Components",
                min_value=2,
                max_value=min(len(assets), 10),
                value=min(3, len(assets)),
                help="Number of independent components to extract"
            )
        
        with col2:
            max_iter = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Maximum iterations for convergence"
            )
        
        if st.button("Build ICA Portfolio", use_container_width=True):
            with st.spinner("Computing ICA..."):
                try:
                    weights, info = ica_portfolio(
                        returns_data,
                        n_components=n_components,
                        max_iter=max_iter
                    )
                    
                    st.success("✅ ICA Portfolio created!")
                    
                    # Métriques ICA
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Mean Kurtosis",
                            f"{info['mean_source_kurtosis']:.3f}",
                            help="Higher kurtosis indicates more non-Gaussian sources"
                        )
                    with col2:
                        st.metric(
                            "Components",
                            info['n_components']
                        )
                    
                    # Importance des sources
                    if 'source_importance' in info:
                        st.markdown("**Source Importance:**")
                        source_df = pd.DataFrame({
                            'Source': [f'IC{i+1}' for i in range(len(info['source_importance']))],
                            'Importance': [f"{v:.2%}" for v in info['source_importance']]
                        })
                        st.dataframe(source_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error building ICA portfolio: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
    
    elif ml_method == "Mixed PCA-ICA":
        st.markdown("##### Mixed PCA-ICA Configuration")
        
        pca_weight = st.slider(
            "PCA Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Weight given to PCA (1-weight goes to ICA)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            n_comp_pca = st.slider("PCA Components", 1, min(5, len(assets)), min(2, len(assets)))
        with col2:
            n_comp_ica = st.slider("ICA Components", 2, min(10, len(assets)), min(3, len(assets)))
        
        if st.button("Build Mixed Portfolio", use_container_width=True):
            with st.spinner("Computing Mixed PCA-ICA..."):
                try:
                    weights, info = mixed_pca_ica_portfolio(
                        returns_data,
                        pca_weight=pca_weight,
                        n_components_pca=n_comp_pca,
                        n_components_ica=n_comp_ica
                    )
                    
                    st.success("✅ Mixed Portfolio created!")
                    
                    # Afficher les métriques des deux méthodes
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**PCA Info:**")
                        st.write(f"Variance: {info['pca_info']['total_variance_explained']:.1%}")
                    with col2:
                        st.markdown("**ICA Info:**")
                        st.write(f"Kurtosis: {info['ica_info']['mean_source_kurtosis']:.3f}")
                    
                except Exception as e:
                    st.error(f"Error building mixed portfolio: {e}")
                    return
    
    elif ml_method == "HRP with ML":
        st.markdown("##### Hierarchical Risk Parity with ML")
        
        hrp_method = st.radio(
            "Clustering Method",
            ["pca", "ica"],
            help="Use PCA or ICA for hierarchical clustering"
        )
        
        if st.button("Build HRP Portfolio", use_container_width=True):
            with st.spinner("Computing HRP..."):
                try:
                    weights, info = hierarchical_risk_parity_ml(
                        returns_data,
                        method=hrp_method
                    )
                    
                    st.success("✅ HRP Portfolio created!")
                    
                    st.info(f"Method: {info['method']}")
                    
                except Exception as e:
                    st.error(f"Error building HRP portfolio: {e}")
                    return
    
    # Si un portfolio a été créé, l'afficher
    if weights is not None and info is not None:
        # Créer le portfolio
        portfolio = Portfolio(assets, data)
        portfolio.set_weights(list(weights))
        
        # Afficher les poids
        st.markdown("---")
        st.markdown("### 🎯 ML Portfolio Weights")
        
        weights_df = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.4f}" for w in weights],
            'Percentage': [f"{w*100:.2f}%" for w in weights]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        # Graphique des poids
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(
                x=assets,
                y=weights,
                marker=dict(
                    color=weights,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Weight")
                )
            )
        ])
        fig.update_layout(
            title="Portfolio Weights Distribution",
            xaxis_title="Assets",
            yaxis_title="Weight",
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explication des composantes
        with st.expander("📊 Component Analysis"):
            try:
                method_name = 'pca' if 'PCA' in ml_method else 'ica'
                components = explain_portfolio_components(
                    returns_data, 
                    weights, 
                    method=method_name
                )
                
                st.markdown("**Component Loadings:**")
                st.dataframe(components.style.format("{:.4f}"), use_container_width=True)
                
                # Heatmap des composantes
                import plotly.express as px
                
                fig = px.imshow(
                    components.drop(columns=[c for c in components.columns if 'Variance' in c or 'Source' in c]),
                    labels=dict(x="Assets", y="Components", color="Loading"),
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                fig.update_layout(
                    title="Component Loadings Heatmap",
                    template=theme['plotly_template'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate component analysis: {e}")
        
        # Afficher les résultats du portfolio avec display_portfolio_results
        st.markdown("---")
        from .portfolio_helpers import display_portfolio_results
        display_portfolio_results(
            portfolio, 
            assets, 
            "ml",
            method=ml_method.lower().replace(' ', '_'),
            ml_info=info
        )
