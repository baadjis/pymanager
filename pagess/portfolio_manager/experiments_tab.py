# pages/experiments_tab.py
"""
Onglet Experiments pour Portfolio Manager - VERSION CORRIGÉE
Comprend : Model Comparison, Backtesting, ML/RL Training, Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from portfolio import Portfolio, get_log_returns
from dataprovider import yahoo
from database import get_portfolios, get_single_portfolio, save_portfolio
from uiconfig import get_theme_colors
from .ml_rl_training import render_ml_rl_training
# Import du module backtesting - vérifier le chemin
user_id = st.session_state.user_id
try:
    from backtesting import PortfolioBacktester, load_portfolio_from_csv, save_backtest_results_to_csv
except ImportError:
    try:
        from .backtesting import PortfolioBacktester, load_portfolio_from_csv, save_backtest_results_to_csv
    except ImportError:
        st.error("⚠️ Backtesting module not found. Please ensure backtesting.py is in the correct directory.")
        PortfolioBacktester = None


def render_experiments_tab():
    """Onglet experiments principal"""
    st.markdown("### 🧪 Portfolio Experiments Lab")
    
    st.info("""
    **Experiments Lab** vous permet de:
    - Comparer différents modèles de portfolio
    - Effectuer du backtesting rigoureux
    - Entraîner et tester des modèles ML/RL
    - Exporter les résultats pour analyse externe
    """)
    
    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "📊 Model Comparison",
        "⏮️ Backtesting",
        "🤖 ML/RL Training",
        "📤 Export"
    ])
    
    with exp_tab1:
        render_model_comparison()
    
    with exp_tab2:
        render_backtesting()
    
    with exp_tab3:
         
        render_ml_rl_training()
    
    with exp_tab4:
        render_export_results()


# ============================================================================
# MODEL COMPARISON - AVEC IMPORTS CORRIGÉS
# ============================================================================

def run_model(model_type, assets, data, returns_df):
    """
    Exécute un modèle et retourne poids + métriques
    CORRIGÉ : Imports fonctionnels pour tous les modèles
    """
    
    if model_type == "equal":
        weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "markowitz":
        from factory import create_portfolio_by_name
        portfolio = create_portfolio_by_name(assets, "sharp", data)
        weights = np.array(portfolio.weights)
    
    elif model_type == "black_litterman":
        try:
            # CORRECTION : Import depuis bl_portfolio.py (pas bl_portfolio_builder)
            from .bl_portfolio import (
                black_litterman_portfolio, 
                estimate_market_caps_from_data
            )
            market_caps = estimate_market_caps_from_data(returns_df)
            weights, _ = black_litterman_portfolio(
                returns_data=returns_df,
                views={},
                market_caps=market_caps
            )
        except ImportError as e:
            st.warning(f"⚠️ Black-Litterman module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ Black-Litterman failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "pca":
        try:
            # CORRECTION : Import depuis ml_portfolio.py (le core module)
            from .ml_portfolio import pca_portfolio
            weights, _ = pca_portfolio(returns_df, n_components=min(2, len(assets)))
        except ImportError as e:
            st.warning(f"⚠️ PCA module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ PCA failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "ica":
        try:
           
            from .ml_portfolio import ica_portfolio
            weights, _ = ica_portfolio(returns_df, n_components=min(3, len(assets)))
        except ImportError as e:
            st.warning(f"⚠️ ICA module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ ICA failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "hrp":
        try:
            
            from .ml_portfolio import hierarchical_risk_parity_ml
            weights, _ = hierarchical_risk_parity_ml(returns_df, method='pca')
        except ImportError as e:
            st.warning(f"⚠️ HRP module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ HRP failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "rl_reinforce":
        try:
            # CORRECTION : Import depuis rl_portfolio_simple.py
            from .rl_portfolio import get_rl_portfolio_weights
            weights, _ = get_rl_portfolio_weights(
                returns_df, 
                agent_type='reinforce', 
                n_episodes=30
            )
        except ImportError as e:
            st.warning(f"⚠️ RL module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ RL REINFORCE failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "rl_ac":
        try:
            # CORRECTION : Import depuis rl_portfolio_simple.py
            from .rl_portfolio import get_rl_portfolio_weights
            weights, _ = get_rl_portfolio_weights(
                returns_df, 
                agent_type='actor_critic', 
                n_episodes=30
            )
        except ImportError as e:
            st.warning(f"⚠️ RL module not found: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
        except Exception as e:
            st.warning(f"⚠️ RL Actor-Critic failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "beta":
        try:
            betas = yahoo.get_assets_beta(assets)
            weights = np.array(betas) / sum(betas)
        except Exception as e:
            st.warning(f"⚠️ Beta calculation failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    elif model_type == "risk_parity":
        try:
            vols = returns_df.std().values
            if np.any(vols == 0):
                raise ValueError("Zero volatility detected")
            weights = (1 / vols) / (1 / vols).sum()
        except Exception as e:
            st.warning(f"⚠️ Risk Parity failed: {e}. Using equal weights.")
            weights = np.ones(len(assets)) / len(assets)
    
    else:
        weights = np.ones(len(assets)) / len(assets)
    
    # Normaliser les poids (sécurité)
    weights = np.array(weights)
    weights = np.abs(weights)  # Forcer positif
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(assets)) / len(assets)
    
    # Calculer les métriques
    try:
        portfolio = Portfolio(assets, data)
        portfolio.set_weights(list(weights))
        
        metrics = {
            'expected_return': portfolio.expected_return,
            'volatility': portfolio.stdev,
            'sharpe_ratio': portfolio.sharp_ratio,
            'sortino_ratio': portfolio.sortino_ratio(),
            'max_drawdown': portfolio.max_drawdown,
            'calmar_ratio': portfolio.calmar_ratio(),
        }
    except Exception as e:
        st.warning(f"⚠️ Could not calculate metrics: {e}")
        metrics = {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
        }
    
    return weights, metrics


def render_model_comparison():
    """Compare plusieurs modèles de portfolio"""
    st.markdown("#### 📊 Model Comparison")
    
    st.markdown("""
    Comparez les performances de différents modèles d'allocation de portfolio
    sur les mêmes données historiques.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_input(
            "Stock Tickers",
            placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
            key="comparison_tickers"
        )
    
    with col2:
        period = st.selectbox(
            "Historical Period",
            ["1y", "2y", "3y", "5y", "10y"],
            index=2,
            key="comparison_period"
        )
    
    if not tickers:
        st.info("👆 Enter tickers to start comparison")
        return
    
    assets = [t.strip().upper() for t in tickers.split(",")]
    
    # Sélection des modèles à comparer
    st.markdown("---")
    st.markdown("**Select Models to Compare:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_equal = st.checkbox("Equal Weight", value=True)
        use_markowitz = st.checkbox("Markowitz (Sharpe)", value=True)
        use_black_litterman = st.checkbox("Black-Litterman", value=True)  # Activé par défaut
        use_risk_parity = st.checkbox("Risk Parity", value=False)
    
    with col2:
        use_pca = st.checkbox("PCA", value=True)
        use_ica = st.checkbox("ICA", value=False)
        use_hrp = st.checkbox("HRP", value=False)
    
    with col3:
        use_rl_reinforce = st.checkbox("RL (REINFORCE)", value=False)
        use_rl_ac = st.checkbox("RL (Actor-Critic)", value=False)  # Désactivé par défaut (long)
        use_beta = st.checkbox("Beta Weighted", value=False)
    
    # Bouton de comparaison
    if st.button("🚀 Run Comparison", use_container_width=True, type="primary"):
        
        # Charger les données
        with st.spinner("Loading data..."):
            try:
                data = yahoo.retrieve_data(tuple(assets), period=period)
                if data.empty:
                    st.error("Could not load data")
                    return
                
                returns_data = get_log_returns(data)
                
                # Préparer DataFrame pour multi-assets
                if len(assets) > 1:
                    if isinstance(data.columns, pd.MultiIndex):
                        returns_df = pd.DataFrame()
                        for asset in assets:
                            prices = data[('Adj Close', asset)]
                            returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
                    else:
                        returns_df = returns_data if isinstance(returns_data, pd.DataFrame) else pd.DataFrame({assets[0]: returns_data})
                else:
                    returns_df = pd.DataFrame({assets[0]: returns_data})
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        
        # Exécuter les comparaisons
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models_to_test = []
        if use_equal: models_to_test.append(("Equal Weight", "equal"))
        if use_markowitz: models_to_test.append(("Markowitz", "markowitz"))
        if use_black_litterman: models_to_test.append(("Black-Litterman", "black_litterman"))
        if use_pca: models_to_test.append(("PCA", "pca"))
        if use_ica: models_to_test.append(("ICA", "ica"))
        if use_hrp: models_to_test.append(("HRP", "hrp"))
        if use_rl_reinforce: models_to_test.append(("RL (REINFORCE)", "rl_reinforce"))
        if use_rl_ac: models_to_test.append(("RL (Actor-Critic)", "rl_ac"))
        if use_beta: models_to_test.append(("Beta Weighted", "beta"))
        if use_risk_parity: models_to_test.append(("Risk Parity", "risk_parity"))
        
        total_models = len(models_to_test)
        
        for idx, (name, model_type) in enumerate(models_to_test):
            status_text.text(f"Testing {name}... ({idx+1}/{total_models})")
            progress_bar.progress((idx + 1) / total_models)
            
            try:
                weights, metrics = run_model(model_type, assets, data, returns_df)
                results[name] = {
                    'weights': weights,
                    'metrics': metrics
                }
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {e}")
                continue
        
        progress_bar.empty()
        status_text.text("✅ Comparison complete!")
        
        # Afficher les résultats
        if results:
            display_comparison_results(results, assets)
        else:
            st.error("No models completed successfully")


def display_comparison_results(results, assets):
    """Affiche les résultats de comparaison"""
    st.markdown("---")
    st.markdown("### 📊 Comparison Results")
    
    # Table de comparaison
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            'Model': name,
            'Sharpe': f"{data['metrics']['sharpe_ratio']:.3f}",
            'Sortino': f"{data['metrics']['sortino_ratio']:.3f}",
            'Return': f"{data['metrics']['expected_return']:.2%}",
            'Volatility': f"{data['metrics']['volatility']:.2%}",
            'Max DD': f"{data['metrics']['max_drawdown']:.2%}",
            'Calmar': f"{data['metrics']['calmar_ratio']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Graphiques de comparaison
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk-Return scatter
        fig1 = go.Figure()
        
        for name, data in results.items():
            fig1.add_trace(go.Scatter(
                x=[data['metrics']['volatility']],
                y=[data['metrics']['expected_return']],
                mode='markers+text',
                name=name,
                text=[name],
                textposition='top center',
                marker=dict(size=15)
            ))
        
        fig1.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility (Annual)",
            yaxis_title="Expected Return (Annual)",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Sharpe ratios bar chart
        fig2 = go.Figure()
        
        sharpe_values = [results[name]['metrics']['sharpe_ratio'] for name in results.keys()]
        colors = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sharpe_values]
        
        fig2.add_trace(go.Bar(
            x=list(results.keys()),
            y=sharpe_values,
            marker_color=colors,
            text=[f"{x:.3f}" for x in sharpe_values],
            textposition='outside'
        ))
        
        fig2.update_layout(
            title="Sharpe Ratios Comparison",
            yaxis_title="Sharpe Ratio",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Weights comparison
    st.markdown("### 🎯 Weights Comparison")
    
    weights_data = {}
    for name, data in results.items():
        weights_data[name] = data['weights']
    
    weights_df = pd.DataFrame(weights_data, index=assets)
    
    fig3 = go.Figure()
    
    for model in weights_df.columns:
        fig3.add_trace(go.Bar(
            name=model,
            x=assets,
            y=weights_df[model],
            text=[f"{x:.1%}" for x in weights_df[model]],
            textposition='outside'
        ))
    
    fig3.update_layout(
        title="Portfolio Weights by Model",
        xaxis_title="Assets",
        yaxis_title="Weight",
        barmode='group',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Winner
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    st.success(f"🏆 **Best Model (Sharpe Ratio):** {best_model[0]} ({best_model[1]['metrics']['sharpe_ratio']:.3f})")


# ============================================================================
# BACKTESTING - Importer les fonctions depuis le document précédent
# ============================================================================

def render_backtesting():
    """Backtesting rigoureux avec walk-forward et import CSV"""
    st.markdown("#### ⏮️ Backtesting")
    
    st.markdown("""
    Testez la performance d'un modèle sur des données historiques avec:
    - 📊 Simple Backtest (Buy & Hold)
    - 🔄 Walk-Forward Optimization
    - 📤 Import Portfolio (CSV)
    - 💾 Save Results to Database
    """)
    
    # Options d'input
    st.markdown("---")
    st.markdown("### 📥 Portfolio Input")
    
    input_method = st.radio(
        "How do you want to provide the portfolio?",
        ["Build from Scratch", "Upload CSV", "Load from Database"],
        horizontal=True
    )
    
    assets = None
    weights = None
    portfolio_name = None
    
    if input_method == "Build from Scratch":
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_input(
                "Stock Tickers",
                placeholder="AAPL, MSFT, GOOGL",
                key="backtest_tickers"
            )
        
        with col2:
            period = st.selectbox(
                "Historical Period",
                ["1y", "2y", "3y", "5y"],
                index=1,
                key="backtest_period"
            )
        
        if tickers:
            assets = [t.strip().upper() for t in tickers.split(",")]
            
            # Sélection de la stratégie
            strategy = st.selectbox(
                "Select Strategy",
                ["Equal Weight", "Markowitz (Sharpe)", "PCA", "ICA", "RL (Actor-Critic)", "BL (Black-Litterman)"],
                key="backtest_strategy"
            )
            
            # Charger données et calculer poids
            if st.button("Calculate Weights", key="calc_weights"):
                with st.spinner("Loading data and calculating weights..."):
                    try:
                        data = yahoo.retrieve_data(tuple(assets), period=period)
                        
                        if strategy == "Equal Weight":
                            weights = np.ones(len(assets)) / len(assets)
                        
                        elif strategy == "Markowitz (Sharpe)":
                            from factory import create_portfolio_by_name
                            portfolio = create_portfolio_by_name(assets, "sharp", data)
                            weights = np.array(portfolio.weights)
                        
                        elif strategy in ["PCA", "ICA", "RL (Actor-Critic)", "BL (Black-Litterman)"]:
                            returns = get_log_returns(data)
                            if len(assets) > 1:
                                returns_df = pd.DataFrame()
                                for asset in assets:
                                    prices = data[('Adj Close', asset)]
                                    returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
                            else:
                                returns_df = pd.DataFrame({assets[0]: returns})
                            
                            if strategy == "PCA":
                                from .ml_portfolio import pca_portfolio  # ✅ CORRECT: depuis ml_portfolio.py
                                weights, _ = pca_portfolio(returns_df, n_components=2)
                            
                            elif strategy == "ICA":
                                from .ml_portfolio import ica_portfolio  # ✅ CORRECT: depuis ml_portfolio.py
                                weights, _ = ica_portfolio(returns_df, n_components=3)
                            
                            elif strategy == "RL (Actor-Critic)":
                                from .rl_portfolio_simple import get_rl_portfolio_weights  # ✅ CORRECT: depuis rl_portfolio_simple.py
                                weights, _ = get_rl_portfolio_weights(returns_df, agent_type='actor_critic', n_episodes=30)
                            
                            elif strategy == "BL (Black-Litterman)":
                                # ✅ CORRECTION: Import depuis bl_portfolio.py (CORE) pas bl_portfolio_builder.py (UI)
                                from .bl_portfolio import black_litterman_portfolio, estimate_market_caps_from_data
                                
                                # Estimer les market caps
                                market_caps = estimate_market_caps_from_data(returns_df)
                                
                                # Pas de vues = équilibre de marché
                                weights, info = black_litterman_portfolio(
                                    returns_data=returns_df,
                                    views={},  # Vues vides = utilise seulement l'équilibre
                                    market_caps=market_caps,
                                    risk_free_rate=0.02,
                                    tau=0.05
                                )
                                
                                st.info(f"ℹ️ Black-Litterman: Market equilibrium portfolio (no views)")
                        
                        # Vérifier que weights n'est pas None
                        if weights is None:
                            st.error("❌ Failed to calculate weights")
                            return
                        
                        st.session_state.backtest_weights = weights
                        st.session_state.backtest_assets = assets
                        st.session_state.backtest_data = data
                        st.success("✅ Weights calculated!")
                        
                        # Afficher les poids
                        weights_df = pd.DataFrame({
                            'Asset': assets,
                            'Weight': [f"{w:.2%}" for w in weights]
                        })
                        st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
    
    elif input_method == "Upload CSV":
        st.markdown("""
        **CSV Format Expected:**
        ```
        Asset,Weight,Quantity (optional)
        AAPL,0.3,10
        GOOGL,0.4,5
        MSFT,0.3,8
        ```
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="upload_portfolio_csv")
        
        if uploaded_file is not None:
            try:
                from .backtesting import load_portfolio_from_csv
                assets, weights, quantities = load_portfolio_from_csv(uploaded_file)
                
                st.session_state.backtest_assets = assets
                st.session_state.backtest_weights = weights
                
                st.success(f"✅ Loaded {len(assets)} assets from CSV")
                
                # Afficher le portfolio
                display_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [f"{w:.2%}" for w in weights]
                })
                if quantities is not None:
                    display_df['Quantity'] = quantities
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Charger les données
                period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1, key="csv_period")
                
                if st.button("Load Data for Backtesting", key="csv_load"):
                    with st.spinner("Loading market data..."):
                        data = yahoo.retrieve_data(tuple(assets), period=period)
                        st.session_state.backtest_data = data
                        st.success("✅ Data loaded!")
                
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    elif input_method == "Load from Database":
        try:
            portfolios = list(get_portfolios())
            portfolio_names = [p['name'] for p in portfolios]
            
            if not portfolio_names:
                st.info("No saved portfolios. Create one first!")
                return
            
            selected = st.selectbox("Select Portfolio", portfolio_names, key="backtest_load_portfolio")
            
            if st.button("Load Portfolio", key="db_load"):
                portfolio = get_single_portfolio(user_id,selected)
                assets = portfolio['assets']
                weights = portfolio['weights']
                
                st.session_state.backtest_assets = assets
                st.session_state.backtest_weights = np.array(weights)
                portfolio_name = selected
                
                st.success(f"✅ Loaded portfolio: {selected}")
                
                # Afficher
                display_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [f"{w:.2%}" for w in weights]
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Charger données
                period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1, key="db_period")
                
                if st.button("Load Data for Backtesting", key="db_data_load"):
                    with st.spinner("Loading market data..."):
                        data = yahoo.retrieve_data(tuple(assets), period=period)
                        st.session_state.backtest_data = data
                        st.success("✅ Data loaded!")
        
        except Exception as e:
            st.error(f"Error loading from database: {e}")
    
    # Si on a des données chargées, proposer les options de backtesting
    if ('backtest_weights' in st.session_state and 
        'backtest_assets' in st.session_state and 
        'backtest_data' in st.session_state):
        
        st.markdown("---")
        st.markdown("### 🧪 Backtesting Configuration")
        
        backtest_type = st.radio(
            "Backtesting Method",
            ["Simple Backtest (Buy & Hold)", "Walk-Forward Optimization"],
            horizontal=True
        )
        
        assets = st.session_state.backtest_assets
        weights = st.session_state.backtest_weights
        data = st.session_state.backtest_data
        
        if backtest_type == "Simple Backtest (Buy & Hold)":
            render_simple_backtest(assets, weights, data, portfolio_name)
        else:
            render_walkforward_backtest(assets, weights, data, portfolio_name)
            
            
            
def render_simple_backtest(assets, weights, data, portfolio_name=None):
    """Simple backtest interface"""
    st.markdown("#### 📊 Simple Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            value=10000.0,
            step=1000.0
        )
    
    with col2:
        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        ) / 100
    
    with col3:
        test_size = st.slider(
            "Test Size (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5
        ) / 100
    
    if st.button("🚀 Run Simple Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running backtest..."):
            try:
                backtester = PortfolioBacktester(
                    assets=assets,
                    data=data,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost
                )
                
                # Split train/test
                train_data, test_data = backtester.train_test_split(train_size=1-test_size)
                
                # Run backtest sur test data
                test_start = test_data.index[0].strftime('%Y-%m-%d')
                test_end = test_data.index[-1].strftime('%Y-%m-%d')
                
                results = backtester.simple_backtest(weights, start_date=test_start, end_date=test_end)
                
                # Afficher résultats
                display_simple_backtest_results(results, assets, weights, portfolio_name, initial_capital)
                
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())


def display_simple_backtest_results(results, assets, weights, portfolio_name, initial_capital):
    """Affiche les résultats du simple backtest"""
    st.markdown("---")
    st.markdown("### 📊 Backtest Results")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{results['total_return']:.2%}",
            delta=f"${results['final_capital'] - initial_capital:,.0f}"
        )
    
    with col2:
        sharpe_color = "🟢" if results['sharpe_ratio'] > 1 else "🟡" if results['sharpe_ratio'] > 0 else "🔴"
        st.metric(
            f"Sharpe Ratio {sharpe_color}",
            f"{results['sharpe_ratio']:.3f}"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown']:.2%}",
            delta=None,
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Sortino Ratio",
            f"{results['sortino_ratio']:.3f}"
        )
    
    # Graphique de performance
    st.markdown("### 📈 Portfolio Performance")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Portfolio Value', 'Daily Returns Distribution'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.12
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=list(range(len(results['portfolio_values']))),
            y=results['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#6366F1', width=3),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ),
        row=1, col=1
    )
    
    # Ligne du capital initial
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        row=1, col=1
    )
    
    # Returns histogram
    fig.add_trace(
        go.Histogram(
            x=results['portfolio_returns'],
            nbinsx=50,
            name='Daily Returns',
            marker=dict(color='#8B5CF6'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Daily Return", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Métriques détaillées
    with st.expander("📋 Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Performance Metrics:**")
            st.write(f"• Annual Return: {results['annual_return']:.2%}")
            st.write(f"• Volatility: {results['volatility']:.2%}")
            st.write(f"• Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            st.write(f"• Sortino Ratio: {results['sortino_ratio']:.3f}")
            st.write(f"• Calmar Ratio: {results['calmar_ratio']:.3f}")
        
        with col2:
            st.markdown("**Risk Metrics:**")
            st.write(f"• Max Drawdown: {results['max_drawdown']:.2%}")
            st.write(f"• Final Capital: ${results['final_capital']:,.2f}")
            st.write(f"• Total Return: {results['total_return']:.2%}")
            st.write(f"• Profit/Loss: ${results['final_capital'] - initial_capital:+,.2f}")
    
    # Option de sauvegarde
    st.markdown("---")
    st.markdown("### 💾 Save Backtest Results")
    
    with st.form("save_backtest_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            save_name = st.text_input(
                "Result Name",
                value=f"{portfolio_name or 'Backtest'} - {datetime.now().strftime('%Y%m%d')}",
                placeholder="My Backtest"
            )
        
        with col2:
            save_to_db = st.checkbox("Save as new portfolio in database", value=False)
        
        submit = st.form_submit_button("Save Results", use_container_width=True)
        
        if submit and save_name:
            try:
                full_data = st.session_state.backtest_data
                portfolio = Portfolio(assets, full_data)
                portfolio.set_weights(list(weights))
                
                if save_to_db:
                    save_portfolio(
                        user_id,
                        portfolio,
                        save_name,
                        model="backtest",
                        amount=initial_capital,
                        backtest_results={
                            'sharpe': results['sharpe_ratio'],
                            'return': results['total_return'],
                            'max_dd': results['max_drawdown']
                        }
                    )
                    st.success(f"✅ Backtest saved as portfolio: {save_name}")
                else:
                    filename = f"{save_name.replace(' ', '_')}.csv"
                    save_backtest_results_to_csv(results, filename)
                    st.success(f"✅ Results saved to {filename}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Error saving: {e}")


def render_walkforward_backtest(assets, weights, data, portfolio_name=None):
    """Walk-forward backtest interface"""
    st.markdown("#### 🔄 Walk-Forward Optimization Configuration")
    
    st.info("""
    Walk-forward optimization simule un rebalancement périodique du portfolio.
    À chaque période, le modèle est ré-entraîné sur les données récentes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Configuration:**")
        train_window = st.number_input(
            "Training Window (days)",
            min_value=60,
            max_value=504,
            value=252,
            step=21,
            help="252 days = 1 year of trading"
        )
        
        test_window = st.number_input(
            "Test Window (days)",
            min_value=21,
            max_value=126,
            value=63,
            step=21,
            help="63 days = 3 months"
        )
    
    with col2:
        st.markdown("**Portfolio Configuration:**")
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            value=10000.0,
            step=1000.0
        )
        
        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        ) / 100
    
    step_size = st.slider(
        "Rebalance Frequency (days)",
        min_value=7,
        max_value=63,
        value=21,
        step=7,
        help="21 days = monthly rebalancing"
    )
    
    strategy_choice = st.selectbox(
        "Strategy for Rebalancing",
        ["Use Fixed Weights", "PCA", "ICA", "RL (Actor-Critic)"],
        help="Fixed Weights uses the current weights, others retrain each period"
    )
    
    if st.button("🚀 Run Walk-Forward Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running walk-forward optimization... This may take a few minutes."):
            try:
                backtester = PortfolioBacktester(
                    assets=assets,
                    data=data,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost
                )
                
                if strategy_choice == "Use Fixed Weights":
                    strategy_func = lambda df: (weights, {})
                    strategy_params = {}
                elif strategy_choice == "PCA":
                    from pages.ml_portfolio_builder import pca_portfolio
                    strategy_func = pca_portfolio
                    strategy_params = {'n_components': 2}
                elif strategy_choice == "ICA":
                    from pages.ml_portfolio_builder import ica_portfolio
                    strategy_func = ica_portfolio
                    strategy_params = {'n_components': 3}
                elif strategy_choice == "RL (Actor-Critic)":
                    from pages.rl_portfolio_builder import get_rl_portfolio_weights
                    strategy_func = lambda df: get_rl_portfolio_weights(df, agent_type='actor_critic', n_episodes=20)
                    strategy_params = {}
                
                results = backtester.walk_forward_optimization(
                    strategy_func=strategy_func,
                    train_window=train_window,
                    test_window=test_window,
                    step_size=step_size,
                    **strategy_params
                )
                
                display_walkforward_results(results, assets, portfolio_name, initial_capital)
                
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())


def display_walkforward_results(results, assets, portfolio_name, initial_capital):
    """Affiche les résultats du walk-forward backtest"""
    st.markdown("---")
    st.markdown("### 📊 Walk-Forward Results")
    
    summary = results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{summary['total_return']:.2%}",
            delta=f"${summary['final_capital'] - initial_capital:,.0f}"
        )
    
    with col2:
        st.metric(
            "Avg Sharpe",
            f"{summary['avg_sharpe']:.3f}"
        )
    
    with col3:
        st.metric(
            "Total Costs",
            f"${summary['total_rebalance_costs']:,.0f}",
            delta=f"{summary['total_rebalance_costs']/initial_capital:.1%} of capital"
        )
    
    with col4:
        st.metric(
            "Rebalances",
            f"{summary['n_periods']}"
        )
    
    st.markdown("### 📈 Performance Evolution")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Period Returns',
            'Rolling Sharpe Ratio'
        ),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['periods'],
            y=results['portfolio_values'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#6366F1', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    colors = ['green' if r > 0 else 'red' for r in results['returns']]
    fig.add_trace(
        go.Bar(
            x=results['periods'],
            y=results['returns'],
            name='Period Return',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['periods'],
            y=results['sharpe_ratios'],
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='#8B5CF6', width=2),
            marker=dict(size=5)
        ),
        row=3, col=1
    )
    
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Sharpe = 1",
        row=3, col=1
    )
    
    fig.update_layout(
        height=900,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Period", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Period", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)
    fig.update_xaxes(title_text="Period", row=3, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🎯 Weights Evolution Over Time")
    
    weights_history = np.array(results['weights_history'])
    
    fig_weights = go.Figure()
    
    for i, asset in enumerate(assets):
        fig_weights.add_trace(go.Scatter(
            x=results['periods'],
            y=weights_history[:, i],
            mode='lines',
            name=asset,
            stackgroup='one',
            fillcolor=f'rgba({100+i*30}, {150+i*20}, {200+i*10}, 0.6)'
        ))
    
    fig_weights.update_layout(
        title="Portfolio Weights Evolution",
        xaxis_title="Period",
        yaxis_title="Weight",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_weights, use_container_width=True)
    
    with st.expander("📋 Period-by-Period Details"):
        details_df = pd.DataFrame({
            'Period': results['periods'],
            'Return': [f"{r:.2%}" for r in results['returns']],
            'Portfolio Value': [f"${v:,.2f}" for v in results['portfolio_values'][1:]],
            'Sharpe Ratio': [f"{s:.3f}" for s in results['sharpe_ratios']],
            'Rebalance Cost': [f"${c:,.2f}" for c in results['rebalance_costs']]
        })
        
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 💾 Save Walk-Forward Results")
    
    with st.form("save_walkforward_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            save_name = st.text_input(
                "Result Name",
                value=f"{portfolio_name or 'WalkForward'} - {datetime.now().strftime('%Y%m%d')}",
                placeholder="My Walk-Forward Results"
            )
        
        with col2:
            save_final_weights = st.checkbox(
                "Save final weights as portfolio",
                value=True,
                help="Save the last rebalanced weights as a new portfolio"
            )
        
        submit = st.form_submit_button("Save Results", use_container_width=True)
        
        if submit and save_name:
            try:
                if save_final_weights and results['weights_history']:
                    final_weights = results['weights_history'][-1]
                    full_data = st.session_state.backtest_data
                    portfolio = Portfolio(assets, full_data)
                    portfolio.set_weights(list(final_weights))
                    
                    save_portfolio(
                        user_id,
                        portfolio,
                        save_name,
                        model="walk_forward",
                        amount=summary['final_capital'],
                        backtest_results={
                            'avg_sharpe': summary['avg_sharpe'],
                            'total_return': summary['total_return'],
                            'n_periods': summary['n_periods']
                        }
                    )
                    st.success(f"✅ Final portfolio saved: {save_name}")
                
                filename = f"{save_name.replace(' ', '_')}_walkforward.csv"
                save_backtest_results_to_csv(results, filename)
                st.success(f"✅ Detailed results saved to {filename}")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Error saving: {e}")



# ============================================================================
# EXPORT
# ============================================================================

def render_export_results():
    """Export des résultats"""
    st.markdown("####📤 Export Results")
    
    st.markdown("""
    Exportez vos portfolios aux formats CSV et JSON.
    """)
    
    try:
        portfolios = list(get_portfolios(user_id=user_id))
        
        if not portfolios:
            st.info("No portfolios to export. Create one first!")
            return
        
        portfolio_names = [p['name'] for p in portfolios]
        selected = st.selectbox("Select Portfolio to Export", portfolio_names)
        
        if selected:
            portfolio = get_single_portfolio(user_id,selected)
            
            st.markdown("---")
            st.markdown("### 📊 Portfolio Preview")
            holdings=portfolio.get('holdings',[])
            
            preview_df = pd.DataFrame({
                'Asset': [h['symbol'] for h in holdings],
                'Weight': [f"{h['weight']:.2%}" for h in holdings],
                'Quantity':  [f"{h['quantity']:.3f}" for h in holdings]
            })
            
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export CSV", use_container_width=True):
                    csv_data = export_to_csv(portfolio)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"{selected.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📋 Export JSON", use_container_width=True):
                    import json
                    json_data = json.dumps(portfolio, indent=2, default=str)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=f"{selected.replace(' ', '_')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col3:
                st.button("📄 Export PDF", use_container_width=True, disabled=True)
                st.caption("🚧 Coming soon!")
    
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")


def export_to_csv(portfolio):
    """Convertit un portfolio en CSV"""
    df = pd.DataFrame({
        'Asset': portfolio['assets'],
        'Weight': portfolio['weights'],
        'Quantity': portfolio.get('quantities', [0] * len(portfolio['assets']))
    })
    return df.to_csv(index=False)
