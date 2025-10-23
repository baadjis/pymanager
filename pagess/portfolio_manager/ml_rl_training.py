# pages/ml_rl_training_lab.py
"""
ML/RL Training Lab - Interface complète pour entraîner et tester des modèles
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from portfolio import Portfolio, get_log_returns
from dataprovider import yahoo
from uiconfig import get_theme_colors


def render_ml_rl_training():
    """ML/RL Training Lab - Interface principale"""
    st.markdown("#### 🤖 ML/RL Training Lab")
    
    st.info("""
    **Training Lab** vous permet d'entraîner et optimiser des modèles ML/RL avec :
    - Hyperparameter tuning
    - Cross-validation
    - Training visualization
    - Performance comparison
    """)
    
    # Sous-onglets
    training_tab1, training_tab2, training_tab3 = st.tabs([
        "🔬 PCA/ICA Training",
        "🎮 RL Training", 
        "📊 Model Comparison"
    ])
    
    with training_tab1:
        render_pca_ica_training()
    
    with training_tab2:
        render_rl_training()
    
    with training_tab3:
        render_training_comparison()


# ============================================================================
# PCA/ICA TRAINING
# ============================================================================

def render_pca_ica_training():
    """Interface d'entraînement PCA/ICA"""
    st.markdown("### 🔬 PCA/ICA Training & Analysis")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_input(
            "Stock Tickers",
            placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA",
            key="pca_ica_tickers"
        )
    
    with col2:
        period = st.selectbox(
            "Training Period",
            ["1y", "2y", "3y", "5y"],
            index=2,
            key="pca_ica_period"
        )
    
    if not tickers:
        st.info("👆 Enter tickers to start training")
        return
    
    assets = [t.strip().upper() for t in tickers.split(",")]
    
    st.markdown("---")
    st.markdown("### ⚙️ Training Configuration")
    
    method = st.radio(
        "Select Method",
        ["PCA", "ICA", "Both (Comparison)"],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if method in ["PCA", "Both (Comparison)"]:
            n_components_pca = st.slider(
                "PCA Components",
                min_value=2,
                max_value=min(10, len(assets)),
                value=min(3, len(assets)),
                key="pca_n_comp"
            )
    
    with col2:
        if method in ["ICA", "Both (Comparison)"]:
            n_components_ica = st.slider(
                "ICA Components", 
                min_value=2,
                max_value=min(10, len(assets)),
                value=min(4, len(assets)),
                key="ica_n_comp"
            )
    
    with col3:
        use_kernel_pca = st.checkbox(
            "Use Kernel PCA",
            value=False,
            help="Non-linear PCA (slower)"
        )
    
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        with st.spinner("Training models..."):
            try:
                # Charger données
                data = yahoo.retrieve_data(tuple(assets), period=period)
                returns_data = get_log_returns(data)
                
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
                
                results = {}
                
                # Train PCA
                if method in ["PCA", "Both (Comparison)"]:
                    from .ml_portfolio import pca_portfolio
                    
                    st.markdown("#### 📊 Training PCA...")
                    weights_pca, info_pca = pca_portfolio(
                        returns_df,
                        n_components=n_components_pca,
                        use_kernel=use_kernel_pca
                    )
                    results['PCA'] = {
                        'weights': weights_pca,
                        'info': info_pca
                    }
                    st.success("✅ PCA trained!")
                
                # Train ICA
                if method in ["ICA", "Both (Comparison)"]:
                    from .ml_portfolio import ica_portfolio
                    
                    st.markdown("#### 📊 Training ICA...")
                    weights_ica, info_ica = ica_portfolio(
                        returns_df,
                        n_components=n_components_ica
                    )
                    results['ICA'] = {
                        'weights': weights_ica,
                        'info': info_ica
                    }
                    st.success("✅ ICA trained!")
                
                # Afficher résultats
                display_pca_ica_results(results, assets, returns_df, data)
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())


def display_pca_ica_results(results, assets, returns_df, data):
    """Affiche les résultats d'entraînement PCA/ICA"""
    st.markdown("---")
    st.markdown("## 📊 Training Results")
    
    # Métriques de comparaison
    if len(results) > 1:
        st.markdown("### 🎯 Model Comparison")
        
        comparison_data = []
        for name, result in results.items():
            # Calculer portfolio metrics
            portfolio = Portfolio(assets, data)
            portfolio.set_weights(list(result['weights']))
            
            comparison_data.append({
                'Model': name,
                'Sharpe': f"{portfolio.sharp_ratio:.3f}",
                'Return': f"{portfolio.expected_return:.2%}",
                'Volatility': f"{portfolio.stdev:.2%}",
                'Max DD': f"{portfolio.max_drawdown:.2%}"
            })
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Détails par modèle
    for model_name, result in results.items():
        with st.expander(f"📋 {model_name} Details", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Weights:**")
                weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [f"{w:.2%}" for w in result['weights']]
                })
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=assets,
                    values=result['weights'],
                    hole=0.4
                )])
                fig.update_layout(
                    title=f"{model_name} Allocation",
                    height=300,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Informations spécifiques
            if model_name == 'PCA':
                info = result['info']
                st.markdown("**PCA Metrics:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Variance", f"{info['total_variance_explained']:.1%}")
                with col2:
                    st.metric("Components", info['n_components'])
                with col3:
                    st.metric("Method", info['method'])
                
                # Variance par composante
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(
                    x=[f"PC{i+1}" for i in range(len(info['explained_variance_ratio']))],
                    y=info['explained_variance_ratio'],
                    name='Individual'
                ))
                fig_var.add_trace(go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(info['explained_variance_ratio']))],
                    y=np.cumsum(info['explained_variance_ratio']),
                    mode='lines+markers',
                    name='Cumulative',
                    yaxis='y2'
                ))
                fig_var.update_layout(
                    title="Explained Variance",
                    yaxis=dict(title="Individual"),
                    yaxis2=dict(title="Cumulative", overlaying='y', side='right'),
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig_var, use_container_width=True)
            
            elif model_name == 'ICA':
                info = result['info']
                st.markdown("**ICA Metrics:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Kurtosis", f"{info['mean_source_kurtosis']:.3f}")
                with col2:
                    st.metric("Components", info['n_components'])
                
                # Source importance
                if 'source_importance' in info:
                    fig_sources = go.Figure(data=[go.Bar(
                        x=[f"IC{i+1}" for i in range(len(info['source_importance']))],
                        y=info['source_importance']
                    )])
                    fig_sources.update_layout(
                        title="Source Importance",
                        yaxis_title="Importance",
                        template='plotly_dark',
                        height=300
                    )
                    st.plotly_chart(fig_sources, use_container_width=True)


# ============================================================================
# RL TRAINING
# ============================================================================

def render_rl_training():
    """Interface d'entraînement RL"""
    st.markdown("### 🎮 Reinforcement Learning Training")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_input(
            "Stock Tickers",
            placeholder="AAPL, MSFT, GOOGL, AMZN",
            key="rl_tickers"
        )
    
    with col2:
        period = st.selectbox(
            "Training Period",
            ["6mo", "1y", "2y", "3y"],
            index=2,
            key="rl_period"
        )
    
    if not tickers:
        st.info("👆 Enter tickers to start RL training")
        return
    
    assets = [t.strip().upper() for t in tickers.split(",")]
    
    st.markdown("---")
    st.markdown("### ⚙️ RL Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agent_type = st.selectbox(
            "Agent Type",
            ["Actor-Critic", "REINFORCE"],
            key="rl_agent"
        )
    
    with col2:
        n_episodes = st.slider(
            "Training Episodes",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            help="Plus = meilleur mais plus lent"
        )
    
    with col3:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
    
    # Options avancées
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            gamma = st.slider("Discount Factor (γ)", 0.9, 0.99, 0.95, 0.01)
            window_size = st.slider("Observation Window", 10, 50, 20, 5)
        
        with col2:
            transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
            initial_capital = st.number_input("Initial Capital ($)", 1000.0, 100000.0, 10000.0, 1000.0)
    
    if st.button("🚀 Start RL Training", use_container_width=True, type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Charger données
            status_text.text("📊 Loading market data...")
            data = yahoo.retrieve_data(tuple(assets), period=period)
            returns_data = get_log_returns(data)
            
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
            
            progress_bar.progress(20)
            
            # Entraîner l'agent
            status_text.text(f"🤖 Training {agent_type} agent for {n_episodes} episodes...")
            
            from .rl_portfolio_simple import train_rl_portfolio, evaluate_rl_agent
            
            agent, training_history, best_weights = train_rl_portfolio(
                returns_data=returns_df,
                agent_type='actor_critic' if agent_type == "Actor-Critic" else 'reinforce',
                n_episodes=n_episodes,
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                gamma=gamma,
                verbose=False
            )
            
            progress_bar.progress(80)
            
            # Évaluer
            status_text.text("📈 Evaluating agent...")
            metrics = evaluate_rl_agent(agent, returns_df, initial_capital, transaction_cost)
            
            progress_bar.progress(100)
            status_text.text("✅ Training complete!")
            
            # Afficher résultats
            display_rl_training_results(
                agent_type,
                training_history,
                metrics,
                best_weights,
                assets,
                n_episodes
            )
            
        except Exception as e:
            st.error(f"RL Training failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        finally:
            progress_bar.empty()
            status_text.empty()


def display_rl_training_results(agent_type, history, metrics, weights, assets, n_episodes):
    """Affiche les résultats d'entraînement RL"""
    st.markdown("---")
    st.markdown("## 🎮 RL Training Results")
    
    # Métriques finales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Sharpe", f"{metrics['sharpe_ratio']:.3f}")
    
    with col2:
        st.metric("Total Return", f"{metrics['total_return']:.2%}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    
    with col4:
        st.metric("Final Capital", f"${metrics['final_capital']:,.0f}")
    
    # Training curves
    st.markdown("### 📈 Training Progress")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Episode Returns',
            'Sharpe Ratio Evolution',
            'Final Capital',
            'Cumulative Portfolio Value'
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    episodes = list(range(1, n_episodes + 1))
    
    # Episode returns
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=history['episode_returns'],
            mode='lines+markers',
            name='Return',
            line=dict(color='#6366F1')
        ),
        row=1, col=1
    )
    
    # Sharpe ratios
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=history['sharpe_ratios'],
            mode='lines+markers',
            name='Sharpe',
            line=dict(color='#8B5CF6')
        ),
        row=1, col=2
    )
    
    # Final capitals
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=history['final_capitals'],
            mode='lines+markers',
            name='Capital',
            line=dict(color='#EC4899')
        ),
        row=2, col=1
    )
    
    # Portfolio values (dernière évaluation)
    fig.add_trace(
        go.Scatter(
            y=metrics['portfolio_values'],
            mode='lines',
            name='Value',
            line=dict(color='#10B981', width=3),
            fill='tonexty'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Poids finaux
    st.markdown("### 🎯 Learned Portfolio Weights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weights_df = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.2%}" for w in weights]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_pie = go.Figure(data=[go.Pie(
            labels=assets,
            values=weights,
            hole=0.4
        )])
        fig_pie.update_layout(
            title="Final Allocation",
            height=300,
            template='plotly_dark'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Métriques détaillées
    with st.expander("📋 Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Performance:**")
            st.write(f"• Total Return: {metrics['total_return']:.2%}")
            st.write(f"• Annual Return: {metrics['annual_return']:.2%}")
            st.write(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            st.write(f"• Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        
        with col2:
            st.markdown("**Risk:**")
            st.write(f"• Volatility: {metrics['volatility']:.2%}")
            st.write(f"• Max Drawdown: {metrics['max_drawdown']:.2%}")
            st.write(f"• Final Capital: ${metrics['final_capital']:,.2f}")


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def render_training_comparison():
    """Comparaison des modèles entraînés"""
    st.markdown("### 📊 Model Comparison & Analysis")
    
    st.info("""
    **Model Comparison** vous permet de comparer plusieurs modèles ML/RL côte à côte
    et d'analyser leurs performances sur les mêmes données.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_input(
            "Stock Tickers",
            placeholder="AAPL, MSFT, GOOGL, AMZN",
            key="comparison_tickers_ml"
        )
    
    with col2:
        period = st.selectbox(
            "Period",
            ["1y", "2y", "3y", "5y"],
            index=2,
            key="comparison_period_ml"
        )
    
    if not tickers:
        st.info("👆 Enter tickers to start comparison")
        return
    
    assets = [t.strip().upper() for t in tickers.split(",")]
    
    st.markdown("---")
    st.markdown("### ⚙️ Select Models to Compare")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Dimensionality Reduction:**")
        use_pca = st.checkbox("PCA", value=True, key="comp_pca")
        if use_pca:
            n_comp_pca = st.slider("Components", 2, min(10, len(assets)), min(3, len(assets)), key="pca_comp_comp")
        
        use_ica = st.checkbox("ICA", value=True, key="comp_ica")
        if use_ica:
            n_comp_ica = st.slider("Components", 2, min(10, len(assets)), min(4, len(assets)), key="ica_comp_comp")
    
    with col2:
        st.markdown("**Reinforcement Learning:**")
        use_rl_ac = st.checkbox("RL Actor-Critic", value=True, key="comp_rl_ac")
        if use_rl_ac:
            episodes_ac = st.slider("Episodes", 20, 100, 30, 10, key="ac_episodes_comp")
        
        use_rl_reinforce = st.checkbox("RL REINFORCE", value=False, key="comp_rl_reinforce")
        if use_rl_reinforce:
            episodes_rf = st.slider("Episodes", 20, 100, 30, 10, key="rf_episodes_comp")
    
    with col3:
        st.markdown("**Baseline:**")
        use_equal = st.checkbox("Equal Weight", value=True, key="comp_equal")
        use_markowitz = st.checkbox("Markowitz", value=True, key="comp_markowitz")
    
    # Options avancées
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            test_split = st.slider("Test Split (%)", 10, 50, 30, 5, help="% des données pour le test")
            n_bootstrap = st.number_input("Bootstrap Samples", 10, 1000, 100, 50, help="Pour intervalles de confiance")
        
        with col2:
            risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100
            show_statistics = st.checkbox("Show Statistical Tests", value=True)
    
    if st.button("🚀 Run Comprehensive Comparison", use_container_width=True, type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Charger données
            status_text.text("📊 Loading market data...")
            data = yahoo.retrieve_data(tuple(assets), period=period)
            returns_data = get_log_returns(data)
            
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
            
            progress_bar.progress(10)
            
            # Split train/test
            split_idx = int(len(returns_df) * (1 - test_split/100))
            train_df = returns_df.iloc[:split_idx]
            test_df = returns_df.iloc[split_idx:]
            
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            status_text.text(f"📊 Train: {len(train_df)} periods | Test: {len(test_df)} periods")
            
            # Dictionnaire pour stocker les résultats
            results = {}
            total_models = sum([use_pca, use_ica, use_rl_ac, use_rl_reinforce, use_equal, use_markowitz])
            current_model = 0
            
            # Entraîner les modèles
            if use_equal:
                status_text.text(f"Training Equal Weight... ({current_model+1}/{total_models})")
                weights = np.ones(len(assets)) / len(assets)
                results['Equal Weight'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            if use_markowitz:
                status_text.text(f"Training Markowitz... ({current_model+1}/{total_models})")
                from factory import create_portfolio_by_name
                portfolio = create_portfolio_by_name(assets, "sharp", train_data)
                weights = np.array(portfolio.weights)
                results['Markowitz'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            if use_pca:
                status_text.text(f"Training PCA... ({current_model+1}/{total_models})")
                from .ml_portfolio import pca_portfolio
                weights, _ = pca_portfolio(train_df, n_components=n_comp_pca)
                results[f'PCA ({n_comp_pca})'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            if use_ica:
                status_text.text(f"Training ICA... ({current_model+1}/{total_models})")
                from .ml_portfolio import ica_portfolio
                weights, _ = ica_portfolio(train_df, n_components=n_comp_ica)
                results[f'ICA ({n_comp_ica})'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            if use_rl_ac:
                status_text.text(f"Training RL Actor-Critic... ({current_model+1}/{total_models})")
                from .rl_portfolio_simple import train_rl_portfolio
                agent, history, weights = train_rl_portfolio(
                    train_df, 
                    agent_type='actor_critic',
                    n_episodes=episodes_ac,
                    verbose=False
                )
                results[f'RL AC ({episodes_ac})'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            if use_rl_reinforce:
                status_text.text(f"Training RL REINFORCE... ({current_model+1}/{total_models})")
                from .rl_portfolio_simple import train_rl_portfolio
                agent, history, weights = train_rl_portfolio(
                    train_df,
                    agent_type='reinforce',
                    n_episodes=episodes_rf,
                    verbose=False
                )
                results[f'RL REINFORCE ({episodes_rf})'] = evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate)
                current_model += 1
                progress_bar.progress(10 + int(80 * current_model / total_models))
            
            progress_bar.progress(90)
            status_text.text("📊 Computing statistics...")
            
            # Bootstrap pour intervalles de confiance
            if show_statistics:
                results = add_bootstrap_statistics(results, test_df, n_bootstrap)
            
            progress_bar.progress(100)
            status_text.text("✅ Comparison complete!")
            
            # Afficher résultats
            display_comprehensive_comparison(results, assets, show_statistics, test_split)
            
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        finally:
            progress_bar.empty()
            status_text.empty()


def evaluate_model(weights, train_data, test_data, assets, returns_df, risk_free_rate):
    """Évalue un modèle sur train et test sets"""
    results = {}
    
    # Train metrics
    train_portfolio = Portfolio(assets, train_data)
    train_portfolio.set_weights(list(weights))
    
    results['train'] = {
        'sharpe': train_portfolio.sharp_ratio,
        'return': train_portfolio.expected_return,
        'volatility': train_portfolio.stdev,
        'max_dd': train_portfolio.max_drawdown,
        'sortino': train_portfolio.sortino_ratio(),
        'calmar': train_portfolio.calmar_ratio()
    }
    
    # Test metrics
    test_portfolio = Portfolio(assets, test_data)
    test_portfolio.set_weights(list(weights))
    
    results['test'] = {
        'sharpe': test_portfolio.sharp_ratio,
        'return': test_portfolio.expected_return,
        'volatility': test_portfolio.stdev,
        'max_dd': test_portfolio.max_drawdown,
        'sortino': test_portfolio.sortino_ratio(),
        'calmar': test_portfolio.calmar_ratio()
    }
    
    results['weights'] = weights
    
    return results


def add_bootstrap_statistics(results, test_df, n_bootstrap):
    """Ajoute des intervalles de confiance par bootstrap"""
    for model_name, result in results.items():
        weights = result['weights']
        
        # Bootstrap des returns
        bootstrap_sharpes = []
        bootstrap_returns = []
        
        for _ in range(n_bootstrap):
            # Échantillonner avec remplacement
            sample_idx = np.random.choice(len(test_df), len(test_df), replace=True)
            sample_df = test_df.iloc[sample_idx]
            
            # Calculer rendement du portfolio
            portfolio_returns = (sample_df.values * weights).sum(axis=1)
            
            # Sharpe
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            sharpe = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            
            bootstrap_sharpes.append(sharpe)
            bootstrap_returns.append(mean_return * 252)
        
        # Intervalles de confiance (95%)
        result['test']['sharpe_ci'] = (
            np.percentile(bootstrap_sharpes, 2.5),
            np.percentile(bootstrap_sharpes, 97.5)
        )
        result['test']['return_ci'] = (
            np.percentile(bootstrap_returns, 2.5),
            np.percentile(bootstrap_returns, 97.5)
        )
    
    return results


def display_comprehensive_comparison(results, assets, show_statistics, test_split):
    """Affiche la comparaison complète des modèles"""
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Model Comparison")
    
    # Tableau de comparaison
    st.markdown("### 📋 Performance Metrics")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Train Sharpe': result['train']['sharpe'],
            'Test Sharpe': result['test']['sharpe'],
            'Test Return': result['test']['return'],
            'Test Vol': result['test']['volatility'],
            'Test Max DD': result['test']['max_dd'],
            'Overfitting': result['train']['sharpe'] - result['test']['sharpe']
        })
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Styling
    def highlight_best(s):
        if s.name in ['Train Sharpe', 'Test Sharpe', 'Test Return']:
            is_max = s == s.max()
            return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_max]
        elif s.name in ['Test Vol', 'Test Max DD', 'Overfitting']:
            is_min = s == s.min()
            return ['background-color: rgba(16, 185, 129, 0.2)' if v else '' for v in is_min]
        return ['' for _ in s]
    
    styled_df = df_comp.style\
        .format({
            'Train Sharpe': '{:.3f}',
            'Test Sharpe': '{:.3f}',
            'Test Return': '{:.2%}',
            'Test Vol': '{:.2%}',
            'Test Max DD': '{:.2%}',
            'Overfitting': '{:+.3f}'
        })\
        .apply(highlight_best, subset=['Train Sharpe', 'Test Sharpe', 'Test Return', 'Test Vol', 'Test Max DD', 'Overfitting'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Winner
    best_model = max(results.items(), key=lambda x: x[1]['test']['sharpe'])
    st.success(f"🏆 **Best Model (Test Sharpe):** {best_model[0]} ({best_model[1]['test']['sharpe']:.3f})")
    
    # Graphiques
    st.markdown("---")
    st.markdown("### 📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Train vs Test Sharpe
        fig1 = go.Figure()
        
        models = list(results.keys())
        train_sharpes = [results[m]['train']['sharpe'] for m in models]
        test_sharpes = [results[m]['test']['sharpe'] for m in models]
        
        fig1.add_trace(go.Bar(
            name='Train',
            x=models,
            y=train_sharpes,
            marker_color='#6366F1',
            text=[f"{x:.3f}" for x in train_sharpes],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Bar(
            name='Test',
            x=models,
            y=test_sharpes,
            marker_color='#10B981',
            text=[f"{x:.3f}" for x in test_sharpes],
            textposition='outside'
        ))
        
        fig1.update_layout(
            title="Train vs Test Sharpe Ratio",
            yaxis_title="Sharpe Ratio",
            template='plotly_dark',
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Risk-Return scatter (Test)
        fig2 = go.Figure()
        
        for name, result in results.items():
            fig2.add_trace(go.Scatter(
                x=[result['test']['volatility']],
                y=[result['test']['return']],
                mode='markers+text',
                name=name,
                text=[name],
                textposition='top center',
                marker=dict(size=15)
            ))
        
        fig2.update_layout(
            title="Risk-Return Profile (Test Set)",
            xaxis_title="Volatility",
            yaxis_title="Return",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Weights comparison
    st.markdown("---")
    st.markdown("### 🎯 Learned Weights Comparison")
    
    weights_data = {name: result['weights'] for name, result in results.items()}
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
    
    # Statistical tests
    if show_statistics:
        render_statistical_tests(results, test_split)
