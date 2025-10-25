"""
Intégration du Reinforcement Learning dans l'interface Streamlit
À ajouter dans portfolio_manager.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from portfolio import Portfolio, get_log_returns
from .rl_portfolio import (
    get_rl_portfolio_weights,
    train_rl_portfolio,
    evaluate_rl_agent
)
user_id = st.session_state.user_id


def build_rl_portfolio(assets, data, theme):
    """
    Build RL-based portfolio
    Intégration dans portfolio_manager.py
    """
    st.markdown("#### 🤖 Reinforcement Learning Portfolio")
    st.info("💡 L'agent RL apprend à optimiser l'allocation en interagissant avec le marché")
    
    # Explication rapide
    with st.expander("📚 Comment ça marche ?"):
        st.markdown("""
        **Reinforcement Learning pour Portfolios:**
        
        1. **Agent** - Prend des décisions d'allocation
        2. **Environment** - Simule le marché
        3. **Reward** - Basé sur les rendements ajustés du risque
        4. **Learning** - L'agent améliore sa stratégie au fil du temps
        
        **Avantages:**
        - Adaptation dynamique aux conditions de marché
        - Optimisation multi-objectifs (rendement + risque)
        - Prend en compte les coûts de transaction
        - Pas d'hypothèses sur la distribution des rendements
        """)
    
    # Préparation des données
    try:
        returns_data = get_log_returns(data)
        
        # Si multi-actifs, créer DataFrame approprié
        if len(assets) > 1:
            if isinstance(data.columns, pd.MultiIndex):
                returns_df = pd.DataFrame()
                for asset in assets:
                    try:
                        prices = data[('Adj Close', asset)]
                        returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
                    except:
                        st.error(f"Could not process {asset}")
                        return
            else:
                if len(assets) == 1:
                    returns_df = pd.DataFrame({assets[0]: returns_data})
                else:
                    returns_df = returns_data
        else:
            returns_df = pd.DataFrame({assets[0]: returns_data})
        
        # Vérifier la qualité des données
        if returns_df.empty or len(returns_df) < 100:
            st.error("❌ Insufficient data for RL training (need at least 100 data points)")
            st.info(f"Current data points: {len(returns_df)}")
            return
        
        st.success(f"✅ Using {len(returns_df)} data points for training")
        
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return
    
    # Configuration de l'agent
    st.markdown("---")
    st.markdown("### ⚙️ Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type = st.selectbox(
            "Algorithm",
            ["actor_critic", "reinforce"],
            help="""
            - **Actor-Critic**: Plus stable, converge plus vite (recommandé)
            - **REINFORCE**: Plus simple, peut prendre plus de temps
            """
        )
        
        n_episodes = st.slider(
            "Training Episodes",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Plus d'épisodes = meilleur apprentissage mais plus long"
        )
    
    with col2:
        transaction_cost = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Coût de transaction en pourcentage (0.1% = 10 basis points)"
        ) / 100
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0
        )
    
    # Options avancées
    with st.expander("🔧 Advanced Options"):
        gamma = st.slider(
            "Discount Factor (γ)",
            min_value=0.9,
            max_value=0.999,
            value=0.99,
            step=0.001,
            help="Importance des récompenses futures (0.99 recommandé)"
        )
        
        window_size = st.slider(
            "Lookback Window",
            min_value=5,
            max_value=50,
            value=20,
            help="Nombre de jours passés utilisés pour la décision"
        )
    
    # Bouton d'entraînement
    if st.button("🚀 Train RL Agent", use_container_width=True, type="primary"):
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing RL environment...")
        progress_bar.progress(10)
        
        try:
            status_text.text(f"🤖 Training {agent_type.upper()} agent...")
            progress_bar.progress(30)
            
            # Entraînement
            with st.spinner(f"Training for {n_episodes} episodes... This may take a minute."):
                weights, info = get_rl_portfolio_weights(
                    returns_data=returns_df,
                    agent_type=agent_type,
                    n_episodes=n_episodes,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    gamma=gamma
                )
            
            progress_bar.progress(80)
            status_text.text("✅ Training completed!")
            
            # Sauvegarder dans session state
            st.session_state.rl_weights = weights
            st.session_state.rl_info = info
            st.session_state.rl_assets = assets
            
            progress_bar.progress(100)
            st.success("🎉 RL Agent trained successfully!")
            
            # Afficher les résultats
            display_rl_results(weights, info, assets, returns_df, theme, data)
            
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    
    # Si des résultats existent déjà
    elif 'rl_weights' in st.session_state and st.session_state.rl_assets == assets:
        st.info("📊 Displaying previous training results")
        display_rl_results(
            st.session_state.rl_weights,
            st.session_state.rl_info,
            assets,
            returns_df,
            theme,
            data
        )


def display_rl_results(weights, info, assets, returns_df, theme, data):
    """Affiche les résultats de l'entraînement RL"""
    
    st.markdown("---")
    st.markdown("## 📊 Training Results")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sharpe Ratio",
            f"{info['final_sharpe']:.3f}",
            help="Rendement ajusté du risque"
        )
    
    with col2:
        st.metric(
            "Total Return",
            f"{info['final_return']:.2%}",
            help="Rendement total durant l'entraînement"
        )
    
    with col3:
        st.metric(
            "Episodes",
            info['n_episodes'],
            help="Nombre d'épisodes d'entraînement"
        )
    
    with col4:
        st.metric(
            "Algorithm",
            info['method'].split('(')[1].rstrip(')').upper(),
            help="Algorithme RL utilisé"
        )
    
    # Graphiques d'entraînement
    st.markdown("### 📈 Training Progress")
    
    history = info['training_history']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sharpe Ratio Evolution',
            'Episode Returns',
            'Final Capital',
            'Cumulative Reward'
        )
    )
    
    # Sharpe ratio
    fig.add_trace(
        go.Scatter(
            y=history['sharpe_ratios'],
            mode='lines',
            name='Sharpe Ratio',
            line=dict(color='#6366F1', width=2)
        ),
        row=1, col=1
    )
    
    # Returns
    fig.add_trace(
        go.Scatter(
            y=history['episode_returns'],
            mode='lines',
            name='Returns',
            line=dict(color='#10B981', width=2)
        ),
        row=1, col=2
    )
    
    # Capital
    fig.add_trace(
        go.Scatter(
            y=history['final_capitals'],
            mode='lines',
            name='Capital',
            line=dict(color='#F59E0B', width=2)
        ),
        row=2, col=1
    )
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(history['episode_rewards'])
    fig.add_trace(
        go.Scatter(
            y=cumulative_rewards,
            mode='lines',
            name='Cumulative Reward',
            line=dict(color='#8B5CF6', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme['text_primary'])
    )
    
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio weights
    st.markdown("### 🎯 Learned Portfolio Weights")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        weights_df = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.4f}" for w in weights],
            'Percentage': [f"{w*100:.2f}%" for w in weights]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Bar chart des poids
        fig_weights = go.Figure(data=[
            go.Bar(
                x=assets,
                y=weights,
                marker=dict(
                    color=weights,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Weight")
                ),
                text=[f"{w:.2%}" for w in weights],
                textposition='outside'
            )
        ])
        
        fig_weights.update_layout(
            title="Portfolio Weights Distribution",
            xaxis_title="Assets",
            yaxis_title="Weight",
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_weights, use_container_width=True)
    
    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4,
        marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6'])
    )])
    
    fig_pie.update_layout(
        title="Portfolio Composition",
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(color=theme['text_primary'])
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Métriques d'évaluation détaillées
    if 'evaluation_metrics' in info:
        st.markdown("### 📊 Evaluation Metrics")
        
        metrics = info['evaluation_metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annual Return</div>
                <div class="metric-value">{metrics['annual_return']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value">{metrics['volatility']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe_color = "#10B981" if metrics['sharpe_ratio'] > 1 else "#F59E0B"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: {sharpe_color};">{metrics['sharpe_ratio']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: #EF4444;">{metrics['max_drawdown']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Capital</div>
                <div class="metric-value">${metrics['final_capital']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            calmar = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-value">{calmar:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cumulative returns chart
        st.markdown("### 📈 Portfolio Performance")
        
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            y=metrics['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#6366F1', width=3),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        fig_perf.update_layout(
            title="Cumulative Portfolio Value",
            xaxis_title="Time Period",
            yaxis_title="Portfolio Value ($)",
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Returns distribution
        st.markdown("### 📊 Returns Distribution")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=metrics['returns'],
            nbinsx=50,
            name='Returns',
            marker=dict(
                color='#8B5CF6',
                line=dict(color='#6366F1', width=1)
            )
        ))
        
        # Add normal distribution overlay
        mean_ret = np.mean(metrics['returns'])
        std_ret = np.std(metrics['returns'])
        x_range = np.linspace(metrics['returns'].min(), metrics['returns'].max(), 100)
        normal_dist = len(metrics['returns']) * (metrics['returns'].max() - metrics['returns'].min()) / 50 * \
                     (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_ret) / std_ret) ** 2)
        
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#EF4444', width=2, dash='dash')
        ))
        
        fig_hist.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color=theme['text_primary']),
            showlegend=True
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Comparaison avec benchmarks
    st.markdown("### 🎯 Comparison with Benchmarks")
    
    # Equal weight benchmark
    equal_weights = np.ones(len(assets)) / len(assets)
    equal_returns = (returns_df * equal_weights).sum(axis=1)
    equal_sharpe = equal_returns.mean() / equal_returns.std() * np.sqrt(252)
    
    # Mean-variance (Markowitz) benchmark
    try:
        from portfolio import Portfolio
        mv_portfolio = Portfolio(assets, data)
        mv_portfolio = Portfolio.optimize_sharpe_ratio(assets, data)
        mv_sharpe = mv_portfolio.sharp_ratio
    except:
        mv_sharpe = None
    
    comparison_data = {
        'Strategy': ['RL Agent', 'Equal Weight'],
        'Sharpe Ratio': [info['final_sharpe'], equal_sharpe],
        'Improvement': ['-', f"{(info['final_sharpe']/equal_sharpe - 1)*100:+.1f}%"]
    }
    
    if mv_sharpe is not None:
        comparison_data['Strategy'].append('Markowitz')
        comparison_data['Sharpe Ratio'].append(mv_sharpe)
        comparison_data['Improvement'].append(f"{(info['final_sharpe']/mv_sharpe - 1)*100:+.1f}%")
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.format({'Sharpe Ratio': '{:.3f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Interprétation
    if info['final_sharpe'] > equal_sharpe:
        improvement = (info['final_sharpe'] / equal_sharpe - 1) * 100
        st.success(f"✅ RL Agent outperforms Equal Weight by {improvement:.1f}%!")
    else:
        st.warning("⚠️ RL Agent underperforms Equal Weight. Consider more training episodes.")
    
    # Save section
    st.markdown("---")
    st.markdown("### 💾 Save RL Portfolio")
    
    with st.form("save_rl_portfolio_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Portfolio Name", placeholder="My RL Portfolio")
        
        with col2:
            amount = st.number_input(
                "Initial Amount ($)", 
                min_value=100.0, 
                value=10000.0, 
                step=100.0
            )
        
        submit = st.form_submit_button("Save Portfolio", use_container_width=True)
        
        if submit:
            if not name:
                st.error("❌ Please enter a portfolio name")
            else:
                try:
                    # Créer le portfolio
                    portfolio = Portfolio(assets, data)
                    portfolio.set_weights(list(weights))
                    
                    # Sauvegarder
                    from database import save_portfolio
                    save_portfolio(
                        user_id=user_id
                        portfolio, 
                        name, 
                        model="rl",
                        amount=amount,
                        method=info['method'],
                        n_episodes=info['n_episodes'],
                        final_sharpe=info['final_sharpe']
                    )
                    
                    st.success(f"✅ Portfolio '{name}' saved successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error saving portfolio: {str(e)}")


# ============================================================================
# MODIFICATION À FAIRE DANS portfolio_manager.py
# ============================================================================

def update_render_portfolio_build_tab():
    """
    Instructions pour modifier render_portfolio_build_tab() dans portfolio_manager.py
    
    1. Ajouter l'import en haut du fichier:
       from rl_portfolio_builder import build_rl_portfolio
    
    2. Modifier la liste des modèles:
       AVANT:
       models = ["Markowitz", "Discretionary", "Naive", "Beta Weighted", "ML (PCA/ICA)"]
       
       APRÈS:
       models = ["Markowitz", "Discretionary", "Naive", "Beta Weighted", "ML (PCA/ICA)", "RL (Reinforcement Learning)"]
    
    3. Ajouter le cas RL:
       elif model_select == "RL (Reinforcement Learning)":
           build_rl_portfolio(assets, data, theme)
    """
    pass


# ============================================================================
# EXEMPLE D'INTÉGRATION COMPLÈTE
# ============================================================================

def example_integration():
    """
    Exemple complet d'intégration dans portfolio_manager.py
    """
    code = '''
# Dans portfolio_manager.py

# 1. Ajouter les imports
from rl_portfolio_builder import build_rl_portfolio

# 2. Dans render_portfolio_build_tab(), modifier:

def render_portfolio_build_tab():
    """Onglet construction de portfolio"""
    theme = get_theme_colors()
    st.markdown("### Build New Portfolio")
    
    tickers = st.text_input(
        "Stock Tickers", 
        placeholder="AAPL, MSFT, GOOGL",
        help="Enter comma-separated ticker symbols"
    )
    
    if not tickers:
        st.info("👆 Enter stock tickers to start building your portfolio")
        return
    
    try:
        assets = [t.strip().upper() for t in tickers.split(",")]
        if len(assets) == 0:
            st.warning("Please enter at least one ticker")
            return
        
        with st.spinner(f"Fetching data for {', '.join(assets)}..."):
            data = yahoo.retrieve_data(tuple(assets))
        
        if data.empty:
            st.error("Could not retrieve data. Please check and try again.")
            return
        
        st.success(f"✅ Data loaded for {len(assets)} asset(s)")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    st.markdown("---")
    st.markdown("### Select Portfolio Model")
    
    # MODIFICATION ICI - Ajouter RL
    models = [
        "Markowitz", 
        "Discretionary", 
        "Naive", 
        "Beta Weighted", 
        "ML (PCA/ICA)",
        "RL (Reinforcement Learning)"  # NOUVEAU
    ]
    
    model_select = st.selectbox(
        "Model", 
        models,
        help="Choose your portfolio optimization model"
    )
    
    if model_select == "Markowitz":
        build_markowitz(assets, data, theme)
    elif model_select == "Discretionary":
        build_discretionary(assets, data, theme)
    elif model_select == "Naive":
        build_naive(assets, data, theme)
    elif model_select == "Beta Weighted":
        build_beta_weighted(assets, data, theme)
    elif model_select == "ML (PCA/ICA)":
        build_ml_portfolio(assets, data, theme)
    elif model_select == "RL (Reinforcement Learning)":  # NOUVEAU
        build_rl_portfolio(assets, data, theme)
'''
    
    return code


if __name__ == "__main__":
    print("=" * 80)
    print("RL PORTFOLIO BUILDER - STREAMLIT INTEGRATION")
    print("=" * 80)
    print("\n📝 Instructions d'intégration:")
    print("\n1. Copiez rl_portfolio_simple.py dans votre projet")
    print("2. Copiez ce fichier (rl_portfolio_builder.py) dans votre projet")
    print("3. Modifiez portfolio_manager.py selon les instructions ci-dessus")
    print("\n" + "=" * 80)
    print("\n📄 Code d'exemple d'intégration:")
    print("\n" + example_integration())
    print("\n" + "=" * 80)
    print("\n✅ Prêt à l'emploi - Aucune dépendance lourde requise!")
    print("   (Utilise seulement NumPy et Pandas déjà installés)")
    print("\n" + "=" * 80)
