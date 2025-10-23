"""
Black-Litterman Portfolio Builder - Streamlit Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from portfolio import Portfolio
from .bl_portfolio import (
    black_litterman_portfolio,
    estimate_market_caps_from_data,
    create_bullish_tech_views,
    create_relative_views,
    create_sector_rotation_views
)


def build_black_litterman_portfolio(assets, data, theme):
    """
    Interface Streamlit pour Black-Litterman
    """
    st.markdown("#### 🎯 Black-Litterman Portfolio")
    
    st.info("""
    **Black-Litterman** combine l'équilibre du marché (CAPM) avec vos opinions personnelles.
    
    **Avantages:**
    - Intègre vos vues de marché
    - Plus stable que Markowitz seul
    - Gère l'incertitude des vues
    """)
    
    # Préparer les rendements
    try:
        from portfolio import get_log_returns
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
        
        if returns_df.empty or len(returns_df) < 50:
            st.error("❌ Insufficient data for Black-Litterman (need at least 50 data points)")
            return
        
        st.success(f"✅ Using {len(returns_df)} data points")
        
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return
    
    # Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tau = st.slider(
            "Uncertainty (τ)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Incertitude du prior. Plus élevé = plus de poids aux vues. Typique: 0.05"
        )
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Taux sans risque annualisé"
        ) / 100
    
    with col2:
        use_market_caps = st.checkbox(
            "Use Market Caps",
            value=False,
            help="Si non coché, utilise poids égaux comme proxy"
        )
        
        risk_aversion = st.slider(
            "Risk Aversion",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.5,
            help="Coefficient d'aversion au risque. 2.5 est typique"
        )
    
    # Market Caps
    market_caps = None
    if use_market_caps:
        st.markdown("**Market Capitalizations (Optional):**")
        st.info("Laissez vide pour estimation automatique basée sur les données")
        
        market_caps_input = {}
        cols = st.columns(min(3, len(assets)))
        
        for idx, asset in enumerate(assets):
            with cols[idx % 3]:
                cap = st.number_input(
                    f"{asset} ($B)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    key=f"mcap_{asset}"
                )
                if cap > 0:
                    market_caps_input[asset] = cap
        
        if market_caps_input:
            market_caps = market_caps_input
        else:
            with st.spinner("Estimating market caps from data..."):
                market_caps = estimate_market_caps_from_data(returns_df)
            st.success("✅ Market caps estimated from data")
    
    # Vues de l'investisseur
    st.markdown("---")
    st.markdown("### 💭 Investor Views")
    
    view_method = st.radio(
        "How would you like to add views?",
        ["Manual", "Presets", "No Views (Market Equilibrium)"],
        horizontal=True
    )
    
    views = {}
    confidences = {}
    view_type = 'absolute'
    
    if view_method == "Manual":
        view_type = st.selectbox(
            "View Type",
            ["absolute", "relative"],
            help="Absolute: 'AAPL will return 15%' | Relative: 'AAPL will outperform MSFT by 5%'"
        )
        
        n_views = st.number_input(
            "Number of Views",
            min_value=1,
            max_value=min(5, len(assets)),
            value=1,
            step=1
        )
        
        st.markdown("**Enter Your Views:**")
        
        for i in range(n_views):
            st.markdown(f"**View {i+1}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if view_type == 'absolute':
                    asset = st.selectbox(
                        "Asset",
                        assets,
                        key=f"view_asset_{i}"
                    )
                    view_key = asset
                else:  # relative
                    asset1 = st.selectbox(
                        "Asset 1 (will outperform)",
                        assets,
                        key=f"view_asset1_{i}"
                    )
                    asset2 = st.selectbox(
                        "Asset 2 (will underperform)",
                        [a for a in assets if a != asset1],
                        key=f"view_asset2_{i}"
                    )
                    view_key = f"{asset1}-{asset2}"
            
            with col2:
                expected_return = st.number_input(
                    "Expected Return (%)" if view_type == 'absolute' else "Spread (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=15.0 if view_type == 'absolute' else 5.0,
                    step=1.0,
                    key=f"view_return_{i}"
                ) / 100
            
            with col3:
                confidence = st.slider(
                    "Confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    key=f"view_conf_{i}",
                    help="0.1 = low confidence, 1.0 = very high confidence"
                )
            
            views[view_key] = expected_return
            confidences[view_key] = confidence
    
    elif view_method == "Presets":
        preset = st.selectbox(
            "Select Preset",
            [
                "Bullish on Tech",
                "Tech vs Finance",
                "Sector Rotation: Tech > Energy"
            ]
        )
        
        if preset == "Bullish on Tech":
            views, confidences = create_bullish_tech_views(assets)
            view_type = 'absolute'
            st.info("📈 Bullish views on tech stocks with 70% confidence")
        
        elif preset == "Tech vs Finance":
            # Trouver un stock tech et un stock finance
            tech = [a for a in assets if a in ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']]
            finance = [a for a in assets if a in ['JPM', 'BAC', 'GS', 'C', 'MS']]
            
            if tech and finance:
                views, confidences = create_relative_views(
                    assets,
                    outperformer=tech[0],
                    underperformer=finance[0],
                    spread=0.05,
                    confidence=0.6
                )
                view_type = 'relative'
                st.info(f"📊 Relative view: {tech[0]} will outperform {finance[0]} by 5%")
            else:
                st.warning("⚠️ Need both tech and finance stocks for this preset")
                views = {}
        
        elif preset == "Sector Rotation: Tech > Energy":
            views, confidences = create_sector_rotation_views(assets, 'tech', 'energy')
            view_type = 'absolute'
            st.info("🔄 Sector rotation: Bullish on Tech, Bearish on Energy")
    
    # Afficher les vues entrées
    if views:
        st.markdown("**Your Views Summary:**")
        views_df = pd.DataFrame({
            'View': list(views.keys()),
            'Expected Return': [f"{v:.1%}" for v in views.values()],
            'Confidence': [f"{confidences.get(k, 0.5):.0%}" for k in views.keys()]
        })
        st.dataframe(views_df, use_container_width=True, hide_index=True)
    
    # Contraintes (optionnelles)
    with st.expander("🔧 Advanced Constraints (Optional)"):
        use_constraints = st.checkbox("Add portfolio constraints")
        
        constraints = {}
        if use_constraints:
            col1, col2 = st.columns(2)
            
            with col1:
                min_weight = st.number_input(
                    "Min Weight per Asset (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.0,
                    step=1.0
                ) / 100
                constraints['min_weight'] = min_weight
            
            with col2:
                max_weight = st.number_input(
                    "Max Weight per Asset (%)",
                    min_value=20.0,
                    max_value=100.0,
                    value=40.0,
                    step=5.0
                ) / 100
                constraints['max_weight'] = max_weight
        else:
            constraints = None
    
    # Bouton de construction
    if st.button("🚀 Build Black-Litterman Portfolio", use_container_width=True, type="primary"):
        with st.spinner("Computing Black-Litterman portfolio..."):
            try:
                # Construire le portfolio
                if not views:
                    st.info("No views provided - using market equilibrium")
                
                weights, info = black_litterman_portfolio(
                    returns_data=returns_df,
                    views=views if views else {},
                    view_type=view_type,
                    confidences=confidences if confidences else None,
                    market_caps=market_caps,
                    risk_free_rate=risk_free_rate,
                    tau=tau,
                    constraints=constraints
                )
                
                # Sauvegarder dans session state
                st.session_state.bl_weights = weights
                st.session_state.bl_info = info
                st.session_state.bl_assets = assets
                
                # Afficher les résultats
                display_black_litterman_results(weights, info, assets, returns_df, data, theme)
                
            except Exception as e:
                st.error(f"❌ Error building portfolio: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())


def display_black_litterman_results(weights, info, assets, returns_df, data, theme):
    """Affiche les résultats du portfolio Black-Litterman"""
    
    st.markdown("---")
    st.markdown("## 📊 Black-Litterman Results")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return",
            f"{info['portfolio_return']:.2%}",
            help="Posterior expected return"
        )
    
    with col2:
        st.metric(
            "Volatility",
            f"{info['portfolio_vol']:.2%}",
            help="Portfolio standard deviation"
        )
    
    with col3:
        sharpe_color = "🟢" if info['sharpe_ratio'] > 1 else "🟡" if info['sharpe_ratio'] > 0 else "🔴"
        st.metric(
            "Sharpe Ratio",
            f"{info['sharpe_ratio']:.3f} {sharpe_color}"
        )
    
    with col4:
        st.metric(
            "Views Used",
            info['n_views']
        )
    
    # Comparaison des poids
    st.markdown("### 🎯 Weights Comparison")
    
    comparison_df = pd.DataFrame({
        'Asset': assets,
        'Market Weights': [f"{w:.2%}" for w in info['market_weights']],
        'BL Weights': [f"{w:.2%}" for w in weights],
        'Difference': [f"{(w - m)*100:+.1f}pp" for w, m in zip(weights, info['market_weights'])]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Graphique de comparaison
    fig = go.Figure()
    
    x = np.arange(len(assets))
    width = 0.35
    
    fig.add_trace(go.Bar(
        x=x - width/2,
        y=info['market_weights'],
        name='Market Weights',
        marker_color='lightblue',
        text=[f"{w:.1%}" for w in info['market_weights']],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=x + width/2,
        y=weights,
        name='BL Weights',
        marker_color='#6366F1',
        text=[f"{w:.1%}" for w in weights],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Market vs Black-Litterman Weights",
        xaxis=dict(tickmode='array', tickvals=x, ticktext=assets),
        yaxis_title="Weight",
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns comparison
    st.markdown("### 📈 Returns Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Equilibrium vs Posterior returns
        returns_comp = pd.DataFrame({
            'Asset': assets,
            'Equilibrium': [f"{r:.2%}" for r in info['equilibrium_returns']],
            'Posterior': [f"{r:.2%}" for r in info['posterior_returns']],
            'Change': [f"{(p - e)*100:+.1f}pp" for e, p in zip(info['equilibrium_returns'], info['posterior_returns'])]
        })
        
        st.markdown("**Expected Returns:**")
        st.dataframe(returns_comp, use_container_width=True, hide_index=True)
    
    with col2:
        # Graphique
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=assets,
            y=info['equilibrium_returns'],
            mode='lines+markers',
            name='Equilibrium',
            line=dict(color='gray', dash='dash'),
            marker=dict(size=8)
        ))
        
        fig2.add_trace(go.Scatter(
            x=assets,
            y=info['posterior_returns'],
            mode='lines+markers',
            name='Posterior (BL)',
            line=dict(color='#6366F1', width=3),
            marker=dict(size=10)
        ))
        
        fig2.update_layout(
            title="Expected Returns: Equilibrium vs Posterior",
            yaxis_title="Expected Return",
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Pie chart
    st.markdown("### 🥧 Portfolio Composition")
    
    fig3 = go.Figure(data=[go.Pie(
        labels=assets,
        values=weights,
        hole=0.4,
        marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6'])
    )])
    
    fig3.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Model details
    with st.expander("🔍 Model Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Parameters:**")
            st.write(f"• τ (tau): {info['tau']:.3f}")
            st.write(f"• Risk-Free Rate: {info.get('risk_free_rate', 0.02):.2%}")
            st.write(f"• View Type: {info['view_type']}")
            st.write(f"• Number of Views: {info['n_views']}")
        
        with col2:
            st.markdown("**Portfolio Statistics:**")
            st.write(f"• Expected Return: {info['portfolio_return']:.2%}")
            st.write(f"• Volatility: {info['portfolio_vol']:.2%}")
            st.write(f"• Sharpe Ratio: {info['sharpe_ratio']:.3f}")
            st.write(f"• Sum of Weights: {weights.sum():.4f}")
        
        if info['views']:
            st.markdown("**Active Views:**")
            for view_key, view_value in info['views'].items():
                st.write(f"• {view_key}: {view_value:.2%}")
    
    # Create Portfolio object for advanced metrics
    st.markdown("---")
    st.markdown("### 📊 Advanced Portfolio Metrics")
    
    try:
        portfolio = Portfolio(assets, data)
        portfolio.set_weights(list(weights))
        
        # Benchmark
        benchmark = None
        try:
            from factory import create_benchmark
            benchmark = create_benchmark("^GSPC", period="5y")
        except:
            pass
        
        # Afficher métriques avancées
        from portfolio_helpers import render_advanced_metrics_section
        render_advanced_metrics_section(portfolio, benchmark)
        
    except Exception as e:
        st.warning(f"Could not compute advanced metrics: {e}")
    
    # Save section
    st.markdown("---")
    st.markdown("### 💾 Save Portfolio")
    
    with st.form("save_bl_portfolio_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Portfolio Name",
                placeholder="My Black-Litterman Portfolio"
            )
        
        with col2:
            amount = st.number_input(
                "Initial Amount ($)",
                min_value=100.0,
                value=10000.0,
                step=1000.0
            )
        
        submit = st.form_submit_button("Save Portfolio", use_container_width=True)
        
        if submit and name:
            try:
                portfolio = Portfolio(assets, data)
                portfolio.set_weights(list(weights))
                
                from database import save_portfolio
                save_portfolio(
                    portfolio,
                    name,
                    model="black_litterman",
                    amount=amount,
                    bl_params={
                        'tau': info['tau'],
                        'n_views': info['n_views'],
                        'sharpe': info['sharpe_ratio']
                    }
                )
                
                st.success(f"✅ Portfolio '{name}' saved successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error saving portfolio: {e}")


# Pour l'intégration dans backtesting
def black_litterman_strategy(returns_df, 
                             views=None, 
                             view_type='absolute',
                             **kwargs):
    """
    Wrapper pour utiliser Black-Litterman dans backtesting
    
    Args:
        returns_df: DataFrame des rendementss
        views: Dict des vues (optionnel)
        view_type: 'absolute' ou 'relative'
        **kwargs: Paramètres additionnels (tau, market_caps, etc.)
    
    Returns:
        weights: Array des poids
        info: Dict d'informations
    """
    if views is None:
        views = {}
    
    weights, info = black_litterman_portfolio(
        returns_data=returns_df,
        views=views,
        view_type=view_type,
        **kwargs
    )
    
    return weights, info
