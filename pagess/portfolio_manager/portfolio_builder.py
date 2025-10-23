# pages/portfolio_builder.py
"""
Portfolio Builder - Fonctions communes pour construire des portfolios
Corrigé pour éviter les imports circulaires
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio import Portfolio
from factory import create_portfolio_by_name
from dataprovider import yahoo
from database import save_portfolio
from uiconfig import get_theme_colors


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
            st.error("Could not retrieve data for these tickers. Please check and try again.")
            return
        
        st.success(f"✅ Data loaded for {len(assets)} asset(s)")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    st.markdown("---")
    st.markdown("### Select Portfolio Model")
    
    models = [
        "Markowitz", 
        "Discretionary", 
        "Naive", 
        "Beta Weighted", 
        "ML (PCA/ICA)",
        "RL (Reinforcement Learning)",
        "BL (Black Litterman)"
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
        from .ml_portfolio_builder import build_ml_portfolio
        build_ml_portfolio(assets, data, theme)
    elif model_select == "RL (Reinforcement Learning)":
        from .rl_portfolio_builder import build_rl_portfolio
        build_rl_portfolio(assets, data, theme)
    elif model_select == "BL (Black Litterman)":
        from .bl_portfolio_builder import build_black_litterman_portfolio
        build_black_litterman_portfolio(assets, data, theme)


def build_markowitz(assets, data, theme):
    """Build Markowitz portfolio"""
    st.markdown("#### Markowitz Optimization")
    
    methods = ["Sharp", "Risk", "Return", "Unsafe", "Compare"]
    method_select = st.selectbox("Optimization Method", methods)
    
    portfolio = None
    method_params = {}
    
    if method_select == "Compare":
        st.info("📊 This will plot the efficient frontier")
        if st.button("Generate Efficient Frontier", use_container_width=True):
            with st.spinner("Generating efficient frontier..."):
                try:
                    from viz import plot_markowitz_curve
                    import matplotlib.pyplot as plt
                    plot_markowitz_curve(assets, n=3000, data=data, show=False)
                    st.pyplot(fig=plt)
                    plt.close()
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
        return
    
    if method_select == "Risk":
        risk_tolerance = st.number_input(
            "Risk Tolerance (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=20.0,
            step=0.1
        )
        if risk_tolerance > 0:
            method_params = {"risk_tolerance": risk_tolerance}
            portfolio = create_portfolio_by_name(
                assets, 
                method_select.lower(), 
                data, 
                risk_tolerance=risk_tolerance
            )
    
    elif method_select == "Return":
        expected_return = st.number_input(
            "Expected Return", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25,
            step=0.01
        )
        if expected_return > 0:
            method_params = {"expected_return": expected_return}
            portfolio = create_portfolio_by_name(
                assets, 
                method_select.lower(), 
                data, 
                expected_return=expected_return
            )
    
    else:
        portfolio = create_portfolio_by_name(assets, method_select.lower(), data)
    
    if portfolio:
        display_portfolio_results(
            portfolio, 
            assets, 
            "markowitz", 
            method=method_select.lower(),
            **method_params
        )


def build_discretionary(assets, data, theme):
    """Build discretionary portfolio with manual weights"""
    st.markdown("#### Discretionary Portfolio")
    st.info("💡 Enter custom weights for each asset (must sum to 1.0)")
    
    weights_str = st.text_input(
        "Weights (comma-separated)", 
        placeholder="0.3, 0.4, 0.3",
        help=f"Enter {len(assets)} weights separated by commas"
    )
    
    if weights_str:
        try:
            weights = [float(w.strip()) for w in weights_str.split(",")]
            
            if len(weights) != len(assets):
                st.error(f"❌ Please enter exactly {len(assets)} weights (one for each asset)")
                return
            
            weights_sum = sum(weights)
            if abs(weights_sum - 1.0) > 1e-6:
                st.error(f"❌ Weights must sum to 1.0, got {weights_sum:.4f}")
                return
            
            portfolio = Portfolio(assets, data)
            portfolio.set_weights(weights)
            
            st.success("✅ Portfolio created successfully!")
            display_portfolio_results(portfolio, assets, "discretionary")
            
        except ValueError as e:
            st.error(f"❌ Invalid weight format: {str(e)}")


def build_naive(assets, data, theme):
    """Build naive (equal-weighted) portfolio"""
    st.markdown("#### Naive Portfolio (Equal Weights)")
    st.info("📊 All assets will be equally weighted")
    
    n = len(assets)
    weight = 1.0 / n
    weights = [weight] * n
    
    st.markdown("**Asset Allocation:**")
    for asset, w in zip(assets, weights):
        st.write(f"- {asset}: {w:.2%}")
    
    if st.button("Create Portfolio", use_container_width=True):
        portfolio = Portfolio(assets, data)
        portfolio.set_weights(weights)
        
        st.success("✅ Portfolio created!")
        display_portfolio_results(portfolio, assets, "naive")


def build_beta_weighted(assets, data, theme):
    """Build beta-weighted portfolio"""
    st.markdown("#### Beta Weighted Portfolio")
    st.info("📈 Weights based on asset beta values")
    
    with st.spinner("Calculating betas..."):
        try:
            betas = yahoo.get_assets_beta(assets, benchmark="^GSPC")
            betas_sum = sum(betas)
            
            if betas_sum == 0 or all(b == 1.0 for b in betas):
                st.warning("⚠️ Using default beta values (1.0) for all assets")
            
            weights = [beta / betas_sum for beta in betas]
            
            st.markdown("**Beta Values & Weights:**")
            beta_df = pd.DataFrame({
                'Asset': assets,
                'Beta': [f"{b:.4f}" for b in betas],
                'Weight': [f"{w:.2%}" for w in weights]
            })
            st.dataframe(beta_df, use_container_width=True, hide_index=True)
            
            if st.button("Create Portfolio", use_container_width=True):
                portfolio = Portfolio(assets, data)
                portfolio.set_weights(weights)
                
                st.success("✅ Portfolio created!")
                display_portfolio_results(portfolio, assets, "betaweighted")
                
        except Exception as e:
            st.error(f"Error calculating betas: {str(e)}")


def display_portfolio_results(portfolio, assets, model, **kwargs):
    """Display portfolio metrics and save option"""
    theme = get_theme_colors()
    
    st.markdown("---")
    st.markdown("### Portfolio Overview")
    
    # Import helpers here to avoid circular imports
    try:
        from .portfolio_helpers import (
            METRIC_TOOLTIPS, 
            create_metric_card, 
            render_advanced_metrics_section
        )
    except ImportError:
        # Fallback si portfolio_helpers n'existe pas encore
        METRIC_TOOLTIPS = {}
        def create_metric_card(label, value, tooltip, **kwargs):
            return f"<div><b>{label}:</b> {value}</div>"
        def render_advanced_metrics_section(portfolio, benchmark):
            st.info("Advanced metrics not available")
    
    # Metrics principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Expected Return",
            f"{portfolio.expected_return:.2%}",
            METRIC_TOOLTIPS.get("Expected Return", ""),
            icon="📈"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Volatility",
            f"{portfolio.stdev:.2%}",
            METRIC_TOOLTIPS.get("Volatility", ""),
            icon="⚠️"
        ), unsafe_allow_html=True)
    
    with col3:
        sharpe_color = "#10B981" if portfolio.sharp_ratio > 1 else "#F59E0B" if portfolio.sharp_ratio > 0 else "#EF4444"
        st.markdown(create_metric_card(
            "Sharpe Ratio",
            f"{portfolio.sharp_ratio:.3f}",
            METRIC_TOOLTIPS.get("Sharpe Ratio", ""),
            color=sharpe_color,
            icon="🎯"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "Max Drawdown",
            f"{portfolio.max_drawdown:.2%}",
            METRIC_TOOLTIPS.get("Max Drawdown", ""),
            color="#EF4444",
            icon="📉"
        ), unsafe_allow_html=True)
    
    # Weights table
    st.markdown("---")
    st.markdown("### Asset Allocation")
    colc1, colc2 = st.columns(2)
   
    with colc1:
        weights_df = pd.DataFrame({
            'Asset': assets,
            'Weight': [f"{w:.2%}" for w in portfolio.weights]
        })
        styler = weights_df.style
        styler.hide()
        st.write(styler.to_html(index=False), unsafe_allow_html=True)
    
    # Composition pie chart
    with colc2:
        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=portfolio.weights,
            hole=0.4,
            marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']),
        )])
        fig.update_layout(
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            font=dict(color=theme['text_primary'])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced metrics
    try:
        from factory import create_benchmark
        benchmark = None
        try:
            benchmark = create_benchmark("^GSPC", period="5y")
        except:
            pass
        
        render_advanced_metrics_section(portfolio, benchmark)
    except Exception as e:
        st.warning(f"Could not load advanced metrics: {str(e)}")
    
    # Save section
    st.markdown("---")
    st.markdown("### 💾 Save Portfolio")
    
    with st.form("save_portfolio_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Portfolio Name", placeholder="My Portfolio")
        
        with col2:
            amount = st.number_input("Initial Amount ($)", min_value=100.0, value=10000.0, step=100.0)
        
        submit = st.form_submit_button("Save Portfolio", use_container_width=True)
        
        if submit:
            if not name:
                st.error("❌ Please enter a portfolio name")
            else:
                try:
                    save_portfolio(portfolio, name, model=model, amount=amount, **kwargs)
                    st.success(f"✅ Portfolio '{name}' saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving portfolio: {str(e)}")
