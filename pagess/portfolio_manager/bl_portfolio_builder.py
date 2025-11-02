import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio import Portfolio
from dataprovider import yahoo
from database import save_portfolio
from .bl_portfolio import (
    black_litterman_portfolio,
    estimate_market_caps_from_data,
    create_bullish_tech_views,
    create_relative_views,
    create_sector_rotation_views
)

user_id = st.session_state.user_id

def build_black_litterman_portfolio(assets, data, theme):
    st.markdown("#### 🎯 Black-Litterman Portfolio")
    st.info("**Black-Litterman** combine l'équilibre du marché (CAPM) avec vos opinions personnelles.")
    
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
            st.error("❌ Insufficient data (need at least 50 points)")
            return
        
        st.success(f"✅ Using {len(returns_df)} data points")
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        tau = st.slider("Uncertainty (τ)", 0.01, 0.10, 0.05, 0.01)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    
    with col2:
        use_market_caps = st.checkbox("Use Market Caps", False)
        risk_aversion = st.slider("Risk Aversion", 1.0, 5.0, 2.5, 0.5)
    
    market_caps = None
    if use_market_caps:
        market_caps_input = {}
        cols = st.columns(min(3, len(assets)))
        for idx, asset in enumerate(assets):
            with cols[idx % 3]:
                cap = st.number_input(f"{asset} ($B)", 0.0, 0.0, 1.0, key=f"mcap_{asset}")
                if cap > 0:
                    market_caps_input[asset] = cap
        
        market_caps = market_caps_input if market_caps_input else estimate_market_caps_from_data(returns_df)
    
    st.markdown("---")
    view_method = st.radio("Views", ["Manual", "Presets", "No Views"], horizontal=True)
    
    views = {}
    confidences = {}
    view_type = 'absolute'
    
    if view_method == "Manual":
        view_type = st.selectbox("View Type", ["absolute", "relative"])
        n_views = st.number_input("Number of Views", 1, min(5, len(assets)), 1, 1)
        
        for i in range(n_views):
            col1, col2, col3 = st.columns(3)
            with col1:
                if view_type == 'absolute':
                    asset = st.selectbox("Asset", assets, key=f"view_asset_{i}")
                    view_key = asset
                else:
                    asset1 = st.selectbox("Asset 1", assets, key=f"view_asset1_{i}")
                    asset2 = st.selectbox("Asset 2", [a for a in assets if a != asset1], key=f"view_asset2_{i}")
                    view_key = f"{asset1}-{asset2}"
            
            with col2:
                expected_return = st.number_input("Expected Return (%)" if view_type == 'absolute' else "Spread (%)", 
                                                 -50.0, 100.0, 15.0 if view_type == 'absolute' else 5.0, 1.0, key=f"view_return_{i}") / 100
            
            with col3:
                confidence = st.slider("Confidence", 0.1, 1.0, 0.6, 0.1, key=f"view_conf_{i}")
            
            views[view_key] = expected_return
            confidences[view_key] = confidence
    
    elif view_method == "Presets":
        preset = st.selectbox("Select Preset", ["Bullish on Tech", "Tech vs Finance", "Sector Rotation: Tech > Energy"])
        if preset == "Bullish on Tech":
            views, confidences = create_bullish_tech_views(assets)
            view_type = 'absolute'
        elif preset == "Tech vs Finance":
            tech = [a for a in assets if a in ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']]
            finance = [a for a in assets if a in ['JPM', 'BAC', 'GS', 'C', 'MS']]
            if tech and finance:
                views, confidences = create_relative_views(assets, tech[0], finance[0], 0.05, 0.6)
                view_type = 'relative'
        elif preset == "Sector Rotation: Tech > Energy":
            views, confidences = create_sector_rotation_views(assets, 'tech', 'energy')
            view_type = 'absolute'
    
    if views:
        views_df = pd.DataFrame({
            'View': list(views.keys()),
            'Expected Return': [f"{v:.1%}" for v in views.values()],
            'Confidence': [f"{confidences.get(k, 0.5):.0%}" for k in views.keys()]
        })
        st.dataframe(views_df, use_container_width=True, hide_index=True)
    
    with st.expander("🔧 Advanced Constraints"):
        use_constraints = st.checkbox("Add constraints")
        constraints = None
        if use_constraints:
            col1, col2 = st.columns(2)
            with col1:
                min_weight = st.number_input("Min Weight (%)", 0.0, 20.0, 0.0, 1.0) / 100
            with col2:
                max_weight = st.number_input("Max Weight (%)", 20.0, 100.0, 40.0, 5.0) / 100
            constraints = {'min_weight': min_weight, 'max_weight': max_weight}
    
    if st.button("🚀 Build", use_container_width=True, type="primary"):
        with st.spinner("Computing..."):
            try:
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
                
                st.session_state.bl_weights = weights
                st.session_state.bl_info = info
                st.session_state.bl_assets = assets
                
                display_black_litterman_results(weights, info, assets, returns_df, data, theme)
            except Exception as e:
                st.error(f"❌ Error: {e}")


def display_black_litterman_results(weights, info, assets, returns_df, data, theme):
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Return", f"{info['portfolio_return']:.2%}")
    with col2:
        st.metric("Volatility", f"{info['portfolio_vol']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{info['sharpe_ratio']:.3f}")
    with col4:
        st.metric("Views Used", info['n_views'])
    
    comparison_df = pd.DataFrame({
        'Asset': assets,
        'Market Weights': [f"{w:.2%}" for w in info['market_weights']],
        'BL Weights': [f"{w:.2%}" for w in weights],
        'Difference': [f"{(w - m)*100:+.1f}pp" for w, m in zip(weights, info['market_weights'])]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    fig = go.Figure()
    x = np.arange(len(assets))
    width = 0.35
    fig.add_trace(go.Bar(x=x - width/2, y=info['market_weights'], name='Market', marker_color='lightblue'))
    fig.add_trace(go.Bar(x=x + width/2, y=weights, name='BL', marker_color='#6366F1'))
    fig.update_layout(title="Market vs BL Weights", xaxis=dict(tickvals=x, ticktext=assets), template='plotly_dark', height=400, barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    with st.form("save_bl"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Portfolio Name", placeholder="My BL Portfolio")
        with col2:
            amount = st.number_input("Initial Amount ($)", 100.0, value=10000.0, step=1000.0)
        
        if st.form_submit_button("Save", use_container_width=True):
            if name:
                try:
                    portfolio = Portfolio(assets, data)
                    portfolio.set_weights(list(weights))
                    save_portfolio(user_id, portfolio, name, model="black_litterman", amount=amount)
                    st.success(f"✅ Saved!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")
