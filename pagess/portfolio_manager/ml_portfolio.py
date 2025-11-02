import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio import Portfolio, get_log_returns
from dataprovider import yahoo
from database import save_portfolio
from .ml_portfolio import pca_portfolio, ica_portfolio, mixed_pca_ica_portfolio, hierarchical_risk_parity_ml

user_id = st.session_state.user_id

def build_ml_portfolio(assets, data, theme):
    st.markdown("#### Machine Learning Portfolio")
    st.info("🤖 Use dimensionality reduction")
    
    ml_method = st.selectbox("ML Method", ["PCA", "ICA", "Mixed PCA-ICA", "HRP with ML"])
    
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
    
    if returns_df.empty or len(returns_df) < 30:
        st.error("❌ Insufficient data (need 30+ points)")
        return
    
    st.success(f"✅ Using {len(returns_df)} points")
    
    weights = None
    info = None
    
    if ml_method == "PCA":
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Components", 1, min(5, len(assets)), min(2, len(assets)))
        with col2:
            use_kernel = st.checkbox("Use Kernel PCA", False)
        
        if st.button("Build PCA", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    weights, info = pca_portfolio(returns_df, n_components=n_components, use_kernel=use_kernel)
                    st.success("✅ Created!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return
    
    elif ml_method == "ICA":
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Components", 2, min(len(assets), 10), min(3, len(assets)))
        with col2:
            max_iter = st.number_input("Max Iterations", 100, 5000, 1000, 100)
        
        if st.button("Build ICA", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    weights, info = ica_portfolio(returns_df, n_components=n_components, max_iter=max_iter)
                    st.success("✅ Created!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return
    
    elif ml_method == "Mixed PCA-ICA":
        pca_weight = st.slider("PCA Weight", 0.0, 1.0, 0.5, 0.1)
        col1, col2 = st.columns(2)
        with col1:
            n_comp_pca = st.slider("PCA Comp", 1, min(5, len(assets)), min(2, len(assets)))
        with col2:
            n_comp_ica = st.slider("ICA Comp", 2, min(10, len(assets)), min(3, len(assets)))
        
        if st.button("Build Mixed", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    weights, info = mixed_pca_ica_portfolio(returns_df, pca_weight=pca_weight, n_components_pca=n_comp_pca, n_components_ica=n_comp_ica)
                    st.success("✅ Created!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return
    
    elif ml_method == "HRP with ML":
        hrp_method = st.radio("Clustering", ["pca", "ica"])
        
        if st.button("Build HRP", use_container_width=True):
            with st.spinner("Computing..."):
                try:
                    weights, info = hierarchical_risk_parity_ml(returns_df, method=hrp_method)
                    st.success("✅ Created!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return
    
    if weights is not None and info is not None:
        portfolio = Portfolio(assets, data)
        portfolio.set_weights(list(weights))
        
        st.markdown("---")
        weights_df = pd.DataFrame({'Asset': assets, 'Weight': [f"{w:.4f}" for w in weights], 'Percentage': [f"{w*100:.2f}%" for w in weights]})
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        fig = go.Figure(data=[go.Bar(x=assets, y=weights, marker=dict(color=weights, colorscale='Viridis', showscale=True))])
        fig.update_layout(title="Weights", template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        from .portfolio_helpers import display_portfolio_results
        display_portfolio_results(portfolio, assets, "ml", method=ml_method.lower().replace(' ', '_'), ml_info=info)
