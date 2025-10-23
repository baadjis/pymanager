# pages/explorer.py
"""
Explorer - Hub pour explorer différents types d'actifs
Stocks, Indices, Funds, Currencies, Commodities, Bonds
"""

import streamlit as st
from uiconfig import get_theme_colors
from .stock_explorer import render_stock_explorer
from .index_explorer import render_index_explorer
from .fund_explorer import render_fund_explorer
from .currency_explorer import render_currency_explorer
from .commodity_explorer import render_commodity_explorer
from .bond_explorer import render_bond_explorer


def render_explorer():
    """Page Explorer principale avec sélection d'actifs"""
    theme = get_theme_colors()
    
    # Header
    st.markdown("## 🔍 Asset Explorer")
    st.caption("Explore detailed information about all asset classes")
    
    # Sélection du type d'actif avec icônes - 2 lignes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Stocks", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset', 'stock') == 'stock' else "secondary"):
            st.session_state.explorer_asset = 'stock'
    
    with col2:
        if st.button("📊 Indices", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset') == 'index' else "secondary"):
            st.session_state.explorer_asset = 'index'
    
    with col3:
        if st.button("🏦 Funds", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset') == 'fund' else "secondary"):
            st.session_state.explorer_asset = 'fund'
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("💱 Currencies", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset') == 'currency' else "secondary"):
            st.session_state.explorer_asset = 'currency'
    
    with col5:
        if st.button("🏭 Commodities", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset') == 'commodity' else "secondary"):
            st.session_state.explorer_asset = 'commodity'
    
    with col6:
        pass
        if st.button("📈 Bonds", use_container_width=True, 
                    type="primary" if st.session_state.get('explorer_asset') == 'bond' else "secondary"):
            st.session_state.explorer_asset = 'bond'
        
    st.divider()
    
    # Initialiser par défaut
    if 'explorer_asset' not in st.session_state:
        st.session_state.explorer_asset = 'stock'
    
    # Render selon le type sélectionné
    asset_type = st.session_state.explorer_asset
    
    if asset_type == "stock":
        render_stock_explorer()
    elif asset_type == "index":
        render_index_explorer()
    elif asset_type == "fund":
        render_fund_explorer()
    elif asset_type == "currency":
        render_currency_explorer()
    elif asset_type == "commodity":
        render_commodity_explorer()
        
    elif asset_type == "bond":
        render_bond_explorer()
        
