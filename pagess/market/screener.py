# pages/stock_explorer.py
"""
Page Stock Explorer
"""

import streamlit as st
from stock import Stock
from charts import create_candlestick_chart
from uiconfig import get_theme_colors
from .stock_explorer import render_stock_explorer

from .stock_screener import render_stock_screener
from datetime import datetime

def render_screener():
    """Page Portfolio Manager principale avec 4 onglets"""
    #load_enhanced_styles()
    
    st.markdown("<h2> Screen </h2>", unsafe_allow_html=True)
    
    # 4 onglets principaux
    asset = st.selectbox( "select asset",[
        "stock", 
        "index", 
        "fund",
        "currency"
       
    ],key=f"select_asset_screener")
    
    if asset=="stock":
        render_stock_screener()
        
    elif asset=="index":
         pass
         
    
    elif asset=="fund":
        #render_portfolio_details_tab()
        pass
    
    elif asset=="currency":
         pass
    
