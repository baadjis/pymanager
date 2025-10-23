# app.py - Fichier principal
"""
ΦManager - Modern Portfolio & Market Intelligence Platform
Point d'entrée principal de l'application
"""
import streamlit as st
from uiconfig import init_session_state, get_theme_colors
from styles import apply_custom_css
from sidebar_collapsible import render_sidebar
from pagess import (
    render_dashboard,
    render_portfolio_manager,
    render_stock_explorer,
    render_stock_screener,
    render_ai_assistant,
    render_market
)

# Configuration de la page

# Force sidebar to show


# Initialisation
init_session_state()
theme = get_theme_colors()
apply_custom_css(theme)

# Affichage de la sidebar
render_sidebar()
page = st.session_state.current_page

# Routing des pages
if page == "Dashboard":
    render_dashboard()
elif page == "Portfolio":
    render_portfolio_manager()

elif page == "Market":
    render_market()
elif page == "AI Assistant":
    render_ai_assistant()
