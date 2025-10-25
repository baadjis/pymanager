# app3.py - Point d'entrée principal avec authentification
"""
ΦManager - Modern Portfolio & Market Intelligence Platform
Point d'entrée principal de l'application avec système d'authentification
"""
import streamlit as st
from uiconfig import init_session_state, get_theme_colors
from styles import apply_custom_css
from sidebar_collapsible import render_sidebar

# Import des pages
from pagess import (
    render_dashboard,
    render_portfolio_manager,
    render_stock_explorer,
    render_stock_screener,
    render_ai_assistant,
    render_market
)

# Import de la page auth
from pagess.auth import render_auth

# Configuration de la page
st.set_page_config(
    page_title="ΦManager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation
init_session_state()
theme = get_theme_colors()
apply_custom_css(theme)

# Vérifier l'authentification
user_id = st.session_state.user_id
is_authenticated = True if user_id!='' else False

# Si pas authentifié, afficher uniquement la page de login
if not is_authenticated:
    # Pas de sidebar pour la page auth
    render_auth()
    
else:
# Si authentifié, afficher la sidebar et les pages
 render_sidebar()

# Récupérer la page courante
 page = st.session_state.get('current_page', 'Dashboard')

# Routing des pages
 if page == "Dashboard":
    render_dashboard()
 elif page == "Portfolio":
    render_portfolio_manager()
 elif page == "Market":
    render_market()
 elif page == "AI Assistant":
    render_ai_assistant()
 elif page == "Stock Explorer":
    render_stock_explorer()
 elif page == "Stock Screener":
    render_stock_screener()
 else:
    # Page par défaut
    render_dashboard()
