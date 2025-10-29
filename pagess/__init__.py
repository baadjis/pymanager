# pagess/__init__.py
"""
Package des pages Streamlit
"""

# Import conditionnel pour éviter les erreurs si les modules n'existent pas encore
try:
    from .dashboard import render_dashboard
except ImportError:
    def render_dashboard():
        import streamlit as st
        st.error("Dashboard module not found")

try:
    from .portfolio_manager import render_portfolio_manager
except ImportError:
    def render_portfolio_manager():
        import streamlit as st
        st.error("Portfolio module not found")

try:
    from .market import render_market
except ImportError:
    def render_market():
        import streamlit as st
        st.error("Market module not found")

try:
    from .ai import render_ai_assistant
except ImportError:
    def render_ai_assistant():
        import streamlit as st
        st.error("AI Assistant module not found")

try:
    from .stock_explorer import render_stock_explorer
except ImportError:
    def render_stock_explorer():
        import streamlit as st
        st.error("Stock Explorer module not found")

try:
    from .screener import render_stock_screener
except ImportError:
    def render_stock_screener():
        import streamlit as st
        st.error("Stock Screener module not found")

# Auth page (obligatoire)
from .auth import render_auth
from .pricing import render_pricing_page

__all__ = [
    'render_dashboard',
    'render_portfolio_manager',
    'render_market',
    'render_ai_assistant',
    'render_stock_explorer',
    'render_stock_screener',
    'render_auth',
    'render_pricing_page'
]
