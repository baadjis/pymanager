# sidebar.py
"""
Sidebar simple et fonctionnelle style Claude
"""

import streamlit as st
from uiconfig import get_theme_colors

try:
    from database import get_portfolios
except:
    def get_portfolios():
        return []


def render_sidebar():
    """Affiche la sidebar avec navigation"""
    theme = get_theme_colors()
    is_expanded = st.session_state.get('sidebar_expanded', True)
    
    # CSS pour la sidebar
    st.markdown(f"""
    <style>
        /* Forcer la largeur de la sidebar */
        [data-testid="stSidebar"] {{
            background: {theme['bg_card']} !important;
            border-right: 1px solid {theme['border']} !important;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background: {theme['bg_card']};
        }}
        
        /* Header logo */
        .sidebar-logo {{
            text-align: center;
            padding: 24px 16px;
            border-bottom: 1px solid {theme['border']};
            margin-bottom: 16px;
        }}
        
        .logo-phi {{
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}
        
        .logo-text {{
            font-size: 20px;
            font-weight: 700;
            color: {theme['text_primary']};
        }}
        
        /* Boutons de navigation */
        div.row-widget.stButton > button {{
            width: 100%;
            background: transparent;
            color: {theme['text_secondary']};
            border: none;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: left;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            margin: 2px 0;
        }}
        
        div.row-widget.stButton > button:hover {{
            background: rgba(99, 102, 241, 0.08);
            color: {theme['text_primary']};
            transform: none;
            box-shadow: none;
        }}
        
        /* Bouton actif */
        div.row-widget.stButton > button[data-baseweb="button"]:focus {{
            background: rgba(99, 102, 241, 0.12);
            color: {theme['accent']};
            border-left: 3px solid {theme['accent']};
        }}
        
        /* Divider */
        hr {{
            margin: 16px 0;
            border: none;
            border-top: 1px solid {theme['border']};
        }}
        
        /* Stats cards */
        .stat-card {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }}
        
        .stat-label {{
            font-size: 11px;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: {theme['text_primary']};
        }}
        
        /* User box */
        .user-section {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 16px;
            background: {theme['bg_card']};
            border-top: 1px solid {theme['border']};
        }}
        
        .user-box {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px;
            border-radius: 8px;
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
        }}
        
        .user-avatar {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 15px;
        }}
        
        .user-info {{
            flex: 1;
        }}
        
        .user-name {{
            font-size: 13px;
            font-weight: 600;
            color: {theme['text_primary']};
            margin: 0;
        }}
        
        .user-email {{
            font-size: 11px;
            color: {theme['text_secondary']};
            margin: 0;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Logo header
        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="logo-phi">Φ</div>
            <div class="logo-text">Manager</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation pages
        st.markdown("### 📍 Navigation")
        
        if st.button("🏠  Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.current_page = "Dashboard"
            st.rerun()
        
        if st.button("📊  Portfolio Manager", key="nav_portfolio", use_container_width=True):
            st.session_state.current_page = "Portfolio Manager"
            st.rerun()
        
        if st.button("🔍  Stock Explorer", key="nav_stock", use_container_width=True):
            st.session_state.current_page = "Stock Explorer"
            st.rerun()
        
        if st.button("🎯  Stock Screener", key="nav_screener", use_container_width=True):
            st.session_state.current_page = "Stock Screener"
            st.rerun()
        
        if st.button("🤖  AI Assistant", key="nav_ai", use_container_width=True):
            st.session_state.current_page = "AI Assistant"
            st.rerun()
        
        # Divider
        st.markdown("---")
        
        # Theme toggle
        st.markdown("### ⚙️ Settings")
        
        theme_emoji = "☀️" if st.session_state.theme == "dark" else "🌙"
        theme_label = f"{theme_emoji}  Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'}"
        
        if st.button(theme_label, key="theme_toggle", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        
        # Stats
        st.markdown("---")
        st.markdown("### 📊 Overview")
        
        try:
            portfolios = list(get_portfolios())
            total = sum([p.get('amount', 0) for p in portfolios])
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Value</div>
                <div class="stat-value">${total:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Portfolios</div>
                <div class="stat-value">{len(portfolios)}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Value</div>
                <div class="stat-value">$0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Portfolios</div>
                <div class="stat-value">0</div>
            </div>
            """, unsafe_allow_html=True)
        
        # User section at bottom
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Spacer
        
        user_initial = st.session_state.get('user_initial', 'U')
        user_name = st.session_state.get('user_name', 'User')
        user_email = st.session_state.get('user_email', 'user@phi.com')
        
        st.markdown(f"""
        <div class="user-section">
            <div class="user-box">
                <div class="user-avatar">{user_initial}</div>
                <div class="user-info">
                    <div class="user-name">{user_name}</div>
                    <div class="user-email">{user_email}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
