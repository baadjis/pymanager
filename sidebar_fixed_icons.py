# sidebar_fixed_icons.py
"""
Sidebar avec icônes SVG au lieu d'émojis
Utilisez cette version si les émojis ne s'affichent pas
"""

import streamlit as st
from uiconfig import get_theme_colors

try:
    from database import get_portfolios
except:
    def get_portfolios():
        return []


# Icônes SVG inline
def get_icon(name):
    """Retourne l'icône SVG"""
    icons = {
        'home': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>''',
        
        'chart': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="12" y1="20" x2="12" y2="10"></line>
            <line x1="18" y1="20" x2="18" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="16"></line>
        </svg>''',
        
        'search': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <path d="m21 21-4.35-4.35"></path>
        </svg>''',
        
        'target': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <circle cx="12" cy="12" r="6"></circle>
            <circle cx="12" cy="12" r="2"></circle>
        </svg>''',
        
        'bot': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect width="18" height="10" x="3" y="11" rx="2"></rect>
            <circle cx="12" cy="5" r="2"></circle>
            <path d="M12 7v4"></path>
            <line x1="8" x2="8" y1="16" y2="16"></line>
            <line x1="16" x2="16" y1="16" y2="16"></line>
        </svg>''',
        
        'sun': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="4"></circle>
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path>
        </svg>''',
        
        'moon': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
        </svg>'''
    }
    return icons.get(name, '')


def render_sidebar():
    """Affiche la sidebar avec icônes SVG"""
    theme = get_theme_colors()
    current = st.session_state.get('current_page', 'Dashboard')
    
    # CSS
    st.markdown(f"""
    <style>
        /* Sidebar base */
        [data-testid="stSidebar"] {{
            background: {theme['bg_card']} !important;
            border-right: 1px solid {theme['border']} !important;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background: {theme['bg_card']};
        }}
        
        /* Logo header */
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
        }}
        
        .logo-text {{
            font-size: 20px;
            font-weight: 700;
            color: {theme['text_primary']};
            margin-top: 4px;
        }}
        
        /* Navigation buttons */
        .nav-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            margin: 2px 8px;
            border-radius: 8px;
            color: {theme['text_secondary']};
            background: transparent;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }}
        
        .nav-item:hover {{
            background: rgba(99, 102, 241, 0.08);
            color: {theme['text_primary']};
        }}
        
        .nav-item.active {{
            background: rgba(99, 102, 241, 0.12);
            color: {theme['accent']};
            font-weight: 600;
            position: relative;
        }}
        
        .nav-item.active::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 3px;
            height: 24px;
            background: {theme['accent']};
            border-radius: 0 2px 2px 0;
        }}
        
        .nav-icon {{
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .nav-label {{
            font-size: 14px;
            font-weight: 500;
        }}
        
        /* Streamlit buttons styling */
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
        
        /* Section headers */
        .section-header {{
            font-size: 11px;
            font-weight: 600;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 16px 16px 8px 16px;
        }}
        
        /* Stats cards */
        .stat-card {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 12px;
            margin: 8px 16px;
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
        
        /* User section */
        .user-section {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 280px;
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
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .user-box:hover {{
            background: rgba(99, 102, 241, 0.05);
            border-color: {theme['accent']};
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
        
        .user-name {{
            font-size: 13px;
            font-weight: 600;
            color: {theme['text_primary']};
        }}
        
        .user-email {{
            font-size: 11px;
            color: {theme['text_secondary']};
        }}
        
        hr {{
            margin: 16px 0;
            border: none;
            border-top: 1px solid {theme['border']};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div class="sidebar-logo">
            <div class="logo-phi">Φ</div>
            <div class="logo-text">Manager</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        
        # Pages avec icônes SVG
        pages = [
            ("Dashboard", "home"),
            ("Portfolio Manager", "chart"),
            ("Stock Explorer", "search"),
            ("Stock Screener", "target"),
            ("AI Assistant", "bot")
        ]
        
        for page_name, icon_name in pages:
            is_active = current == page_name
            active_class = "active" if is_active else ""
            
            # Afficher l'icône HTML
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.markdown(f'<div class="nav-icon">{get_icon(icon_name)}</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
        
        # Settings
        st.markdown("---")
        st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)
        
        theme_icon = get_icon("sun" if st.session_state.theme == "dark" else "moon")
        theme_label = "Light Mode" if st.session_state.theme == "dark" else "Dark Mode"
        
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f'<div class="nav-icon">{theme_icon}</div>', unsafe_allow_html=True)
        with col2:
            if st.button(theme_label, key="theme1", use_container_width=True):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
        
        # Overview
        st.markdown("---")
        st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
        
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
        except:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Value</div>
                <div class="stat-value">$0</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Spacer
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        # User
        user_initial = st.session_state.get('user_initial', 'U')
        user_name = st.session_state.get('user_name', 'User')
        user_email = st.session_state.get('user_email', 'user@phi.com')
        
        st.markdown(f"""
        <div class="user-section">
            <div class="user-box">
                <div class="user-avatar">{user_initial}</div>
                <div>
                    <div class="user-name">{user_name}</div>
                    <div class="user-email">{user_email}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
