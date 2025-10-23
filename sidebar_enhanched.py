# sidebar.py
"""
Sidebar collapsible ET fonctionnelle
Mode étendu: icônes + texte | Mode réduit: icônes seules
"""

import streamlit as st
from uiconfig import get_theme_colors

try:
    from database import get_portfolios
except:
    def get_portfolios():
        return []


def render_sidebar():
    """Sidebar avec toggle collapsible qui marche"""
    theme = get_theme_colors()
    
    # Init collapsed state
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    collapsed = st.session_state.sidebar_collapsed
    current = st.session_state.get('current_page', 'Dashboard')
    
    # CSS dynamique selon collapsed state
    st.markdown(f"""
    <style>
        /* Sidebar width dynamique */
        [data-testid="stSidebar"] {{
            min-width: {('80px' if collapsed else '280px')} !important;
            max-width: {('80px' if collapsed else '280px')} !important;
            background: {theme['bg_card']} !important;
            border-right: 1px solid {theme['border']} !important;
            transition: all 0.3s ease;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            width: {('80px' if collapsed else '280px')} !important;
            transition: all 0.3s ease;
        }}
        
        /* Logo */
        .sb-logo {{
            
            padding: {('16px 8px' if collapsed else '24px 16px')} !important;
            border-bottom: 1px solid {theme['border']};
            margin-bottom: 5px;
           
            
        }}
        
        .logo-phi {{
            font-size: {('16px' if collapsed else '36px')};
            font-weight: {(100 if collapsed else 700)};
            background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: font-size 0.3s ease;
            
        }}
        
        .logo-text {{
            font-size: 18px;
            font-weight: 700;
            color: {theme['text_primary']};
           
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
        /* Toggle button - TOUJOURS visible */
        [data-testid="stSidebar"] button[kind="secondary"] {{
            width: 100%;
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            padding: 8px !important;
            margin: 8px 0 !important;
            margin-top:0 !important;
            font-size: 16px !important;
            transition: all 0.2s ease !important;
        }}
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background: rgba(99, 102, 241, 0.15) !important;
            border-color: {theme['accent']} !important;
        }}
        
        /* Boutons navigation */
        [data-testid="stSidebar"] button[kind="primary"] {{
            width: 100%;
            background: transparent !important;
            color: {theme['text_secondary']} !important;
            border: none !important;
            border-radius: 8px !important;
            padding: {('12px 8px' if collapsed else '12px 16px')} !important;
            text-align: {('center' if collapsed else 'left')} !important;
            font-size: {('18px' if collapsed else '14px')} !important;
            font-weight: 500 !important;
            margin: 2px 0 !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
            white-space: nowrap !important;
            overflow: hidden !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"]:hover {{
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
            transform: none !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"]:focus {{
            background: rgba(99, 102, 241, 0.12) !important;
            color: {theme['accent']} !important;
            border-left: 3px solid {theme['accent']} !important;
            box-shadow: none !important;
        }}
        
        /* Section titles */
        .section-title {{
            font-size: 11px;
            font-weight: 600;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: {('8px 8px 4px' if collapsed else '16px 16px 8px')};
            margin-top: 8px;
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
        /* Stats - cachées en mode collapsed */
        .stats-wrapper {{
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
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
        }}
        
        .stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: {theme['text_primary']};
            margin-top: 4px;
        }}
        
        /* User section */
        .user-box {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px {('8px' if collapsed else '16px')};
            margin: 16px {('8px' if collapsed else '16px')};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            background: {theme['bg_card']};
            justify-content: {('center' if collapsed else 'flex-start')};
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
            flex-shrink: 0;
        }}
        
        .user-info {{
            {('display: none;' if collapsed else 'display: block;')}
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
            border: none;
            border-top: 1px solid {theme['border']};
            margin: 16px {('8px' if collapsed else '0')};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div class="sb-logo">
            <div class="logo-phi">Φ</div>
            <div class="logo-text">Manager</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle button - VISIBLE et CLIQUABLE
        toggle_icon = "▶" if collapsed else "◀"
        if st.button(toggle_icon, key="toggle_btn", help="Toggle sidebar", type="secondary"):
            st.session_state.sidebar_collapsed = not collapsed
            st.rerun()
        
        # Navigation
        st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
        
        # Pages - texte caché en mode collapsed
        pages = [
            ("Dashboard", "🏠"),
            ("Portfolio Manager", "📊"),
            ("Stock Explorer", "🔍"),
            ("Stock Screener", "🎯"),
            ("AI Assistant", "🤖")
        ]
        
        for page_name, icon in pages:
            # Mode collapsed: icône seule, Mode étendu: icône + texte
            button_label = icon if collapsed else f"{icon}  {page_name}"
            
            if st.button(button_label, key=f"nav_{page_name}", use_container_width=True, type="primary"):
                st.session_state.current_page = page_name
                st.rerun()
        
        # Settings
        st.markdown("---")
        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
        
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        theme_text = "Light" if st.session_state.theme == "dark" else "Dark"
        theme_label = theme_icon if collapsed else f"{theme_icon}  {theme_text} Mode"
        
        if st.button(theme_label, key="theme_toggle", use_container_width=True, type="primary"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        
        # Stats - seulement en mode étendu
        if not collapsed:
            st.markdown("---")
            st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
            
            try:
                portfolios = list(get_portfolios())
                total = sum([p.get('amount', 0) for p in portfolios])
                
                st.markdown(f"""
                <div class="stats-wrapper">
                    <div class="stat-card">
                        <div class="stat-label">Total Value</div>
                        <div class="stat-value">${total:,.0f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Portfolios</div>
                        <div class="stat-value">{len(portfolios)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="stats-wrapper">
                    <div class="stat-card">
                        <div class="stat-label">Total Value</div>
                        <div class="stat-value">$0</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # User
        st.markdown("---")
        
        user_initial = st.session_state.get('user_initial', 'U')
        user_name = st.session_state.get('user_name', 'User')
        user_email = st.session_state.get('user_email', 'user@phi.com')
        
        st.markdown(f"""
        <div class="user-box" title="{user_name}">
            <div class="user-avatar">{user_initial}</div>
            <div class="user-info">
                <div class="user-name">{user_name}</div>
                <div class="user-email">{user_email}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
