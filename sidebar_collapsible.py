# sidebar_collapsible.py
"""
Sidebar collapsible optimisée avec Login/Logout
Compact: tout visible sans scroll
"""

import streamlit as st
from uiconfig import get_theme_colors, toggle_theme


def render_sidebar():
    """Sidebar optimisée avec Login/Logout"""
    theme = get_theme_colors()
    
    # Init collapsed state
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    collapsed = st.session_state.sidebar_collapsed
    #is_logged_in = 'user_id' in st.session_state
    is_logged_in = True
    
    # CSS dynamique optimisé
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
        
        /* Logo - compact */
        .sb-logo {{
            text-align: center;
            padding: {('12px 8px' if collapsed else '16px')};
            border-bottom: 1px solid {theme['border']};
            margin-bottom: 8px;
        }}
        
        .logo-phi {{
            font-size: {('24px' if collapsed else '32px')};
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: font-size 0.3s ease;
        }}
        
        .logo-text {{
            font-size: 16px;
            font-weight: 700;
            color: {theme['text_primary']};
            margin-top: 2px;
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
        /* Toggle button */
        [data-testid="stSidebar"] button[kind="secondary"] {{
            width: 100%;
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            padding: 6px !important;
            margin: 6px 0 !important;
            font-size: 16px !important;
            transition: all 0.2s ease !important;
        }}
        
        [data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background: rgba(99, 102, 241, 0.15) !important;
            border-color: {theme['accent']} !important;
        }}
        
        /* Boutons navigation - compact */
        [data-testid="stSidebar"] button[kind="primary"] {{
            width: 100%;
            background: transparent !important;
            color: {theme['text_secondary']} !important;
            border: none !important;
            border-radius: 8px !important;
            padding: {('10px 8px' if collapsed else '10px 16px')} !important;
            text-align: {('center' if collapsed else 'left')} !important;
            font-size: {('18px' if collapsed else '14px')} !important;
            font-weight: 500 !important;
            margin: 1px 0 !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
            white-space: nowrap !important;
            overflow: hidden !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"]:hover {{
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"]:focus {{
            background: rgba(99, 102, 241, 0.12) !important;
            color: {theme['accent']} !important;
            border-left: 3px solid {theme['accent']} !important;
            box-shadow: none !important;
        }}
        
        /* Section titles - compact */
        .section-title {{
            font-size: 10px;
            font-weight: 600;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: {('6px 8px 3px' if collapsed else '8px 16px 4px')};
            margin-top: 4px;
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
        /* User section - compact */
        .user-box {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: {('8px' if collapsed else '10px 12px')};
            margin: {('8px 6px' if collapsed else '8px 12px')};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            background: {theme['bg_card']};
            justify-content: {('center' if collapsed else 'flex-start')};
        }}
        
        .user-avatar {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 14px;
            flex-shrink: 0;
        }}
        
        .user-info {{
            {('display: none;' if collapsed else 'display: block;')}
            flex: 1;
            min-width: 0;
        }}
        
        .user-name {{
            font-size: 13px;
            font-weight: 600;
            color: {theme['text_primary']};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .user-email {{
            font-size: 10px;
            color: {theme['text_secondary']};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        /* Logout button style */
        .logout-btn {{
            background: rgba(239, 68, 68, 0.1) !important;
            color: #EF4444 !important;
            border: 1px solid rgba(239, 68, 68, 0.2) !important;
        }}
        
        .logout-btn:hover {{
            background: rgba(239, 68, 68, 0.2) !important;
        }}
        
        /* Login button style */
        .login-btn {{
            background: rgba(34, 197, 94, 0.1) !important;
            color: #22C55E !important;
            border: 1px solid rgba(34, 197, 94, 0.2) !important;
        }}
        
        .login-btn:hover {{
            background: rgba(34, 197, 94, 0.2) !important;
        }}
        
        /* Divider - compact */
        hr {{
            border: none;
            border-top: 1px solid {theme['border']};
            margin: 8px {('6px' if collapsed else '0')};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Logo - compact
        st.markdown(f"""
        <div class="sb-logo">
            <div class="logo-phi">Φ</div>
            <div class="logo-text">Manager</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle button
        toggle_icon = "▶" if collapsed else "◀"
        if st.button(toggle_icon, key="toggle_btn", help="Toggle sidebar", type="secondary"):
            st.session_state.sidebar_collapsed = not collapsed
            st.rerun()
        
        # Navigation - seulement si connecté
        if is_logged_in:
            st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
            
            pages = [
                ("Dashboard", "🏠"),
                ("Portfolio", "💼"),
                ("Market", "📊"),
                ("AI Assistant", "🤖")
            ]
            
            for page_name, icon in pages:
                button_label = icon if collapsed else f"{icon}  {page_name}"
                
                if st.button(button_label, key=f"nav_{page_name}", use_container_width=True, type="primary"):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
        
        # Settings
        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
        
        # Theme toggle
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        theme_text = "Light" if st.session_state.theme == "dark" else "Dark"
        theme_label = theme_icon if collapsed else f"{theme_icon}  {theme_text} Mode"
        
        if st.button(theme_label, key="theme_toggle", use_container_width=True, type="primary"):
            toggle_theme()
            st.rerun()
        
        st.markdown("---")
        
        # User section avec Login/Logout
        if is_logged_in:
            # User info
            user_initial = st.session_state.get('username', 'U')[0].upper()
            user_name = st.session_state.get('username', 'User')
            user_email = st.session_state.get('user_email', 'user@pymanager.com')
            
            st.markdown(f"""
            <div class="user-box" title="{user_email}">
                <div class="user-avatar">{user_initial}</div>
                <div class="user-info">
                    <div class="user-name">{user_name}</div>
                    <div class="user-email">{user_email}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Logout button
            logout_label = "🚪" if collapsed else "🚪  Logout"
            if st.button(logout_label, key="logout_btn", use_container_width=True, type="secondary"):
                # Clear session
                keys_to_keep = ['theme', 'sidebar_collapsed']
                keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.session_state.current_page = 'Login'
                st.rerun()
        
        else:
            # Not logged in - show login prompt
            if not collapsed:
                st.markdown(f"""
                <div class="user-box">
                    <div class="user-avatar">?</div>
                    <div class="user-info">
                        <div class="user-name">Guest</div>
                        <div class="user-email">Not logged in</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="user-box" title="Not logged in">
                    <div class="user-avatar">?</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Login button
            login_label = "🔑" if collapsed else "🔑  Login"
            if st.button(login_label, key="login_btn", use_container_width=True, type="secondary"):
                st.session_state.current_page = 'Login'
                st.rerun()
