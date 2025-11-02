# sidebar_collapsible.py
"""
Sidebar optimisée - Gère AI Assistant conversations + Navigation globale
"""

import streamlit as st
from uiconfig import get_theme_colors, toggle_theme
from datetime import datetime

def init_ai_conversations():
    """Initialise conversations AI si sur page AI Assistant"""
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_conversation_id' not in st.session_state:
        create_new_conversation()

def create_new_conversation():
    """Crée nouvelle conversation AI"""
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversations[conv_id] = {
        'id': conv_id,
        'title': 'Nouvelle conversation',
        'created_at': datetime.now(),
        'messages': [],
        'context': []
    }
    st.session_state.current_conversation_id = conv_id
    st.session_state.chat_history = []
    st.session_state.conversation_context = []

def render_ai_conversations_section(collapsed):
    """Section conversations AI (seulement si page AI)"""
    theme = get_theme_colors()
    
    if not collapsed:
        st.markdown('<div class="section-title" >💬 Conversations</div>', unsafe_allow_html=True)
    
    # New button
    new_label = "➕" if collapsed else "➕ Nouvelle"
    if st.button(new_label, use_container_width=True, key="new_conv_btn", type="secondary"):
        create_new_conversation()
        st.rerun()
    
    if not collapsed:
        # Liste conversations (max 5 pour éviter scroll)
        conversations = sorted(
            st.session_state.conversations.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )[:5]
        
        current_id = st.session_state.get('current_conversation_id')
        
        for conv_id, conv in conversations:
            is_current = conv_id == current_id
            
            # Row avec switch et delete
            col1, col2 = st.columns([4, 1])
            
            with col1:
                title = conv['title'][:25] + ('...' if len(conv['title']) > 25 else '')
                icon = "📌" if is_current else "💬"
                
                if st.button(
                    f"{icon} {title}",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    disabled=is_current,
                    type="primary"
                ):
                    # Switch conversation
                    st.session_state.current_conversation_id = conv_id
                    conv_data = st.session_state.conversations[conv_id]
                    st.session_state.chat_history = conv_data['messages']
                    st.session_state.conversation_context = conv_data['context']
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_conv_{conv_id}", disabled=is_current):
                    # Delete conversation
                    del st.session_state.conversations[conv_id]
                    if is_current:
                        if st.session_state.conversations:
                            st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                        else:
                            create_new_conversation()
                    st.rerun()
    else:
        # Mode collapsed: juste le count
        count = len(st.session_state.conversations)
        st.markdown(f'<div style="text-align: center; font-size: 10px; color: {theme["text_secondary"]};">{count}</div>', unsafe_allow_html=True)

def render_ai_status_section(collapsed):
    """Section status AI (seulement si page AI)"""
    theme = get_theme_colors()
    
    if not collapsed:
        st.markdown('<div class="section-title">🔌 Status</div>', unsafe_allow_html=True)
    
    # Check statuses
    mcp_ok = False
    claude_ok = st.secrets.get("ANTHROPIC_API_KEY", "") != ""
    
    try:
        import requests
        MCP_URL = st.secrets.get("MCP_SERVER_URL", "http://localhost:8000")
        response = requests.get(f"{MCP_URL}/health", timeout=1)
        mcp_ok = response.status_code == 200
    except:
        pass
    
    rag_count = 0
    try:
        from knowledge.rag_engine import SimpleRAG
        rag = SimpleRAG()
        stats = rag.get_stats()
        rag_count = stats.get('total_documents', 0)
    except:
        pass
    
    web_ok = False
    try:
        from knowledge.web_search import WebSearchEngine
        web_ok = True
    except:
        pass
    
    # Affichage
    if collapsed:
        # Mode collapsed: juste les icônes colorées
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 16px;">{"🟢" if claude_ok else "🔴"}</div>
            <div style="font-size: 16px;">{"🟢" if mcp_ok else "🔴"}</div>
            <div style="font-size: 16px;">{"🟢" if rag_count > 0 else "🟡"}</div>
            <div style="font-size: 16px;">{"🟢" if web_ok else "🔴"}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Mode expanded: labels + status
        st.markdown(f"**Claude:** {'🟢 Ready' if claude_ok else '🔴 No Key'}")
        st.markdown(f"**MCP:** {'🟢 Online' if mcp_ok else '🔴 Offline'}")
        st.markdown(f"**RAG:** {'🟢' if rag_count > 0 else '🟡'} {rag_count} docs")
        st.markdown(f"**Web:** {'🟢 Active' if web_ok else '🔴 Inactive'}")

def render_sidebar():
    """Sidebar optimisée - Gère tout (nav + AI + user)"""
    theme = get_theme_colors()
    
    # Init
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    collapsed = st.session_state.sidebar_collapsed
    is_logged_in = 'user_id' in st.session_state and st.session_state.user_id
    current_page = st.session_state.get('current_page', 'Dashboard')
    is_ai_page = current_page == 'AI Assistant'
    
    # CSS (identique à avant)
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{
            min-width: {('80px' if collapsed else '280px')} !important;
            max-width: {('80px' if collapsed else '280px')} !important;
            background: {theme['bg_card']} !important;
            border-right: 1px solid {theme['border']} !important;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            width: {('80px' if collapsed else '280px')} !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }}
        
        .sb-logo {{
            text-align: center;
            padding: {('8px' if collapsed else '12px')};
            border-bottom: 1px solid {theme['border']};
            margin-bottom: 6px;
        }}
        
        .logo-phi {{
            font-size: {('20px' if collapsed else '28px')};
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        [data-testid="stSidebar"] button[kind="secondary"] {{
            width: 100%;
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 6px !important;
            padding: 4px !important;
            margin: 4px 0 !important;
            font-size: 14px !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"] {{
            width: 100%;
            background: transparent !important;
            color: {theme['text_secondary']} !important;
            border: none !important;
            border-radius: 6px !important;
            padding: {('6px' if collapsed else '8px 12px')} !important;
            text-align: {('center' if collapsed else 'left')} !important;
            font-size: {('16px' if collapsed else '13px')} !important;
            font-weight: 500 !important;
            margin: 0 0 2px 0 !important;
        }}
        
        [data-testid="stSidebar"] button[kind="primary"]:hover {{
            background: rgba(99, 102, 241, 0.08) !important;
            color: {theme['text_primary']} !important;
        }}
        
        .section-title {{
            font-size: 9px;
            font-weight: 600;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 6px 12px 2px;
            margin-top: 2px;
            {('display: none;' if collapsed else 'display: block;')}
        }}
        
        hr {{
            border: none;
            border-top: 1px solid {theme['border']};
            margin: 4px 0;
        }}
        
        .user-box {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: {('6px' if collapsed else '8px 10px')};
            margin: {('4px 6px' if collapsed else '6px 10px')};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            background: {theme['bg_card']};
            justify-content: {('center' if collapsed else 'flex-start')};
        }}
        
        .user-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 12px;
            flex-shrink: 0;
        }}
        
        .user-info {{
            {('display: none;' if collapsed else 'display: block;')}
            flex: 1;
            min-width: 0;
        }}
        
        .user-name {{
            font-size: 12px;
            font-weight: 600;
            color: {theme['text_primary']};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .user-email {{
            font-size: 9px;
            color: {theme['text_secondary']};
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # 1. LOGO
        if collapsed:
            st.markdown('<div class="sb-logo"><div class="logo-phi">Φ</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-logo"><div class="logo-phi">ΦManager</div></div>', unsafe_allow_html=True)
        
        # 2. TOGGLE
        toggle_icon = "▶" if collapsed else "◀"
        if st.button(toggle_icon, key="toggle_btn", help="Toggle sidebar", type="secondary"):
            st.session_state.sidebar_collapsed = not collapsed
            st.rerun()
        
        # 3. USER BOX (en haut si logged in)
        if is_logged_in:
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
            
            st.markdown("---")
        
        # 4. AI CONVERSATIONS (seulement si page AI)
        if is_ai_page and is_logged_in:
            init_ai_conversations()
            render_ai_conversations_section(collapsed)
            st.markdown("---")
        
        # 5. NAVIGATION (si logged in)
        if is_logged_in:
            st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
            
            pages = [
                ("Dashboard", "🏠"),
                ("Portfolio", "💼"),
                ("Market", "📊"),
                ("AI Assistant", "🤖"),
                ("Pricing", "💎")
            ]
            
            for page_name, icon in pages:
                button_label = icon if collapsed else f"{icon}  {page_name}"
                if st.button(button_label, key=f"nav_{page_name}", use_container_width=True, type="primary"):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
        
        # 6. AI STATUS (seulement si page AI)
        if is_ai_page and is_logged_in:
            render_ai_status_section(collapsed)
            st.markdown("---")
        
        # 7. SETTINGS
        st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
        
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        theme_text = "Light" if st.session_state.theme == "dark" else "Dark"
        theme_label = theme_icon if collapsed else f"{theme_icon}  {theme_text}"
        
        if st.button(theme_label, key="theme_toggle", use_container_width=True, type="primary"):
            toggle_theme()
            st.rerun()
        
        st.markdown("---")
        
        # 8. LOGOUT / LOGIN
        if is_logged_in:
            logout_label = "🚪" if collapsed else "🚪  Logout"
            if st.button(logout_label, key="logout_btn", use_container_width=True, type="secondary"):
                keys_to_keep = ['theme', 'sidebar_collapsed']
                keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.session_state.current_page = 'Login'
                st.rerun()
        else:
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
                st.markdown('<div class="user-box"><div class="user-avatar">?</div></div>', unsafe_allow_html=True)
            
            login_label = "🔑" if collapsed else "🔑  Login"
            if st.button(login_label, key="login_btn", use_container_width=True, type="secondary"):
                st.session_state.current_page = 'Login'
                st.rerun()

__all__ = ['render_sidebar']
