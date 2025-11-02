"""
AI Assistant - Version Optimisée avec MCP v4.0
================================================
✅ Support complet MCP v4.0 (15 tools)
✅ Market Intelligence (sectors, sentiment, quantum, semiconductors)
✅ Backtesting & Predictions
✅ RAG + Web Search
✅ Conversations Management
✅ Feedback System
"""

import streamlit as st
import anthropic
import json
import requests
import sys
import re
import hashlib

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from functools import lru_cache
import pickle
from uiconfig import get_theme_colors
from dataprovider import yahoo
from pagess.auth import render_auth
try: 
    user_id = st.session_state.user_id
    user_name = st.session_state.get('user_name', 'User')
except:
    render_auth()
    st.stop()

# Vérification user
try:
  from .handlers import ( route_query,WEB_SEARCH_AVAILABLE,  
       RAG_AVAILABLE ,KNOWLEDGE_ENHANCED ,RAG_AVAILABLE,check_mcp_connection,USE_MCP, 
       ANTHROPIC_API_KEY ,get_rag_engine,_rag_engine,_web_search ,_fed_data ,get_web_search,get_fed_data)

except Exception as e:
       st.write(e)


try:
    from .ai_assistant_feedback import (
        FeedbackTracker,
        add_feedback_to_chat_message,
        track_user_action,
        show_feedback_dashboard
    )
    FEEDBACK_AVAILABLE = True
except Exception as e:
    FEEDBACK_AVAILABLE = False




# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

def init_conversations():
    """Initialise le système de conversations"""
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_conversation_id' not in st.session_state:
        create_new_conversation()

def create_new_conversation():
    """Crée nouvelle conversation"""
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

def get_current_conversation():
    """Récupère conversation courante"""
    conv_id = st.session_state.get('current_conversation_id')
    return st.session_state.conversations.get(conv_id)

def switch_conversation(conv_id: str):
    """Bascule vers une conversation"""
    if conv_id in st.session_state.conversations:
        st.session_state.current_conversation_id = conv_id
        conv = st.session_state.conversations[conv_id]
        st.session_state.chat_history = conv['messages']
        st.session_state.conversation_context = conv['context']

def save_current_conversation():
    """Sauvegarde conversation courante"""
    conv = get_current_conversation()
    if conv:
        conv['messages'] = st.session_state.chat_history
        conv['context'] = st.session_state.conversation_context
        
        # Auto-titre si premier message
        if len(conv['messages']) == 2 and conv['title'] == 'Nouvelle conversation':
            first_msg = conv['messages'][0]['content']
            conv['title'] = first_msg[:40] + ('...' if len(first_msg) > 40 else '')

def delete_conversation(conv_id: str):
    """Supprime une conversation"""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            if st.session_state.conversations:
                st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
            else:
                create_new_conversation()




# =============================================================================
# PROCESS MESSAGE
# =============================================================================

def process_message(prompt: str):
    """Traiter et router un message utilisateur"""
    timestamp = datetime.now().isoformat()
    message_id = hashlib.md5(f"{prompt}{timestamp}".encode()).hexdigest()[:16]
    
    # Ajouter message utilisateur
    st.session_state.chat_history.append({
        'role': 'user',
        'content': prompt,
        'timestamp': timestamp
    })
    st.session_state.conversation_context.append({
        'role': 'user',
        'content': prompt
    })
    
    # Traiter avec spinner
    with st.spinner("🤖 Analyse en cours..."):
        response = route_query(prompt)
    
    # Ajouter réponse
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'message_id': message_id,
        'timestamp': timestamp
    })
    st.session_state.conversation_context.append({
        'role': 'assistant',
        'content': response
    })
    
    # Sauvegarder conversation
    save_current_conversation()
    
    # Track action si feedback disponible
    if FEEDBACK_AVAILABLE:
        track_user_action('message_sent', {'prompt_length': len(prompt)})
    
    # Rerun pour afficher
    st.rerun()

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Sidebar optimisée avec conversations + status"""
    theme = get_theme_colors()
    collapsed = st.session_state.get('sidebar_collapsed', False)
    
    with st.sidebar:
        # User Avatar
        initials = ''.join([n[0].upper() for n in user_name.split()[:2]])
        avatar_html = f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem; 
                    background: {theme['bg_card']}; border-radius: 8px; margin-bottom: 1rem;">
            <div style="width: 36px; height: 36px; border-radius: 50%; 
                        background: {theme['gradient_primary']}; display: flex; 
                        align-items: center; justify-content: center; font-weight: 700; 
                        color: white; font-size: 14px;">
                {initials}
            </div>
            {'<div style="flex: 1;"><div style="font-weight: 600; font-size: 14px;">' + user_name + '</div><div style="font-size: 11px; opacity: 0.7;">AI Assistant v4.0</div></div>' if not collapsed else ''}
        </div>
        """
        st.html(avatar_html)
        
        # Conversations
        st.markdown("### 💬" + ("" if collapsed else " Conversations"))
        
        if st.button("➕" + ("" if collapsed else " Nouvelle"), use_container_width=True, key="new_conv"):
            create_new_conversation()
            st.rerun()
        
        if not collapsed:
            conversations = sorted(
                st.session_state.conversations.items(),
                key=lambda x: x[1]['created_at'],
                reverse=True
            )
            
            current_id = st.session_state.current_conversation_id
            
            for conv_id, conv in conversations[:10]:
                is_current = conv_id == current_id
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    label = f"{'📌' if is_current else '💬'} {conv['title'][:25]}"
                    if st.button(label, key=f"switch_{conv_id}", use_container_width=True, disabled=is_current):
                        switch_conversation(conv_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}", disabled=is_current):
                        delete_conversation(conv_id)
                        st.rerun()
        
        st.divider()
        
        # Status MCP v4.0
        st.markdown("### 🔌" + ("" if collapsed else " Status"))
        
        mcp_connected = check_mcp_connection()
        
        if USE_MCP:
            status = "🟢" if mcp_connected else "🔴"
            st.markdown(f"**MCP v4.0:** {status}")
            
            if mcp_connected and not collapsed:
                # Afficher nombre de tools disponibles
                tools = get_mcp_tools()
                st.caption(f"📊 {len(tools)} tools disponibles")
        
        claude_status = "🟢" if ANTHROPIC_API_KEY else "🔴"
        st.markdown(f"**Claude AI:** {claude_status}")
        
        # RAG & Web
        rag = get_rag_engine()
        if rag and not collapsed:
            stats = rag.get_stats()
            st.markdown(f"**RAG:** 🟢 ({stats.get('total_documents', 0)} docs)")
        
        web = get_web_search()
        if not collapsed:
            st.markdown(f"**Web Search:** {'🟢' if web else '🔴'}")
        
        st.divider()
        
        # Feedback
        if FEEDBACK_AVAILABLE and not collapsed:
            with st.expander("📊 Feedback Stats"):
                show_feedback_dashboard()
        
        # Metrics
        if st.session_state.chat_history:
            msg_count = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
            st.metric("💬" if collapsed else "Messages", msg_count)

# =============================================================================
# UI MAIN
# =============================================================================

def render_ai_assistant():
    """Interface principale AI Assistant"""
    theme = get_theme_colors()
    
    # Init
    init_conversations()
    if FEEDBACK_AVAILABLE and 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    # Render sidebar
    render_sidebar()
    
    # Header
    st.html(f"""
    <div style="background: {theme['gradient_primary']}; padding: 1.5rem 2rem; 
                border-radius: 12px; margin-bottom: 1.5rem; 
                box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">
            🤖 Φ AI Assistant v4.0
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            Portfolio · Market Intelligence · Backtesting · Predictions
        </p>
    </div>
    """)
    
    # Chat area
    if not st.session_state.chat_history:
        render_welcome_screen(theme)
    else:
        render_chat_history()
    
    render_chat_input()

def render_welcome_screen(theme):
    """Écran d'accueil avec suggestions"""
    st.html(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']};">Assistant IA Φ v4.0</h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Portfolio Management · Market Intelligence · Backtesting · AI Predictions
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Suggestions (Nouveautés v4.0)")
    
    col1, col2, col3 = st.columns(3)
    
    # Suggestions catégorisées
    suggestions = [
        # Portfolio
        ("📊", "Analyse mon portfolio", 0),
        ("💰", "Mes portfolios", 0),
        ("📉", "Risque de mon portfolio", 0),
        
        # Market Intelligence (NEW)
        ("🌍", "Vue marché US", 1),
        ("🔬", "Secteur semiconductors", 1),
        ("⚛️", "Analyse quantum computing", 1),
        
        # Research & Analysis (NEW)
        ("💹", "Sentiment NVDA", 2),
        ("⚖️", "Compare AAPL MSFT", 2),
        ("🔍", "Recherche Tesla", 2),
        
        # Advanced (NEW)
        ("🧪", "Backtest mon portfolio", 0),
        ("🔮", "Prédis performance 3 mois", 0),
        ("🎓", "Explique Sharpe", 0),
    ]
    
    for icon, prompt, col_idx in suggestions:
        target_col = col1 if col_idx == 0 else col2 if col_idx == 1 else col3
        with target_col:
            if st.button(f"{icon} {prompt}", key=f"sug_{hash(prompt)}", use_container_width=True):
                process_message(prompt)

def render_chat_history():
    """Afficher historique de chat avec feedback"""
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                # Feedback
                if FEEDBACK_AVAILABLE and 'message_id' in msg:
                    query = st.session_state.chat_history[idx-1]['content'] if idx > 0 else ""
                    add_feedback_to_chat_message(msg['message_id'], content, query)

def render_chat_input():
    """Input chat avec exemples"""
    if prompt := st.chat_input("Posez votre question... (ex: 'Analyse le marché US', 'Secteur quantum', 'Backtest mon portfolio')"):
        process_message(prompt)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    render_ai_assistant()
