"""
AI Assistant - Version avec Sidebar Optimisée + Conversations Management
"""

import streamlit as st
import anthropic
import json
import requests
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

from uiconfig import get_theme_colors
from dataprovider import yahoo
from pagess.auth import render_auth

try:
    from knowledge.web_search import WebSearchEngine
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

try:
    from knowledge.rag_engine import SimpleRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

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

try: 
    user_id = st.session_state.user_id
    user_name = st.session_state.get('user_name', 'User')
except:
    render_auth()
    st.stop()

ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://localhost:8000")
USE_MCP = st.secrets.get("USE_MCP", True)
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")

_web_search_engine = None
_rag_engine = None
_fed_data = None

def get_web_search_engine():
    global _web_search_engine
    if _web_search_engine is None and WEB_SEARCH_AVAILABLE:
        try:
            _web_search_engine = WebSearchEngine()
        except:
            pass
    return _web_search_engine

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None and RAG_AVAILABLE:
        try:
            _rag_engine = SimpleRAG()
        except:
            pass
    return _rag_engine

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
# KNOWLEDGE BASE (condensé)
# =============================================================================

KNOWLEDGE_BASE = {
    "sharpe": {"title": "Ratio de Sharpe", "content": "📊 Mesure rendement/risque. Formule: (R-Rf)/σ. >2=Excellent, 1-2=Bon, <1=Faible"},
    "var": {"title": "VaR", "content": "📉 Perte max probable. Ex: VaR95%=10k€ → 95% chances perte≤10k€"},
    "sortino": {"title": "Sortino", "content": "📊 Sharpe mais pénalise seulement volatilité négative"},
    "markowitz": {"title": "Markowitz", "content": "📊 Théorie 1952: optimiser rendement/risque via diversification"},
}

# =============================================================================
# MCP + HANDLERS (identiques, pas de changements)
# =============================================================================

def check_mcp_connection() -> bool:
    if not USE_MCP:
        return False
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def execute_mcp_tool(tool_name: str, params: Dict[str, Any], require_confirmation: bool = False):
    if not USE_MCP:
        return None
    try:
        if require_confirmation:
            with st.expander("🔧 Action MCP", expanded=True):
                st.write(f"**Outil:** `{tool_name}`")
                st.json(params)
                col1, col2 = st.columns(2)
                with col1:
                    if not st.button("✅ Confirmer", key=f"mcp_{tool_name}_{hash(str(params))}"):
                        st.info("⏳ En attente...")
                        st.stop()
                with col2:
                    if st.button("❌ Annuler", key=f"mcp_cancel_{tool_name}"):
                        return None
        
        response = requests.post(f"{MCP_SERVER_URL}/execute", json={"tool": tool_name, "params": params}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data")
        return None
    except:
        return None

def handle_education_query(prompt: str) -> str:
    for key, content in KNOWLEDGE_BASE.items():
        if key in prompt.lower():
            return content['content']
    return "❌ Info non trouvée. Essayez: sharpe, var, sortino"

def handle_portfolio_query(prompt: str) -> str:
    try:
        data = execute_mcp_tool("get_portfolios", {"user_id": str(user_id)}, require_confirmation=False)
        if not data or not data.get('portfolios'):
            return "📁 Aucun portfolio"
        
        count = len(data['portfolios'])
        total = sum(p.get('total_amount', 0) for p in data['portfolios'])
        return f"📊 {count} portfolio(s) | Total: ${total:,.2f}"
    except:
        return "❌ Erreur"

def handle_research_query(prompt: str) -> str:
    ticker = extract_ticker_from_prompt(prompt)
    if not ticker:
        return "🔍 Précisez ticker (ex: AAPL)"
    
    try:
        data = yahoo.get_ticker_data(ticker, period='1y')
        info = yahoo.get_ticker_info(ticker)
        if data is None:
            return f"❌ Données indisponibles: {ticker}"
        
        current = float(data['Close'].iloc[-1])
        return f"📊 **{ticker}** | Prix: ${current:.2f}"
    except:
        return f"❌ Erreur: {ticker}"

def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    common = {'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA'}
    prompt_lower = prompt.lower()
    for name, ticker in common.items():
        if name in prompt_lower:
            return ticker
    matches = re.findall(r'\b([A-Z]{2,5})\b', prompt.upper())
    return matches[0] if matches else None

def handle_general_query(prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        return "🤖 Configurez ANTHROPIC_API_KEY"
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        messages = st.session_state.conversation_context[-5:]
        response = client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=1500, messages=messages)
        return response.content[0].text
    except:
        return "⚠️ Erreur IA"

def route_query(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ['mon portfolio', 'mes portfolios']):
        return handle_portfolio_query(prompt)
    elif any(w in p for w in ['recherche', 'analyse', 'action']):
        return handle_research_query(prompt)
    elif any(w in p for w in ['explique', 'qu\'est-ce', 'comment']):
        return handle_education_query(prompt)
    else:
        return handle_general_query(prompt)

def process_message(prompt: str):
    timestamp = datetime.now().isoformat()
    message_id = hashlib.md5(f"{prompt}{timestamp}".encode()).hexdigest()[:16]
    
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    st.session_state.conversation_context.append({'role': 'user', 'content': prompt})
    
    with st.spinner("🤖 Réflexion..."):
        response = route_query(prompt)
    
    st.session_state.chat_history.append({'role': 'assistant', 'content': response, 'message_id': message_id})
    st.session_state.conversation_context.append({'role': 'assistant', 'content': response})
    
    save_current_conversation()
    st.rerun()

# =============================================================================
# SIDEBAR OPTIMISÉE
# =============================================================================

def render_sidebar():
    """Sidebar avec conversations + status + user avatar"""
    theme = get_theme_colors()
    collapsed = st.session_state.get('sidebar_collapsed', False)
    
    with st.sidebar:
        # User Avatar (en haut)
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
            {'<div style="flex: 1;"><div style="font-weight: 600; font-size: 14px;">' + user_name + '</div><div style="font-size: 11px; opacity: 0.7;">AI Assistant</div></div>' if not collapsed else ''}
        </div>
        """
        st.html(avatar_html)
        
        # Conversations Section
        st.markdown("### 💬" + ("" if collapsed else " Conversations"))
        
        # New conversation button
        if st.button("➕" + ("" if collapsed else " Nouvelle"), use_container_width=True, key="new_conv"):
            create_new_conversation()
            st.rerun()
        
        # Conversations list
        if not collapsed:
            conversations = sorted(
                st.session_state.conversations.items(), 
                key=lambda x: x[1]['created_at'], 
                reverse=True
            )
            
            current_id = st.session_state.current_conversation_id
            
            # Container scrollable
            conv_html = '<div style="max-height: 200px; overflow-y: auto; margin-bottom: 1rem;">'
            
            for conv_id, conv in conversations[:10]:  # Max 10 conversations
                is_current = conv_id == current_id
                bg_color = theme['border_hover'] if is_current else theme['bg_card']
                
                conv_html += f"""
                <div style="padding: 0.5rem; margin-bottom: 0.25rem; 
                            background: {bg_color}; border-radius: 6px; 
                            border: 1px solid {theme['border']}; cursor: pointer;"
                     onclick="alert('Use Streamlit button')">
                    <div style="font-size: 12px; font-weight: {'600' if is_current else '400'}; 
                                color: {theme['text_primary']}; margin-bottom: 0.2rem;">
                        {'📌 ' if is_current else '💬 '}{conv['title'][:30]}{'...' if len(conv['title']) > 30 else ''}
                    </div>
                    <div style="font-size: 10px; color: {theme['text_secondary']};">
                        {conv['created_at'].strftime('%d/%m %H:%M')} • {len(conv['messages'])//2} msgs
                    </div>
                </div>
                """
                
                # Buttons for switching/deleting
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"💬 {conv['title'][:25]}", key=f"switch_{conv_id}", use_container_width=True, disabled=is_current):
                        switch_conversation(conv_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}", disabled=is_current):
                        delete_conversation(conv_id)
                        st.rerun()
            
            conv_html += '</div>'
            st.html(conv_html)  # Optionnel: affichage HTML statique
        
        st.divider()
        
        # Status Section (compact)
        st.markdown("### 🔌" + ("" if collapsed else " Status"))
        
        mcp_connected = check_mcp_connection()
        
        status_items = []
        
        if USE_MCP:
            status_items.append(("MCP", "🟢" if mcp_connected else "🔴"))
        
        status_items.append(("Claude", "🟢" if ANTHROPIC_API_KEY else "🔴"))
        
        rag = get_rag_engine()
        if rag:
            stats = rag.get_stats()
            status_items.append(("RAG", f"🟢 {stats['total_documents']}" if not collapsed else "🟢"))
        
        web = get_web_search_engine()
        status_items.append(("Web", "🟢" if web else "🔴"))
        
        # Affichage compact
        for label, status in status_items:
            if collapsed:
                st.markdown(status)
            else:
                st.markdown(f"**{label}:** {status}")
        
        st.divider()
        
        # Feedback Dashboard (compact)
        if FEEDBACK_AVAILABLE and not collapsed:
            with st.expander("📊 Feedback Stats"):
                show_feedback_dashboard()
        
        # Metrics
        if st.session_state.chat_history:
            st.metric("💬" if collapsed else "Messages", len(st.session_state.chat_history))

# =============================================================================
# UI MAIN
# =============================================================================

def render_ai_assistant():
    theme = get_theme_colors()
    
    # Init
    init_conversations()
    if FEEDBACK_AVAILABLE and 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    # Header
    st.html(f"""
    <div style="background: {theme['gradient_primary']}; padding: 1.5rem 2rem; 
                border-radius: 12px; margin-bottom: 1.5rem; 
                box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">
            🤖 Φ AI Assistant
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            RAG + Web + Conversations
        </p>
    </div>
    """)
    
  
    
    # Chat
    if not st.session_state.chat_history:
        render_welcome_screen(theme)
    else:
        render_chat_history()
    
    render_chat_input()

def render_welcome_screen(theme):
    st.html(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']};">Assistant IA Φ</h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Aide pour vos investissements
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Exemples")
    col1, col2 = st.columns(2)
    
    suggestions = [
        ("📊", "Analyse mon portfolio"),
        ("🔍", "Recherche Apple"),
        ("🏗️", "Crée un portfolio AAPL MSFT"),
        ("📈", "Compare Tesla et Ford"),
        ("🎓", "Explique le Sharpe"),
    ]
    
    for idx, (icon, prompt) in enumerate(suggestions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"{icon} {prompt.split()[0]}", key=f"sug_{idx}", use_container_width=True):
                process_message(prompt)

def render_chat_history():
    for msg in st.session_state.chat_history:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                if FEEDBACK_AVAILABLE and 'message_id' in msg:
                    idx = st.session_state.chat_history.index(msg)
                    query = st.session_state.chat_history[idx-1]['content'] if idx > 0 else ""
                    add_feedback_to_chat_message(msg['message_id'], content, query)

def render_chat_input():
    if prompt := st.chat_input("Posez votre question..."):
        process_message(prompt)

if __name__ == "__main__":
    render_ai_assistant()
