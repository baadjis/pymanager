# pagess/ai_assistant.py
"""
AI Assistant - Version avec Feedback System Intégré
"""

import streamlit as st
import anthropic
import json
import requests
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

# Imports du projet
from uiconfig import get_theme_colors
from dataprovider import yahoo
from pagess.auth import render_auth

# Import WebSearchEngine et RAG
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

# Import Feedback System
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
    st.warning(f"⚠️ Feedback system non disponible {e}")

# Vérification user
try: 
    user_id = st.session_state.user_id
except:
    render_auth()
    st.stop()

# Configuration
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://localhost:8000")
USE_MCP = st.secrets.get("USE_MCP", True)

# Lazy loading
_web_search_engine = None
_rag_engine = None

def get_web_search_engine():
    global _web_search_engine
    if _web_search_engine is None and WEB_SEARCH_AVAILABLE:
        try:
            _web_search_engine = WebSearchEngine()
        except Exception as e:
            st.error(f"WebSearchEngine error: {e}")
    return _web_search_engine

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None and RAG_AVAILABLE:
        try:
            _rag_engine = SimpleRAG()
        except Exception as e:
            st.error(f"RAG error: {e}")
    return _rag_engine

# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = {
    "sharpe": {
        "title": "Ratio de Sharpe",
        "content": """📊 **Ratio de Sharpe**

**Définition:**
Mesure le rendement ajusté au risque.

**Formule:**
```
Sharpe = (Rendement - Taux sans risque) / Volatilité
```

**Interprétation:**
- **> 2.0** : Excellent ⭐⭐⭐
- **1.0-2.0** : Très bon ⭐⭐
- **0.5-1.0** : Acceptable ⭐
- **< 0.5** : Faible ❌

**Dans PyManager:** Portfolio Details → Analytics"""
    },
    
    "var": {
        "title": "Value at Risk (VaR)",
        "content": """📉 **Value at Risk (VaR)**

**Définition:**
Perte maximale probable sur une période, à un niveau de confiance donné.

**Exemple:**
VaR 95% sur 1 jour = 10,000€
→ 95% de chances que perte ≤ 10,000€

**3 Méthodes:**
1. **Historique:** Données passées
2. **Paramétrique:** Distribution normale
3. **Monte Carlo:** Simulations

**Dans PyManager:** Portfolio Details → Analytics → VaR"""
    },
    
    "sortino": {
        "title": "Ratio de Sortino",
        "content": """📊 **Ratio de Sortino**

**Définition:**
Comme Sharpe, mais pénalise SEULEMENT la volatilité négative.

**Formule:**
```
Sortino = (Rendement - Cible) / Downside deviation
```

**Avantage:** Ne pénalise que les mauvaises surprises."""
    },
    
    "markowitz": {
        "title": "Théorie de Markowitz",
        "content": """📊 **Théorie Moderne du Portfolio (1952)**

**Principe:**
Optimiser rendement/risque via diversification.

**Dans PyManager:**
Portfolio Manager → Build → Markowitz (4 modes: Sharp, Risk, Return, Unsafe)"""
    },
    
    "diversification": {
        "title": "Diversification",
        "content": """🎯 **Diversification**

**Principe:** "Ne pas mettre tous ses œufs dans le même panier"

**Dimensions:**
- Actifs (actions, obligations)
- Secteurs (tech, santé, énergie)
- Géographie (US, Europe, Asie)
- Capitalisation (large, mid, small)

**Vérifier:** Portfolio Details → Sector Allocation"""
    },
}

# =============================================================================
# MCP Functions
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
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json={"tool": tool_name, "params": params},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data")
        return None
            
    except Exception as e:
        st.error(f"❌ MCP Error: {e}")
        return None

# =============================================================================
# EDUCATION HANDLER (avec RAG + Web)
# =============================================================================

def handle_education_query(prompt: str) -> str:
    """Gestion éducation avec RAG + Web Search"""
    
    st.info(f"🔍 Recherche: **{prompt}**")
    
    # 1. Knowledge Base locale
    for key, content in KNOWLEDGE_BASE.items():
        if key in prompt.lower() or content['title'].lower() in prompt.lower():
            st.success("📚 Base locale")
            if ANTHROPIC_API_KEY:
                return enrich_with_claude(content['content'], prompt)
            return content['content']
    
    # 2. RAG Search
    rag = get_rag_engine()
    if rag:
        rag_results = rag.search(prompt, top_k=2, hybrid=True)
        if rag_results:
            st.success(f"📚 RAG: {len(rag_results)} doc(s)")
            context = "\n\n".join([r['text'] for r in rag_results])
            
            if ANTHROPIC_API_KEY:
                return synthesize_with_claude_education(prompt, context, 'RAG')
            else:
                response = f"📚 **Résultats RAG:**\n\n"
                for i, r in enumerate(rag_results, 1):
                    response += f"**{i}. {r['metadata'].get('title', 'Doc')}** (score: {r['score']:.2f})\n"
                    response += f"{r['text'][:300]}...\n\n"
                return response
    
    # 3. Web Search
    search_engine = get_web_search_engine()
    if search_engine:
        with st.spinner("🌐 Recherche web..."):
            web_results = search_engine.search(prompt, sources=['all'], max_results=3)
        
        # Enrichir RAG avec résultats web
        if rag and web_results.get('sources'):
            rag.add_from_web_search(prompt, web_results)
        
        context = build_context_from_web(web_results)
        
        if context:
            if ANTHROPIC_API_KEY:
                return synthesize_with_claude_education(prompt, context, 'Web')
            else:
                return format_web_results(prompt, web_results)
    
    return f"""❌ **Aucune information pour: {prompt}**

**Essayez:**
- Termes plus simples
- En anglais (ex: "Value at Risk")
- Acronymes (VaR, CAPM)

💡 Configurez ANTHROPIC_API_KEY pour des réponses complètes!"""

def build_context_from_web(results: Dict) -> str:
    context = ""
    wiki = results.get('sources', {}).get('wikipedia', {})
    if wiki.get('found'):
        context += f"**Wikipedia:** {wiki['summary']}\n\n"
    
    ddg = results.get('sources', {}).get('duckduckgo', {})
    if ddg.get('results'):
        for r in ddg['results'][:2]:
            context += f"**{r['title']}:** {r['snippet']}\n\n"
    
    return context

def synthesize_with_claude_education(prompt: str, context: str, source: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Contexte ({source}):

{context}

Question: {prompt}

Réponds en français de manière pédagogique avec:
1. 📖 **Définition simple**
2. 💡 **Explication détaillée** avec exemples concrets
3. 📊 **Formules/Calculs** (si applicable)
4. 🎯 **Application pratique** en investissement
5. ⚠️ **Points d'attention**

Sois clair, utilise emojis et exemples chiffrés!"""
            }]
        )
        
        return f"🤖 **Réponse IA** (source: {source})\n\n{response.content[0].text}"
        
    except Exception as e:
        st.error(f"Claude error: {e}")
        return context

def enrich_with_claude(base_content: str, prompt: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Base:

{base_content}

Question: {prompt}

Enrichis avec plus d'exemples et tips pratiques."""
            }]
        )
        
        return f"📚 **Enrichi IA**\n\n{response.content[0].text}"
        
    except:
        return base_content

def format_web_results(concept: str, results: Dict) -> str:
    response = f"📚 **{concept}**\n\n"
    
    wiki = results.get('sources', {}).get('wikipedia', {})
    if wiki.get('found'):
        response += f"### 📖 {wiki['title']}\n\n{wiki['summary']}\n\n[Lire]({wiki['url']})\n\n"
    
    ddg = results.get('sources', {}).get('duckduckgo', {})
    if ddg.get('results'):
        response += "### 🌐 Sources\n\n"
        for i, r in enumerate(ddg['results'][:3], 1):
            response += f"{i}. **{r['title']}**\n{r['snippet']}\n[Source]({r['url']})\n\n"
    
    return response

# =============================================================================
# AUTRES HANDLERS (Portfolio, Research, etc.)
# =============================================================================

def handle_portfolio_query(prompt: str) -> str:
    """Analyse portfolio via MCP"""
    try:
        data = execute_mcp_tool("get_portfolios", {"user_id": str(user_id)}, require_confirmation=False)
        
        if not data:
            return "❌ Impossible de récupérer vos portfolios."
        
        portfolios = data.get('portfolios', [])
        
        if not portfolios:
            return """📁 **Aucun portfolio**

Créez-en un avec:
"Crée un portfolio growth avec AAPL, MSFT, GOOGL" """
        
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Données:
{json.dumps(data, indent=2)}

Question: {prompt}

Fournis analyse avec:
1. 📊 Vue d'ensemble
2. 💪 Points forts
3. ⚠️ À améliorer
4. 🎯 Recommandations

Sois actionnable!"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback
        total = data.get('total_value', 0)
        count = data.get('count', 0)
        
        response = f"""📊 **Analyse Portfolio**

**Vue d'ensemble:**
- Portfolios: **{count}**
- Valeur totale: **${total:,.2f}**

**Vos Portfolios:**

"""
        
        for idx, pf in enumerate(portfolios, 1):
            name = pf.get('name', f'Portfolio {idx}')
            value = pf.get('total_amount', 0)
            model = pf.get('model', 'N/A')
            
            response += f"**{idx}. {name}**\n- Valeur: ${value:,.2f}\n- Modèle: {model.title()}\n\n"
        
        return response
        
    except Exception as e:
        return f"❌ Erreur: {e}"

def handle_research_query(prompt: str) -> str:
    """Recherche entreprise"""
    ticker = extract_ticker_from_prompt(prompt)
    
    if not ticker:
        return """🔍 **Recherche d'entreprise**

Précisez l'entreprise ou ticker.

**Exemples:**
- "Recherche Apple"
- "Analyse TSLA"
"""
    
    try:
        with st.spinner(f"📊 Données {ticker}..."):
            data = yahoo.get_ticker_data(ticker, period='1y')
            info = yahoo.get_ticker_info(ticker)
            
            if data is None or data.empty:
                return f"❌ Données indisponibles pour {ticker}"
        
        current = float(data['Close'].iloc[-1])
        start = float(data['Close'].iloc[0])
        ytd = ((current - start) / start) * 100
        
        context = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'price': current,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'N/A'),
            'ytd_return': ytd,
            'dividend': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
        
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyse:
{json.dumps(context, indent=2)}

Question: {prompt}

Fournis:
1. 🏢 Vue d'ensemble
2. 📊 Analyse financière
3. 📈 Performance
4. ⚖️ Achat/Vente
5. ⚠️ Risques
6. 🎯 Recommandation

Sois précis!"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback
        return f"""🔍 **{context['name']}**

**{ticker}** | {context['sector']}

**Métriques:**
- Prix: ${context['price']:.2f}
- Cap: ${context['market_cap']/1e9:.1f}B
- P/E: {context['pe_ratio']:.2f}
- Div: {context['dividend']:.2f}%
- YTD: {context['ytd_return']:.2f}%

**Évaluation:**
- Valorisation: {'🔴 Chère' if context['pe_ratio'] > 25 else '🟡 OK' if context['pe_ratio'] > 15 else '🟢 Attractive'}
- Perf: {'🟢 Forte' if context['ytd_return'] > 15 else '🟡 Modérée' if context['ytd_return'] > 0 else '🔴 Faible'}"""
        
    except Exception as e:
        return f"❌ Erreur: {e}"

def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    """Extrait ticker (simplifié)"""
    common = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META',
        'nvidia': 'NVDA'
    }
    
    prompt_lower = prompt.lower()
    for name, ticker in common.items():
        if name in prompt_lower:
            return ticker
    
    # Pattern matching
    matches = re.findall(r'\b([A-Z]{2,5})\b', prompt.upper())
    if matches and matches[0] not in ['US', 'AI', 'ML']:
        return matches[0]
    
    return None

def handle_general_query(prompt: str) -> str:
    """Requêtes générales"""
    if not ANTHROPIC_API_KEY:
        return """🤖 **AI Assistant**

Configurez ANTHROPIC_API_KEY pour réponses complètes!"""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        messages = st.session_state.conversation_context[-5:]
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=messages,
            system="Tu es conseiller financier expert pour PyManager. Sois précis et actionnable."
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Erreur: {e}"

# =============================================================================
# ROUTING
# =============================================================================

def route_query(prompt: str) -> str:
    """Route vers handler approprié"""
    p = prompt.lower()
    
    if any(w in p for w in ['mon portfolio', 'mes portfolios']):
        return handle_portfolio_query(prompt)
    elif any(w in p for w in ['recherche', 'analyse', 'action', 'entreprise']):
        return handle_research_query(prompt)
    elif any(w in p for w in ['explique', 'qu\'est-ce', 'comment', 'apprendre']):
        return handle_education_query(prompt)
    else:
        return handle_general_query(prompt)

# =============================================================================
# MESSAGE PROCESSING avec FEEDBACK
# =============================================================================

def process_message(prompt: str):
    """Traite message avec feedback intégré"""
    
    # Générer message ID unique
    timestamp = datetime.now().isoformat()
    message_id = hashlib.md5(f"{prompt}{timestamp}".encode()).hexdigest()[:16]
    
    # Ajouter à historique
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    st.session_state.conversation_context.append({'role': 'user', 'content': prompt})
    
    # Traiter
    with st.spinner("🤖 Réflexion..."):
        response = route_query(prompt)
    
    # Ajouter réponse
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'message_id': message_id  # Stocker l'ID
    })
    st.session_state.conversation_context.append({'role': 'assistant', 'content': response})
    
    st.rerun()

# =============================================================================
# UI RENDERING
# =============================================================================

def render_ai_assistant():
    """Point d'entrée principal"""
    theme = get_theme_colors()
    
    # Header
    st.html(f"""
    <div style="
        background: {theme['gradient_primary']};
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">
            🤖 Φ AI Assistant
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            RAG + Web Search + Feedback System
        </p>
    </div>
    """)
    
    # Initialiser session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    if FEEDBACK_AVAILABLE and 'feedback_tracker' not in st.session_state:
        st.session_state.feedback_tracker = FeedbackTracker()
    
    # Sidebar status
    mcp_connected = check_mcp_connection()
    
    with st.sidebar:
        collapsed=st.session_state.sidebar_collapsed 
        st.markdown("### 🤖 Status")
        on_icon ="🟢"
        off_icon = "🔴"
        claude_text= " Ready" if ANTHROPIC_API_KEY else " No Key"
        mcp_text=" Online" if mcp_connected else " Offline" 
        web = get_web_search_engine()
        web_text=' Active' if web else ' Inactive'
        button_text =f"🗑️" if collapsed else f"🗑️ Nouvelle conv"
        if collapsed:
               mcp_text=""
               claude_text=""
               web_text=""
        if USE_MCP:
            
            
            
            mcp_status = on_icon + mcp_text if mcp_connected  else off_icon + mcp_text
            st.markdown(f"***MCP:*** {mcp_status}")
        
        #claude_status = on_icon + claude_text if ANTHROPIC_API_KEY else off_icon + claude_text
        #st.markdown(f"***claude***: {claude_status}")
        
        rag = get_rag_engine()
        if rag:
            stats = rag.get_stats()
            rag_text= f"{stats['total_documents']}" if collapsed else  f"{stats['total_documents']} docs"
            st.markdown(f"**RAG:** 🟢 {rag_text}")
        
      
        
        st.markdown(f"**Web:** {on_icon + web_text if web else off_icon + web_text }")
        
        st.divider()
        
        # Feedback Dashboard
        #if FEEDBACK_AVAILABLE:
            #show_feedback_dashboard()
        
        if st.button(button_text, use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.rerun()
        
        if st.session_state.chat_history:
            st.metric("Messages", len(st.session_state.chat_history))
    
    # Chat UI
    if not st.session_state.chat_history:
        render_welcome_screen(theme)
    else:
        render_chat_history()
    
    render_chat_input()

def render_welcome_screen(theme):
    """Écran d'accueil"""
    st.html(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']};">Assistant IA Φ</h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Je peux vous aider avec vos investissements
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Exemples")
    
    col1, col2 = st.columns(2)
    
    suggestions = [
        ("📊", "Analyse", "Analyse mon portfolio"),
        ("🔍", "Recherche", "Recherche Apple"),
        ("🏗️", "Créer", "Crée un portfolio avec AAPL, MSFT"),
        ("📈", "Comparer", "Compare Tesla et Ford"),
        ("🎓", "Apprendre", "Explique le ratio de Sharpe"),
    ]
    
    for idx, (icon, title, prompt) in enumerate(suggestions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"{icon} {title}", key=f"sug_{idx}", use_container_width=True):
                process_message(prompt)

def render_chat_history():
    """Affiche l'historique avec feedback"""
    for msg in st.session_state.chat_history:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                # Ajouter feedback si disponible
                if FEEDBACK_AVAILABLE and 'message_id' in msg:
                    # Récupérer query depuis historique
                    idx = st.session_state.chat_history.index(msg)
                    query = st.session_state.chat_history[idx-1]['content'] if idx > 0 else ""
                    
                    add_feedback_to_chat_message(
                        msg['message_id'],
                        content,
                        query
                    )

def render_chat_input():
    """Zone de saisie"""
    if prompt := st.chat_input("Posez votre question..."):
        process_message(prompt)


if __name__ == "__main__":
    render_ai_assistant()
