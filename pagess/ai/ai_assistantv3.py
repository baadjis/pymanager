# pagess/ai_assistant.py
"""
AI Assistant - Version Enrichie
Architecture complète:
- MCP pour données internes
- Claude AI pour analyses
- Yahoo Finance pour données de marché
- RAG pour documents locaux
- Web Search (DuckDuckGo + Wikipedia)
- FRED pour données économiques
"""

import streamlit as st
import anthropic
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import sys
from pathlib import Path
import yfinance as yf
from functools import lru_cache
import pickle



# Add knowledge module to path
sys.path.append(str(Path(__file__).parent.parent))

# Imports du projet
from uiconfig import get_theme_colors
from dataprovider import yahoo
from pagess.auth import render_auth

# Imports knowledge
try:
    from knowledge.rag_engine import SimpleRAG
    from knowledge.web_search import WebSearchEngine
    from knowledge.fed_data import FREDDataProvider
    KNOWLEDGE_ENHANCED = True
except ImportError:
    KNOWLEDGE_ENHANCED = False
    st.warning("⚠️ Knowledge modules not available. Install: pip install duckduckgo-search wikipedia-api fredapi sentence-transformers")

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
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")

# Initialize knowledge engines (lazy loading)
_rag_engine = None
_web_search = None
_fed_data = None




# =============================================================================
# TICKER CACHE PERSISTANT
# =============================================================================

CACHE_FILE = Path("data/ticker_cache.pkl")
CACHE_FILE.parent.mkdir(exist_ok=True)

def load_ticker_cache() -> dict:
    """Charge le cache de tickers depuis le disque"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_ticker_cache(cache: dict):
    """Sauvegarde le cache sur disque"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        st.warning(f"⚠️ Impossible de sauvegarder le cache: {e}")

# Initialiser le cache global
if 'ticker_cache' not in st.session_state:
    st.session_state.ticker_cache = load_ticker_cache()



def get_rag_engine():
    """Lazy load RAG engine"""
    global _rag_engine
    if _rag_engine is None and KNOWLEDGE_ENHANCED:
        try:
            _rag_engine = SimpleRAG()
        except Exception as e:
            st.error(f"Failed to load RAG: {e}")
    return _rag_engine

def get_web_search():
    """Lazy load web search"""
    global _web_search
    if _web_search is None and KNOWLEDGE_ENHANCED:
        try:
            _web_search = WebSearchEngine()
        except Exception as e:
            st.error(f"Failed to load web search: {e}")
    return _web_search

def get_fed_data():
    """Lazy load FRED data"""
    global _fed_data
    if _fed_data is None and KNOWLEDGE_ENHANCED and FRED_API_KEY:
        try:
            _fed_data = FREDDataProvider(FRED_API_KEY)
        except Exception as e:
            st.error(f"Failed to load FRED: {e}")
    return _fed_data

# =============================================================================
# KNOWLEDGE BASE (Hardcodé - Fallback)
# =============================================================================

KNOWLEDGE_BASE = {
    "sharpe": {
        "title": "Ratio de Sharpe",
        "content": """📊 **Ratio de Sharpe**

**Définition:**
Le ratio de Sharpe mesure le rendement ajusté au risque.

**Formule:**
```
Ratio de Sharpe = (Rendement - Taux sans risque) / Écart-type
```

**Interprétation:**
- **> 2.0** : Excellent ⭐⭐⭐
- **1.0-2.0** : Très bon ⭐⭐
- **0.5-1.0** : Acceptable ⭐
- **< 0.5** : Faible ❌"""
    },
    
    "diversification": {
        "title": "Diversification",
        "content": """🎯 **Diversification du Portfolio**

**Principe:** "Ne mettez pas tous vos œufs dans le même panier"

**Pourquoi?**
- 📉 Réduit le risque global
- 📊 Lisse les rendements
- 🛡️ Protection contre les chocs"""
    },
    
    "markowitz": {
        "title": "Théorie de Markowitz",
        "content": """📊 **Théorie Moderne du Portfolio (Markowitz)**

Développée par Harry Markowitz en 1952, optimise le ratio risque/rendement."""
    }
}
def search_ticker_with_yfinance(company_name: str) -> Optional[str]:
    """
    Recherche un ticker via yfinance search
    Plus rapide et plus fiable que le web scraping
    """
    try:
        # Nettoyer le nom
        company_name = company_name.strip().lower()
        
        # Essayer la recherche yfinance
        import yfinance as yf
        
        # yfinance Ticker peut prendre un symbole et vérifier s'il existe
        # Essayer quelques variations communes
        variations = [
            company_name.upper(),
            company_name.upper() + '.PA',  # Paris
            company_name.upper() + '.L',   # London
            company_name.upper() + '.DE',  # Germany
        ]
        
        for variant in variations:
            try:
                ticker = yf.Ticker(variant)
                info = ticker.info
                
                # Vérifier si le ticker est valide
                if info and 'symbol' in info and info.get('longName'):
                    st.success(f"✅ Ticker trouvé: **{info['symbol']}** = {info['longName']}")
                    return info['symbol']
            except:
                continue
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Erreur recherche yfinance: {e}")
        return None

def search_ticker_with_web(company_name: str) -> Optional[str]:
    """
    Recherche de ticker via web search (DuckDuckGo)
    Fallback si yfinance échoue
    """
    if not KNOWLEDGE_ENHANCED:
        return None
    
    try:
        web_search = get_web_search()
        if not web_search:
            return None
        
        # Recherche optimisée
        query = f"{company_name} stock ticker symbol"
        results = web_search.search(query, sources=['duckduckgo'], max_results=3)
        
        if 'duckduckgo' in results['sources']:
            ddg = results['sources']['duckduckgo']
            if ddg.get('results'):
                # Parser les résultats pour trouver le ticker
                for result in ddg['results']:
                    text = (result.get('title', '') + ' ' + result.get('snippet', '')).upper()
                    
                    # Pattern pour extraire ticker (ex: "LVMH (MC.PA)")
                    patterns = [
                        r'\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)',  # (TICKER.EX)
                        r'ticker:\s*([A-Z]{1,5}(?:\.[A-Z]{1,2})?)',  # ticker: AAPL
                        r'symbol:\s*([A-Z]{1,5}(?:\.[A-Z]{1,2})?)',  # symbol: MSFT
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            ticker = matches[0]
                            # Valider avec yfinance
                            if validate_ticker(ticker):
                                st.info(f"🌐 Ticker trouvé via web: **{ticker}**")
                                return ticker
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Erreur web search: {e}")
        return None

def search_ticker_with_claude(company_name: str) -> Optional[str]:
    """
    Utilise Claude AI pour trouver le ticker
    Dernier recours si tout le reste échoue
    """
    if not ANTHROPIC_API_KEY:
        return None
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Quelle est le ticker boursier (stock symbol) pour l'entreprise: {company_name}

Réponds UNIQUEMENT avec le ticker en majuscules (ex: AAPL, MSFT, MC.PA).
Si c'est une entreprise française/européenne, inclus le suffixe de bourse (.PA pour Paris, .L pour Londres, etc.)
Si tu ne sais pas, réponds "UNKNOWN".

Ticker:"""
            }]
        )
        
        ticker = response.content[0].text.strip().upper()
        
        # Nettoyer la réponse
        ticker = re.sub(r'[^A-Z\.]', '', ticker)
        
        if ticker and ticker != "UNKNOWN" and validate_ticker(ticker):
            st.info(f"🤖 Ticker trouvé via Claude AI: **{ticker}**")
            return ticker
        
        return None
        
    except Exception as e:
        st.warning(f"⚠️ Erreur Claude AI: {e}")
        return None

def validate_ticker(ticker: str) -> bool:
    """
    Valide qu'un ticker existe vraiment en essayant de récupérer des données
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        
        # Vérifier qu'on a des données minimales
        return info and ('symbol' in info or 'shortName' in info or 'longName' in info)
    except:
        return False

def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    """
    Extraction intelligente de ticker avec cache persistant
    
    Pipeline:
    1. Vérifier le cache en mémoire
    2. Pattern matching direct (tickers explicites)
    3. Recherche yfinance (rapide)
    4. Recherche web (DuckDuckGo)
    5. Claude AI (dernier recours)
    6. Sauvegarder dans le cache
    """
    
    # Normaliser le prompt
    prompt_lower = prompt.lower().strip()
    
    # 1. CACHE - Vérifier si on connaît déjà ce nom
    cache = st.session_state.ticker_cache
    
    for company_name, ticker in cache.items():
        if company_name in prompt_lower:
            # Message discret avec toast
            st.toast(f"💾 {company_name.title()} = {ticker}", icon="✅")
            return ticker
    
    # 2. TICKERS EXPLICITES - Pattern matching
    # Tickers communs hardcodés
    COMMON_TICKERS = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE',
        'lvmh': 'MC.PA', 'total': 'TTE.PA', 'l\'oreal': 'OR.PA', 'loreal': 'OR.PA',
        'sanofi': 'SAN.PA', 'bnp': 'BNP.PA', 'airbus': 'AIR.PA',
        'hermes': 'RMS.PA', 'hermès': 'RMS.PA', 'dior': 'CDI.PA',
    }
    
    for name, ticker in COMMON_TICKERS.items():
        if name in prompt_lower:
            # Ajouter au cache
            cache[name] = ticker
            st.session_state.ticker_cache = cache
            save_ticker_cache(cache)
            
            # Message discret
            return ticker
    
    # Pattern pour ticker explicite dans le texte
    patterns = [
        r'\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)',      # $AAPL ou $MC.PA
        r'\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)',    # (AAPL)
        r'\b([A-Z]{2,5})\b',                      # AAPL
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, prompt.upper())
        if matches:
            ticker = matches[0]
            # Éviter les faux positifs
            if ticker not in ['US', 'AI', 'ML', 'RL', 'API', 'PDF', 'CEO', 'CFO', 'YTD']:
                if validate_ticker(ticker):
                    return ticker
    
    # 3. EXTRACTION NOM D'ENTREPRISE
    company_patterns = [
        r'(?:recherche|analyse|action|entreprise|société)\s+(?:l\'|la |le |les )?([a-zéèêàâùûôîïç\s\'-]+?)(?:\s+et|\s+pour|\s+stock|$)',
        r'(?:ticker|symbole|code)\s+(?:de |pour |d\')(?:l\'|la |le )?([a-zéèêàâùûôîïç\s\'-]+?)(?:\s|$)',
        r'\b([A-Z][a-zéèêàâùûôîïç]+(?:\s+[A-Z][a-zéèêàâùûôîïç]+)*)\s+(?:stock|action)',
    ]
    
    company_name = None
    for pattern in company_patterns:
        matches = re.search(pattern, prompt, re.IGNORECASE)
        if matches:
            company_name = matches.group(1).strip()
            if len(company_name) > 2:
                break
    
    if not company_name:
        # Dernière tentative: prendre le mot le plus long qui ressemble à un nom
        words = re.findall(r'\b[A-Z][a-zéèêàâùûôîïç]+\b', prompt)
        if words:
            company_name = max(words, key=len)
    
    if not company_name or len(company_name) <= 2:
        return None
    
    st.info(f"🔍 Recherche du ticker pour: **{company_name}**")
    
    # 4. RECHERCHE YFINANCE
    with st.spinner(f"📊 Recherche {company_name} via Yahoo Finance..."):
        ticker = search_ticker_with_yfinance(company_name)
        if ticker:
            # Sauvegarder dans le cache
            cache[company_name.lower()] = ticker
            st.session_state.ticker_cache = cache
            save_ticker_cache(cache)
            return ticker
    
    # 5. RECHERCHE WEB
    with st.spinner(f"🌐 Recherche {company_name} sur le web..."):
        ticker = search_ticker_with_web(company_name)
        if ticker:
            # Sauvegarder dans le cache
            cache[company_name.lower()] = ticker
            st.session_state.ticker_cache = cache
            save_ticker_cache(cache)
            return ticker
    
    # 6. CLAUDE AI (dernier recours)
    with st.spinner(f"🤖 Demande à Claude AI pour {company_name}..."):
        ticker = search_ticker_with_claude(company_name)
        if ticker:
            # Sauvegarder dans le cache
            cache[company_name.lower()] = ticker
            st.session_state.ticker_cache = cache
            save_ticker_cache(cache)
            return ticker
    
    # Échec total
    st.error(f"❌ Impossible de trouver le ticker pour: **{company_name}**")
    st.info("💡 Essayez avec le ticker exact (ex: AAPL, MC.PA, MSFT)")
    
    return None

# =============================================================================

# =============================================================================
# MCP Functions
# =============================================================================

def check_mcp_connection() -> bool:
    """Vérifie la connexion au serveur MCP"""
    if not USE_MCP:
        return False
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def execute_mcp_tool(tool_name: str, params: Dict[str, Any], require_confirmation: bool = False) -> Optional[Any]:
    """Exécute un outil MCP"""
    if not USE_MCP:
        return None
    
    try:
        if require_confirmation:
            with st.expander("🔧 Action MCP", expanded=True):
                st.write(f"**Outil:** `{tool_name}`")
                st.json(params)
                
                col1, col2 = st.columns(2)
                with col1:
                    if not st.button("✅ Exécuter", key=f"mcp_exec_{tool_name}_{hash(str(params))}"):
                        st.info("⏳ En attente de confirmation...")
                        st.stop()
                with col2:
                    if st.button("❌ Annuler", key=f"mcp_cancel_{tool_name}_{hash(str(params))}"):
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
        st.error(f"❌ Erreur MCP: {str(e)}")
        return None

# =============================================================================
# Enhanced Knowledge Search
# =============================================================================

def search_knowledge(query: str) -> str:
    """
    Pipeline de recherche intelligent:
    1. Knowledge Base (hardcodé) - Instantané
    2. RAG (documents locaux) - Rapide
    3. Web Search (DuckDuckGo + Wikipedia) - Moyen
    4. Synthèse avec Claude - Final
    """
    
    # 1. Chercher dans KB hardcodée
    """for key, content in KNOWLEDGE_BASE.items():
        if key in query.lower() or content['title'].lower() in query.lower():
            st.info("📚 Trouvé dans la knowledge base locale")
            return content['content']
    
    # 2. Chercher dans RAG
    rag = get_rag_engine()
    if rag:
        rag_results = rag.search(query, top_k=3, min_score=0.4)
        if rag_results:
            st.info(f"📄 Trouvé {len(rag_results)} document(s) pertinent(s) dans RAG")
            
            context = "\n\n".join([
                f"**Source: {r['metadata'].get('title', 'Document')}**\n{r['text']}"
                for r in rag_results
            ])
            
            # Synthétiser avec Claude
            if ANTHROPIC_API_KEY:
                return synthesize_with_claude(query, context, source="RAG")
            else:
                return f"📄 **Documents trouvés:**\n\n{context}"
     """
    
    # 3. Web Search
    if KNOWLEDGE_ENHANCED:
        web_search = get_web_search()
        if web_search:
            st.info(f"🌐 Recherche sur le web...{query}")
            
            web_results = web_search.search(query, sources=['all'], max_results=3)
            
            # Construire contexte
            context = ""
            
            # Wikipedia
            if 'wikipedia' in web_results['sources']:
                wiki = web_results['sources']['wikipedia']
                if wiki.get('found'):
                    context += f"**Wikipedia: {wiki['title']}**\n\n{wiki['summary']}\n\n"
            
            # DuckDuckGo
            if 'duckduckgo' in web_results['sources']:
                ddg = web_results['sources']['duckduckgo']
                if ddg.get('results'):
                    for r in ddg['results'][:2]:
                        context += f"**{r['title']}**\n{r['snippet']}\n\n"
            
            if context:
                # Synthétiser avec Claude
                if ANTHROPIC_API_KEY:
                    return synthesize_with_claude(query, context, source="Web")
                else:
                    return f"🌐 **Résultats Web:**\n\n{context}"
    
    # 4. Fallback
    return f"""❓ **Aucune information trouvée pour: {query}**

**Suggestions:**
- Reformulez votre question
- Utilisez des termes plus spécifiques
- Consultez la documentation en ligne

💡 Configurez ANTHROPIC_API_KEY pour des réponses enrichies par IA!"""

def synthesize_with_claude(query: str, context: str, source: str = "Unknown") -> str:
    """Synthétise les informations avec Claude"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Contexte trouvé ({source}):
{context}

Question: {query}

Synthétise ces informations de manière pédagogique et actionnable.
Utilise des emojis et des exemples pratiques."""
            }]
        )
        
        return f"🤖 **Réponse enrichie** (source: {source})\n\n" + response.content[0].text
    
    except Exception as e:
        return f"⚠️ Erreur synthèse: {str(e)}\n\n**Contexte brut:**\n{context}"

# =============================================================================
# Economic Context
# =============================================================================

def get_economic_context() -> str:
    """Récupère le contexte économique via FRED"""
    fed = get_fed_data()
    if fed and fed.fred:
        return fed.format_economic_context()
    return ""

# =============================================================================
# Main UI
# =============================================================================

def render_ai_assistant():
    """Point d'entrée principal"""
    theme = get_theme_colors()
    
    # Pour activer le debug, ajoutez dans render_ai_assistant():
    # if st.session_state.get('debug_mode', False):
    #     render_cache_manager()
    
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
            🤖 Φ AI Assistant Enriched
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            RAG + Web Search + Economic Data
        </p>
    </div>
    """)
    
    # Initialiser session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    
    # Status sidebar
    mcp_connected = check_mcp_connection()
    
    with st.sidebar:
        st.markdown("### 🤖 AI Status")
        
        if USE_MCP:
            status_icon = "🟢" if mcp_connected else "🔴"
            st.markdown(f"**MCP Server:** {status_icon} {'Online' if mcp_connected else 'Offline'}")
        
        claude_icon = "🟢" if ANTHROPIC_API_KEY else "🔴"
        st.markdown(f"**Claude AI:** {claude_icon} {'Ready' if ANTHROPIC_API_KEY else 'No API Key'}")
        
        # Knowledge status
        st.markdown("**Knowledge Enhanced:**")
        if KNOWLEDGE_ENHANCED:
            rag = get_rag_engine()
            if rag:
                stats = rag.get_stats()
                st.markdown(f"- 📚 RAG: 🟢 {stats['total_documents']} docs")
            else:
                st.markdown("- 📚 RAG: 🔴 Error")
            
            st.markdown("- 🌐 Web Search: 🟢 Active")
            
            fed = get_fed_data()
            if fed and fed.fred:
                st.markdown("- 📊 FRED: 🟢 Connected")
            else:
                st.markdown("- 📊 FRED: 🔴 No API Key")
        else:
            st.markdown("- ⚠️ Not installed")
        
        st.divider()
        
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
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

# ... (Les fonctions render_welcome_screen, render_chat_history, render_chat_input restent identiques)
# ... (Les fonctions handle_portfolio_query, handle_research_query restent identiques)

def handle_education_query(prompt: str) -> str:
    """Questions éducatives avec recherche enrichie"""
    
    # Ajout contexte économique si pertinent
    economic_context = ""
    if any(w in prompt.lower() for w in ['économie', 'fed', 'inflation', 'taux', 'récession']):
        economic_context = "\n\n" + get_economic_context()
    
    # Recherche enrichie
    result = search_knowledge(prompt)
    
    return result + economic_context

# Import des autres fonctions depuis la version précédente
# (handle_portfolio_query, handle_create_portfolio, handle_research_query, etc.)


def render_welcome_screen(theme):
    """Écran d'accueil"""
    st.html(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']}; margin-bottom: 0.5rem;">
            Assistant IA Φ
        </h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Je peux vous aider à gérer vos investissements
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Exemples de commandes")
    
    col1, col2 = st.columns(2)
    
    suggestions = [
        ("📊", "Analyse portfolio", "Analyse mon portfolio et donne des recommandations"),
        ("🔍", "Recherche Apple", "Recherche l'action Apple et analyse-la"),
        ("🏗️", "Créer portfolio", "Crée un portfolio growth avec AAPL, MSFT et GOOGL"),
        ("📈", "Comparer", "Compare Tesla et Ford"),
        ("⏮️", "Backtesting", "Teste mon portfolio sur 2 ans"),
        ("🎓", "Apprendre", "Explique-moi le modèle Black-Litterman"),
    ]
    
    for idx, (icon, title, prompt) in enumerate(suggestions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"{icon} {title}", key=f"sug_{idx}", use_container_width=True):
                process_message(prompt)

def render_chat_history():
    """Affiche l'historique"""
    for msg in st.session_state.chat_history:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

def render_chat_input():
    """Zone de saisie"""
    if prompt := st.chat_input("Posez votre question..."):
        process_message(prompt)

# =============================================================================
# Message Processing
# =============================================================================

def process_message(prompt: str):
    """Traite le message"""
    # Ajouter à l'historique
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    st.session_state.conversation_context.append({'role': 'user', 'content': prompt})
    
    # Traiter
    with st.spinner("🤖 Réflexion..."):
        response = route_query(prompt)
    
    # Ajouter réponse
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
    st.session_state.conversation_context.append({'role': 'assistant', 'content': response})
    
    st.rerun()

def route_query(prompt: str) -> str:
    """Route la requête vers le bon handler"""
    prompt_lower = prompt.lower()
    
    # Portfolio operations
    if any(w in prompt_lower for w in ['mon portfolio', 'mes portfolios', 'mes investissements']):
        return handle_portfolio_query(prompt)
    
    # Create portfolio
    elif any(w in prompt_lower for w in ['crée', 'créer', 'nouveau portfolio', 'construire']):
        return handle_create_portfolio(prompt)
    
    # Research
    elif any(w in prompt_lower for w in ['recherche', 'analyse', 'action', 'entreprise']):
        return handle_research_query(prompt)
    
    # Compare
    elif any(w in prompt_lower for w in ['compare', 'comparaison', 'vs', 'versus']):
        return handle_comparison_query(prompt)
    
    # Backtesting
    elif any(w in prompt_lower for w in ['test', 'backtesting', 'performance historique']):
        return handle_backtesting_query(prompt)
    
    # Education
    elif any(w in prompt_lower for w in ['explique', 'qu\'est-ce', 'comment', 'apprendre', 'comprendre']):
        return handle_education_query(prompt)
    
    # Général
    else:
        return handle_general_query(prompt)

# =============================================================================
# Query Handlers
# =============================================================================

def handle_portfolio_query(prompt: str) -> str:
    """Analyse portfolio via MCP"""
    try:
        # GET - pas de confirmation
        data = execute_mcp_tool("get_portfolios", {"user_id": str(user_id)}, require_confirmation=False)
        
        if not data:
            return "❌ Impossible de récupérer vos portfolios. Vérifiez que le serveur MCP est actif."
        
        portfolios = data.get('portfolios', [])
        
        if not portfolios:
            return """📁 **Aucun portfolio trouvé**

Vous n'avez pas encore créé de portfolio.

**Pour commencer:**
Dites-moi "Crée un portfolio growth avec AAPL, MSFT, GOOGL" et je m'en occupe!"""
        
        # Analyser avec Claude si disponible
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Données des portfolios:
{json.dumps(data, indent=2)}

Question: {prompt}

Fournis une analyse détaillée en français avec:
1. 📊 Vue d'ensemble
2. 💪 Points forts
3. ⚠️ Points d'amélioration
4. 🎯 Recommandations concrètes

Sois actionnable et utilise des emojis!"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback sans Claude
        total_value = data.get('total_value', 0)
        count = data.get('count', 0)
        
        response = f"""📊 **Analyse de Portfolio**

**Vue d'ensemble:**
- Nombre: **{count}** portfolio(s)
- Valeur totale: **${total_value:,.2f}**

**Vos Portfolios:**

"""
        
        for idx, pf in enumerate(portfolios, 1):
            name = pf.get('name', f'Portfolio {idx}')
            value = pf.get('total_amount', 0)
            model = pf.get('model', 'N/A')
            
            response += f"**{idx}. {name}**\n- Valeur: ${value:,.2f}\n- Modèle: {model.title()}\n\n"
        
        response += "\n**💡 Besoin d'analyse plus détaillée?** Configurez Claude AI!"
        
        return response
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

def handle_create_portfolio(prompt: str) -> str:
    """Crée un portfolio via l'IA"""
    try:
        # Extraire les tickers
        tickers = []
        words = prompt.upper().split()
        for word in words:
            # Pattern ticker
            if len(word) >= 2 and len(word) <= 5 and word.isalpha():
                if word not in ['AVEC', 'CRÉE', 'CRÉER', 'PORTFOLIO', 'UN', 'ET']:
                    tickers.append(word)
        
        if not tickers or len(tickers) < 2:
            return """❓ **Création de Portfolio**

Pour créer un portfolio, précisez:
- Les tickers (min 2)
- Le type: growth, income, balanced

**Exemple:**
"Crée un portfolio growth avec AAPL, MSFT, GOOGL"
"Construis un portfolio balanced avec TSLA, NVDA, AMD"
"""
        
        # Détecter le modèle
        model = "growth"  # Par défaut
        if 'income' in prompt.lower():
            model = "income"
        elif 'balanced' in prompt.lower() or 'équilibré' in prompt.lower():
            model = "balanced"
        
        # Charger les données
        with st.spinner(f"📊 Chargement des données pour {', '.join(tickers)}..."):
            data = yahoo.retrieve_data(tuple(tickers), period="2y")
            
            if data.empty:
                return f"❌ Impossible de charger les données pour: {', '.join(tickers)}"
        
        # Calculer les poids (Markowitz Sharpe par défaut)
        with st.spinner("⚙️ Optimisation du portfolio..."):
            from factory import create_portfolio_by_name
            portfolio = create_portfolio_by_name(tickers, "sharp", data)
        
        # Afficher le résultat
        response = f"""✅ **Portfolio {model.title()} Créé!**

**Assets:** {', '.join(tickers)}

**Poids optimaux (Markowitz):**
"""
        
        for asset, weight in zip(tickers, portfolio.weights):
            response += f"\n- {asset}: {weight:.2%}"
        
        response += f"""

**Métriques:**
- Rendement attendu: {portfolio.expected_return:.2%}
- Volatilité: {portfolio.stdev:.2%}
- Sharpe Ratio: {portfolio.sharp_ratio:.3f}

**💾 Voulez-vous sauvegarder ce portfolio?**
Répondez "Oui, sauvegarde-le sous le nom [NOM]" """
        
        # Stocker temporairement
        st.session_state['pending_portfolio'] = {
            'assets': tickers,
            'weights': portfolio.weights,
            'portfolio': portfolio,
            'model': model
        }
        
        return response
        
    except Exception as e:
        return f"❌ Erreur lors de la création: {str(e)}"
def handle_comparison_query(prompt: str) -> str:
    """Compare plusieurs actions"""
    return """📊 **Comparaison d'actions**

**Pour comparer:**
Précisez 2-3 entreprises

**Exemples:**
- "Compare Apple et Microsoft"
- "Tesla vs Ford"
- "GOOGL versus META"

🚧 Fonctionnalité en cours de développement..."""

def handle_backtesting_query(prompt: str) -> str:
    """Backtesting de portfolio"""
    return """⏮️ **Backtesting de Portfolio**

**Pour tester:**
1. Allez dans Portfolio Manager
2. Onglet "Experiments"
3. Sélectionnez votre portfolio
4. Lancez le backtesting

Ou utilisez:
"Test mon portfolio [NOM] sur 2 ans"

🚧 Intégration IA en cours..."""     


def handle_research_query(prompt: str) -> str:
    """Recherche d'entreprise avec Yahoo Finance"""
    try:
        ticker = extract_ticker_from_prompt(prompt)
        
        if not ticker:
            return """🔍 **Recherche d'entreprise**

Veuillez préciser l'entreprise ou le ticker.

**Exemples:**
- "Recherche Apple"
- "Analyse TSLA"
- "Action Microsoft"
"""
        
        with st.spinner(f"📊 Récupération des données pour {ticker}..."):
            data = yahoo.get_ticker_data(ticker, period='1y')
            info = yahoo.get_ticker_info(ticker)
            
            if data is None or data.empty:
                return f"❌ Impossible de récupérer les données pour {ticker}"
        
        # Préparer le contexte
        current_price = float(data['Close'].iloc[-1])
        start_price = float(data['Close'].iloc[0])
        ytd_return = ((current_price - start_price) / start_price) * 100
        
        context = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'N/A'),
            'ytd_return': ytd_return,
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
        
        # Analyser avec Claude
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyse l'entreprise:
{json.dumps(context, indent=2)}

Question: {prompt}

Fournis une analyse complète avec:
1. 🏢 Vue d'ensemble
2. 📊 Analyse financière
3. 📈 Performance et tendances
4. ⚖️ Évaluation (Achat/Conservation/Vente)
5. ⚠️ Risques
6. 🎯 Recommandation finale

Sois précis et actionnable!"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback
        return f"""🔍 **Analyse: {context['name']}**

**{ticker}** | {context['sector']}

**📊 Métriques:**
- Prix: ${context['price']:.2f}
- Capitalisation: ${context['market_cap']/1e9:.1f}B
- P/E Ratio: {context['pe_ratio']:.2f}
- Dividende: {context['dividend_yield']:.2f}%
- Performance YTD: {context['ytd_return']:.2f}%

**⚖️ Évaluation rapide:**
- Valorisation: {'🔴 Chère' if context['pe_ratio'] > 25 else '🟡 Correcte' if context['pe_ratio'] > 15 else '🟢 Attractive'}
- Performance: {'🟢 Forte' if context['ytd_return'] > 15 else '🟡 Modérée' if context['ytd_return'] > 0 else '🔴 Faible'}

💡 Configurez Claude AI pour une analyse détaillée!"""
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"
    
def handle_general_query(prompt: str) -> str:
    """Requêtes générales"""
    if not ANTHROPIC_API_KEY:
        return """🤖 **Assistant IA**

**Je peux vous aider avec:**
- 📊 Analyse de portfolio
- 🔍 Recherche d'entreprises
- 🏗️ Création de portfolios
- 📈 Comparaison d'actions
- 🎓 Éducation financière

**Exemples:**
- "Analyse mon portfolio"
- "Recherche Apple"
- "Crée un portfolio avec AAPL, MSFT"
- "Explique-moi le Sharpe ratio"

⚙️ Configurez ANTHROPIC_API_KEY pour des réponses plus complètes!"""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Utiliser le contexte conversationnel
        messages = st.session_state.conversation_context[-5:]  # Derniers 5 messages
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=messages,
            system="""Tu es un conseiller financier expert et assistant IA pour PyManager.

Tu peux:
- Analyser des portfolios
- Rechercher des entreprises
- Donner des conseils d'investissement
- Expliquer des concepts financiers

Sois précis, actionnable et utilise des emojis pour rendre tes réponses engageantes."""
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Erreur: {str(e)}"
        
        

# =============================================================================

def clear_ticker_cache():
    """Efface le cache de tickers (pour debug ou reset)"""
    st.session_state.ticker_cache = {}
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    st.success("✅ Cache de tickers effacé!")

# =============================================================================
# INTERFACE CACHE (Optionnel - pour debug uniquement)
# =============================================================================

def render_cache_manager():
    """
    Interface de gestion du cache - À mettre dans une page Settings
    ou à appeler avec un raccourci debug
    """
    with st.expander("🗃️ Cache de Tickers (Debug)"):
        cache = st.session_state.get('ticker_cache', {})
        
        if not cache:
            st.info("Cache vide")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric("Tickers mémorisés", len(cache))
        
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                clear_ticker_cache()
                st.rerun()
        
        # Afficher le cache
        df = pd.DataFrame([
            {"Entreprise": k.title(), "Ticker": v}
            for k, v in sorted(cache.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        

if __name__ == "__main__":
    render_ai_assistant()
