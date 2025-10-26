# pagess/ai_assistant.py
"""
AI Assistant - Version Finale Complète
Architecture:
- MCP pour données internes (portfolios, transactions, watchlist)
- Claude AI + Yahoo Finance pour recherche d'entreprises
- Web search pour trouver tickers
- Knowledge base + web pour éducation
"""

import streamlit as st
import anthropic
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import pandas as pd
import numpy as np

# Imports du projet
from uiconfig import get_theme_colors
from dataprovider import yahoo
from pagess.auth import render_auth
from portfolio import Portfolio, get_log_returns

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

# =============================================================================
# KNOWLEDGE BASE - Éducation financière
# =============================================================================

KNOWLEDGE_BASE = {
    "sharpe": {
        "title": "Ratio de Sharpe",
        "content": """📊 **Ratio de Sharpe**

**Définition:**
Le ratio de Sharpe mesure le rendement ajusté au risque. Il indique combien de rendement vous obtenez par unité de risque prise.

**Formule:**
```
Ratio de Sharpe = (Rendement - Taux sans risque) / Écart-type
```

**Interprétation:**
- **> 2.0** : Excellent ⭐⭐⭐
- **1.0-2.0** : Très bon ⭐⭐
- **0.5-1.0** : Acceptable ⭐
- **< 0.5** : Faible ❌

**Exemple:**
Portfolio A: 15% rendement, 10% volatilité → Sharpe = 1.3 ⭐⭐
Portfolio B: 20% rendement, 25% volatilité → Sharpe = 0.72 ⭐

→ Portfolio A a un meilleur rendement ajusté au risque!"""
    },
    
    "pe_ratio": {
        "title": "Ratio P/E",
        "content": """💰 **Ratio P/E (Price-to-Earnings)**

**Définition:**
Le P/E mesure le prix que vous payez pour chaque dollar de bénéfice.

**Formule:**
```
P/E = Prix de l'action / Bénéfice par action
```

**Interprétation:**
- **P/E < 15** : Sous-évalué 🟢
- **P/E 15-25** : Valorisation normale 🟡
- **P/E > 25** : Potentiellement cher 🔴

**Attention:** P/E élevé peut indiquer forte croissance attendue. Comparez avec le secteur!"""
    },
    
    "diversification": {
        "title": "Diversification",
        "content": """🎯 **Diversification du Portfolio**

**Principe:** "Ne mettez pas tous vos œufs dans le même panier"

**Pourquoi?**
- 📉 Réduit le risque global
- 📊 Lisse les rendements
- 🛡️ Protection contre les chocs

**Comment diversifier:**
1. **Par classes d'actifs:** Actions 60%, Obligations 30%, Cash 10%
2. **Par secteurs:** Tech, Santé, Finance (aucun > 25%)
3. **Par géographie:** US 60%, International 30%, Émergents 10%

**💡 Règle:** 15-30 actions offrent ~90% des bénéfices de diversification"""
    },
    
    "markowitz": {
        "title": "Théorie de Markowitz",
        "content": """📊 **Théorie Moderne du Portfolio (Markowitz)**

**Concept clé:** Optimiser le ratio risque/rendement

**Frontière efficiente:**
Ensemble des portfolios offrant le meilleur rendement pour un niveau de risque donné.

**3 stratégies:**
1. **Max Sharpe:** Meilleur rendement ajusté au risque
2. **Min Risk:** Volatilité minimale
3. **Max Return:** Rendement maximal (plus risqué)

**Dans PyManager:** Utilisez le Portfolio Builder > Markowitz"""
    },
    
    "black_litterman": {
        "title": "Modèle Black-Litterman",
        "content": """🎯 **Black-Litterman Model**

**Avantage sur Markowitz:**
Intègre vos convictions personnelles (views) sur le marché.

**Comment ça marche:**
1. Commence avec l'équilibre de marché
2. Ajoute vos vues (ex: "Apple va surperformer de 5%")
3. Combine les deux pour des poids optimaux

**Dans PyManager:** Portfolio Builder > BL (Black-Litterman)

**Cas d'usage:** Quand vous avez des insights spécifiques sur certaines actions."""
    }
}

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
    """
    Exécute un outil MCP
    - GET operations: pas de confirmation
    - WRITE operations: avec confirmation si require_confirmation=True
    """
    if not USE_MCP:
        st.warning("⚠️ MCP Server désactivé. Activez-le dans secrets.toml")
        return None
    
    try:
        # Afficher l'action si confirmation requise
        if require_confirmation:
            with st.expander("🔧 Action MCP", expanded=True):
                st.write(f"**Outil:** `{tool_name}`")
                st.json(params)
                
                col1, col2 = st.columns(2)
                with col1:
                    if not st.button("✅ Exécuter", key=f"mcp_exec_{tool_name}_{hash(str(params))}"):
                        st.info("⏳ En attente de votre confirmation...")
                        st.stop()
                with col2:
                    if st.button("❌ Annuler", key=f"mcp_cancel_{tool_name}_{hash(str(params))}"):
                        return None
        
        # Exécuter
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json={
                "tool": tool_name,
                "params": params,
                "require_approval": False  # On gère la confirmation côté Streamlit
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data")
            else:
                st.error(f"❌ Erreur MCP: {result.get('error')}")
                return None
        else:
            st.error(f"❌ Erreur HTTP {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Erreur MCP: {str(e)}")
        return None

# =============================================================================
# Ticker Search avec Web
# =============================================================================

def search_ticker_from_name(company_name: str) -> Optional[str]:
    """
    Recherche un ticker à partir du nom d'entreprise
    Utilise yfinance pour la recherche
    """
    try:
        import yfinance as yf
        
        # Essayer recherche directe
        ticker_obj = yf.Ticker(company_name)
        info = ticker_obj.info
        
        if info and 'symbol' in info and info.get('symbol'):
            return info['symbol']
        
        # Essayer variations
        variations = [
            company_name.upper(),
            company_name.replace(' ', ''),
            company_name.split()[0] if ' ' in company_name else company_name
        ]
        
        for variant in variations:
            try:
                ticker_obj = yf.Ticker(variant)
                info = ticker_obj.info
                if info and 'symbol' in info and info.get('symbol'):
                    return info['symbol']
            except:
                continue
                
    except Exception as e:
        st.error(f"Erreur recherche: {e}")
    
    return None

def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    """Extrait un ticker du texte"""
    # Tickers communs
    common_tickers = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE'
    }
    
    prompt_lower = prompt.lower()
    for name, ticker in common_tickers.items():
        if name in prompt_lower:
            return ticker
    
    # Pattern matching pour tickers explicites
    patterns = [
        r'\$([A-Z]{1,5})',
        r'\(([A-Z]{1,5})\)',
        r'\b([A-Z]{2,5})\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, prompt.upper())
        if matches and matches[0] not in ['US', 'AI', 'ML', 'RL']:
            return matches[0]
    
    # Essayer d'extraire le nom d'entreprise
    company_patterns = [
        r'(?:recherche|analyse|action)\s+([A-Za-z\s]+?)(?:\s+et|\s+pour|$)',
        r'entreprise\s+([A-Za-z\s]+?)(?:\s+et|$)',
    ]
    
    for pattern in company_patterns:
        matches = re.search(pattern, prompt, re.IGNORECASE)
        if matches:
            company_name = matches.group(1).strip()
            if len(company_name) > 2:
                ticker = search_ticker_from_name(company_name)
                if ticker:
                    st.info(f"✅ Ticker trouvé: **{ticker}** pour '{company_name}'")
                    return ticker
    
    return None

# =============================================================================
# Main UI
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
            Votre conseiller portfolio intelligent
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
        st.markdown("**Yahoo Finance:** 🟢 Active")
        
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
            value = pf.get('amount', 0)
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

def handle_education_query(prompt: str) -> str:
    """Questions éducatives avec knowledge base"""
    prompt_lower = prompt.lower()
    
    # Chercher dans la knowledge base
    for key, content in KNOWLEDGE_BASE.items():
        if key in prompt_lower or content['title'].lower() in prompt_lower:
            return content['content']
    
    # Si pas trouvé et Claude disponible
    if ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Tu es un professeur de finance. Explique de manière pédagogique et concise: {prompt}

Utilise des emojis et des exemples pratiques."""
            }]
        )
        
        return response.content[0].text
    
    # Liste des sujets disponibles
    return f"""🎓 **Centre d'Éducation Financière**

**Sujets disponibles:**

{chr(10).join([f'- {content["title"]}' for content in KNOWLEDGE_BASE.values()])}

**Demandez:**
- "Explique-moi le ratio de Sharpe"
- "Qu'est-ce que la diversification?"
- "Comment fonctionne Black-Litterman?"

💡 Configurez Claude AI pour plus de sujets!"""

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
# Entry Point
# =============================================================================

if __name__ == "__main__":
    render_ai_assistant()
