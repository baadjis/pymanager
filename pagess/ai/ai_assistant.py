# pagess/ai_assistant.py
"""
AI Assistant - Version intégrée dans PyManager
S'adapte au thème et à la structure existante
"""

import streamlit as st
import anthropic
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import depuis votre projet existant
from uiconfig import get_theme_colors
from dataprovider import yahoo

# Import database - adapter selon votre structure
try:
    from database import get_portfolios, get_transactions
except ImportError:
    def get_portfolios():
        return []
    def get_transactions():
        return []

# Configuration
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://localhost:8000")
USE_MCP = True  # Mettre False pour désactiver MCP

# =============================================================================
# MCP Integration
# =============================================================================

def check_mcp_connection():
    """Vérifie la connexion au serveur MCP"""
    if not USE_MCP:
        return False
    try:
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def execute_mcp_tool(tool_name: str, params: Dict[str, Any] = {}):
    """Exécute un outil MCP"""
    if not USE_MCP:
        return None
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json={"tool": tool_name, "params": params},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("data") if result.get("success") else None
    except:
        return None
    return None

# =============================================================================
# Main Function
# =============================================================================

def render_ai_assistant():
    """Page AI Assistant - Point d'entrée principal"""
    
    theme = get_theme_colors()
    
    # Header avec votre style
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
    
    # Initialiser l'historique
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Vérifier statut MCP
    mcp_connected = check_mcp_connection()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 🤖 AI Status")
        
        # MCP Status
        if USE_MCP:
            status_icon = "🟢" if mcp_connected else "🔴"
            status_text = "Connected" if mcp_connected else "Offline"
            st.markdown(f"**MCP Server:** {status_icon} {status_text}")
        
        # Claude Status
        claude_icon = "🟢" if ANTHROPIC_API_KEY else "🔴"
        claude_text = "Ready" if ANTHROPIC_API_KEY else "No API Key"
        st.markdown(f"**Claude AI:** {claude_icon} {claude_text}")
        
        if not ANTHROPIC_API_KEY:
            st.warning("⚠️ Configure ANTHROPIC_API_KEY in secrets.toml")
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Effacer conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Stats
        if st.session_state.chat_history:
            st.metric("Messages", len(st.session_state.chat_history))
    
    # Zone de chat
    if not st.session_state.chat_history:
        render_welcome_screen(theme)
    else:
        render_chat_history(theme)
    
    # Input utilisateur
    render_chat_input()

# =============================================================================
# UI Components
# =============================================================================

def render_welcome_screen(theme):
    """Écran d'accueil avec suggestions"""
    
    st.html(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']}; margin-bottom: 0.5rem;">
            Assistant IA Φ
        </h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Posez-moi des questions sur vos investissements
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Suggestions rapides")
    
    col1, col2 = st.columns(2)
    
    suggestions = [
        ("📊", "Analyser mon portfolio", "Analyse mon portfolio et donne-moi des recommandations"),
        ("🔍", "Rechercher une action", "Recherche Apple (AAPL) et dis-moi si c'est un bon investissement"),
        ("📈", "Trouver des actions", "Trouve-moi des actions technologiques à forte croissance"),
        ("🎓", "Apprendre", "Explique-moi le ratio de Sharpe et comment l'utiliser"),
    ]
    
    for idx, (icon, title, prompt) in enumerate(suggestions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"{icon} {title}", key=f"sug_{idx}", use_container_width=True):
                process_message(prompt)

def render_chat_history(theme):
    """Affiche l'historique du chat"""
    
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                
                # Actions
                if 'actions' in msg:
                    cols = st.columns(len(msg['actions']))
                    for i, action in enumerate(msg['actions']):
                        with cols[i]:
                            if st.button(action['label'], key=f"action_{idx}_{i}"):
                                if action['type'] == 'navigate':
                                    st.session_state.current_page = action['target']
                                    st.rerun()

def render_chat_input():
    """Zone de saisie"""
    
    if prompt := st.chat_input("Posez votre question..."):
        process_message(prompt)

# =============================================================================
# Message Processing
# =============================================================================

def process_message(prompt: str):
    """Traite le message utilisateur"""
    
    # Ajouter message utilisateur
    st.session_state.chat_history.append({
        'role': 'user',
        'content': prompt
    })
    
    # Traiter avec IA
    with st.spinner("🤖 Réflexion en cours..."):
        response = handle_query(prompt)
    
    # Ajouter réponse
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response
    })
    
    st.rerun()

def handle_query(prompt: str) -> str:
    """Route la requête vers le bon handler"""
    
    prompt_lower = prompt.lower()
    
    # Requêtes portfolio
    if any(w in prompt_lower for w in ['portfolio', 'mes actions', 'mes investissements']):
        return handle_portfolio_query(prompt)
    
    # Recherche entreprise
    elif any(w in prompt_lower for w in ['recherche', 'analyse', 'action', 'aapl', 'msft', 'tsla']):
        return handle_research_query(prompt)
    
    # Questions éducatives
    elif any(w in prompt_lower for w in ['explique', 'qu\'est-ce', 'comment', 'apprendre']):
        return handle_education_query(prompt)
    
    # Général
    else:
        return handle_general_query(prompt)

# =============================================================================
# Query Handlers
# =============================================================================

def handle_portfolio_query(prompt: str) -> str:
    """Analyse du portfolio"""
    
    try:
        # Récupérer données via MCP ou direct
        if USE_MCP:
            portfolios_data = execute_mcp_tool("get_portfolios")
        else:
            portfolios = list(get_portfolios())
            portfolios_data = {
                'portfolios': portfolios,
                'count': len(portfolios),
                'total_value': sum([p.get('amount', 0) for p in portfolios])
            }
        
        if not portfolios_data:
            return "❌ Impossible de récupérer les données du portfolio."
        
        portfolios = portfolios_data.get('portfolios', [])
        total_value = portfolios_data.get('total_value', 0)
        count = portfolios_data.get('count', 0)
        
        if count == 0:
            return """📁 **Aucun portfolio trouvé**

Vous n'avez pas encore créé de portfolio. 

**Pour commencer:**
1. Allez sur la page **Portfolio**
2. Créez votre premier portfolio
3. Ajoutez vos actifs

Besoin d'aide pour la stratégie? Demandez-moi!"""
        
        # Utiliser Claude si disponible
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Données du portfolio:
{json.dumps(portfolios_data, indent=2)}

Question: {prompt}

Fournis une analyse détaillée avec:
1. Vue d'ensemble
2. Points forts
3. Points d'amélioration
4. Recommandations concrètes"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback sans Claude
        response = f"""📊 **Analyse de Portfolio**

Vous avez **{count} portfolio(s)** d'une valeur totale de **${total_value:,.2f}**

**Vos Portfolios:**

"""
        
        for idx, pf in enumerate(portfolios, 1):
            name = pf.get('name', f'Portfolio {idx}')
            value = pf.get('amount', 0)
            model = pf.get('model', 'balanced')
            pct = (value / total_value * 100) if total_value > 0 else 0
            
            response += f"""**{idx}. {name}**
- Valeur: ${value:,.2f} ({pct:.1f}%)
- Modèle: {model.title()}

"""
        
        response += """
**💡 Recommandations:**
- Vérifiez votre allocation d'actifs
- Envisagez un rééquilibrage si nécessaire
- Surveillez les positions individuelles

Voulez-vous une analyse plus détaillée?"""
        
        return response
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

def handle_research_query(prompt: str) -> str:
    """Recherche d'entreprise"""
    
    # Extraire le ticker
    ticker = extract_ticker(prompt)
    
    if not ticker:
        return "🔍 Veuillez spécifier un symbole boursier (ex: AAPL, MSFT, TSLA)"
    
    try:
        # Récupérer données Yahoo Finance
        data = yahoo.get_ticker_data(ticker, period='1y')
        info = yahoo.get_ticker_info(ticker)
        
        if data is None or data.empty:
            return f"❌ Impossible de récupérer les données pour {ticker}"
        
        # Calculer métriques
        current_price = float(data['Close'].iloc[-1])
        ytd_return = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
        print(current_price)
        context = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'sector': info.get('sector', 'N/A'),
            'ytd_return': ytd_return
        }
        
        # Utiliser Claude si disponible
        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": f"""Données de l'entreprise:
{json.dumps(context, indent=2)}

Question: {prompt}

Fournis une analyse complète avec:
1. Vue d'ensemble de l'entreprise
2. Analyse financière
3. Évaluation (valorisation)
4. Recommandation (Achat/Conservation/Vente)
5. Facteurs de risque"""
                }]
            )
            
            return response.content[0].text
        
        # Fallback
        return f"""🔍 **Recherche: {context['name']}**

**{ticker}** | {context['sector']}

**Métriques clés:**
- Prix: ${context['price']:.2f}
- Capitalisation: ${context['market_cap']/1e9:.1f}B
- P/E: {context['pe_ratio']:.2f}
- Rendement YTD: {context['ytd_return']:.2f}%

**Évaluation rapide:**
- Valorisation: {'Chère' if context['pe_ratio'] > 25 else 'Correcte' if context['pe_ratio'] > 15 else 'Bon marché'}
- Performance: {'Forte' if context['ytd_return'] > 15 else 'Modérée' if context['ytd_return'] > 0 else 'Faible'}

Pour une analyse détaillée, configurez l'API Claude dans secrets.toml"""
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

def handle_education_query(prompt: str) -> str:
    """Questions éducatives"""
    
    prompt_lower = prompt.lower()
    
    # Base de connaissances
    if 'sharpe' in prompt_lower or 'ratio de sharpe' in prompt_lower:
        return """📊 **Ratio de Sharpe**

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
```
Portfolio A: 15% rendement, 10% volatilité → Sharpe = 1.2
Portfolio B: 20% rendement, 25% volatilité → Sharpe = 0.68
→ Portfolio A a un meilleur rendement ajusté au risque!
```

**Astuce:** Un ratio de Sharpe élevé = meilleure performance pour le risque pris."""
    
    elif 'diversification' in prompt_lower:
        return """🎯 **Diversification du Portfolio**

**Principe:**
"Ne mettez pas tous vos œufs dans le même panier"

**Pourquoi diversifier?**
- Réduit le risque global
- Lisse les rendements
- Protection contre les chocs sectoriels

**Comment diversifier:**

**1. Classes d'actifs**
- Actions: 60%
- Obligations: 30%
- Liquidités/Alternatives: 10%

**2. Secteurs**
- Technologie, Santé, Finance, etc.
- Aucun secteur > 25%

**3. Géographie**
- Domestique: 60%
- International: 30%
- Marchés émergents: 10%

**Règle:** 15-30 actions différentes offrent ~90% des bénéfices de diversification"""
    
    # Réponse générale
    return """🎓 **Centre d'éducation financière**

Je peux expliquer:

**📊 Métriques de risque**
- Ratio de Sharpe
- Bêta
- Écart-type

**💰 Valorisation**
- Ratio P/E
- PEG Ratio
- Valeur comptable

**🎯 Théorie du portfolio**
- Diversification
- Allocation d'actifs
- Frontière efficiente

**📈 Stratégies**
- Dollar-Cost Averaging
- Investissement de valeur
- Investissement de croissance

Posez une question spécifique!"""

def handle_general_query(prompt: str) -> str:
    """Requêtes générales avec Claude"""
    
    if not ANTHROPIC_API_KEY:
        return """🤖 **Assistant IA**

Je peux vous aider avec:
- 📊 Analyse de portfolio
- 🔍 Recherche d'entreprises
- 📈 Screening d'actions
- 🎓 Éducation financière

**Note:** Fonctionnalités complètes nécessitent la configuration de l'API Claude dans secrets.toml

Essayez: "Analyse mon portfolio" ou "Recherche Apple (AAPL)"!"""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Tu es un conseiller financier expert.

Question: {prompt}

Fournis une réponse claire et actionnable."""
            }]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Erreur: {str(e)}"

# =============================================================================
# Utilities
# =============================================================================

def extract_ticker(prompt: str) -> Optional[str]:
    """Extrait le symbole boursier du prompt"""
    import re
    
    common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    prompt_upper = prompt.upper()
    for ticker in common_tickers:
        if ticker in prompt_upper:
            return ticker
    
    # Pattern matching
    patterns = [r'\b([A-Z]{1,5})\b', r'\$([A-Z]{1,5})', r'\(([A-Z]{1,5})\)']
    
    for pattern in patterns:
        matches = re.findall(pattern, prompt_upper)
        if matches:
            return matches[0]
    
    return None
