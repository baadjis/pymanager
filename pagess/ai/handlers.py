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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from functools import lru_cache
import pickle

# Add knowledge module to path
sys.path.append(str(Path(__file__).parent.parent))


from dataprovider import yahoo
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


# Imports knowledge
try:
   
    from knowledge.fed_data import FREDDataProvider
    KNOWLEDGE_ENHANCED = True
except ImportError:
    KNOWLEDGE_ENHANCED = False
    st.warning("⚠️ Knowledge modules not available. Install: pip install duckduckgo-search wikipedia-api fredapi sentence-transformers")

user_id = st.session_state.user_id
user_name = st.session_state.get('user_name', 'User')

# Configuration
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
MCP_SERVER_URL = st.secrets.get("MCP_SERVER_URL", "http://localhost:8000")
USE_MCP = st.secrets.get("USE_MCP", True)
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
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

def clear_ticker_cache():
    """Efface le cache de tickers (pour debug ou reset)"""
    st.session_state.ticker_cache = {}
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    st.success("✅ Cache de tickers effacé!")


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
# KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = {
    "sharpe": {
        "title": "Ratio de Sharpe",
        "content": """📊 **Ratio de Sharpe**
        
Mesure le rendement ajusté au risque d'un investissement.

**Formule:** (R - Rf) / σ
- R = Rendement du portfolio
- Rf = Taux sans risque
- σ = Écart-type (volatilité)

**Interprétation:**
- > 2.0 : Excellent
- 1.0 - 2.0 : Bon
- 0.5 - 1.0 : Acceptable
- < 0.5 : Faible

**Exemple:** Sharpe = 1.5 → Pour chaque unité de risque, vous obtenez 1.5 unités de rendement excédentaire.
"""
    },
    "var": {
        "title": "Value at Risk (VaR)",
        "content": """📉 **Value at Risk (VaR)**
        
Perte maximale probable sur une période donnée à un niveau de confiance donné.

**Exemple:** VaR 95% = 10,000€
→ 95% de chances que la perte ne dépasse pas 10,000€

**Types:**
- VaR 95% : Perte dépassée 5% du temps
- VaR 99% : Perte dépassée 1% du temps

**Limites:** Ne dit rien sur la taille des pertes au-delà du seuil (voir CVaR).
"""
    },
    "sortino": {
        "title": "Ratio de Sortino",
        "content": """📊 **Ratio de Sortino**
        
Variante du Sharpe qui pénalise uniquement la volatilité négative (downside).

**Avantage:** Plus réaliste car seules les baisses sont considérées comme du risque, pas les hausses.

**Interprétation:** Similaire au Sharpe mais généralement plus élevé.
"""
    },
    "markowitz": {
        "title": "Théorie de Markowitz",
        "content": """📊 **Théorie Moderne du Portfolio (Markowitz, 1952)**
        
Principe : Optimiser le couple rendement/risque via la diversification.

**Frontière Efficiente:** Ensemble des portfolios optimaux offrant le meilleur rendement pour chaque niveau de risque.

**Hypothèses:**
- Rendements suivent une loi normale
- Investisseurs rationnels
- Pas de coûts de transaction
"""
    },
    "quantum": {
        "title": "Quantum Computing Sector",
        "content": """⚛️ **Secteur Quantum Computing**
        
**Principales entreprises:**
- IONQ : Leader hardware quantique
- RGTI (Rigetti Computing) : Processeurs quantiques
- QUBT (Quantum Computing Inc) : Software quantique
- IBM : Recherche quantique avancée
- GOOGL : Google Quantum AI

**Applications:** Cryptographie, optimisation, simulation moléculaire, ML
**Risque:** Secteur émergent, haute volatilité
"""
    },
}

# =============================================================================
# MCP v4.0 INTEGRATION
# =============================================================================

def check_mcp_connection() -> bool:
    """Vérifier connexion MCP Server v4.0"""
    if not USE_MCP:
        return False
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get('status') in ['healthy', 'degraded']
        return False
    except:
        return False

def get_mcp_tools() -> List[Dict]:
    """Récupérer liste des tools MCP disponibles"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/tools", timeout=5)
        if response.status_code == 200:
            return response.json().get('tools', [])
        return []
    except:
        return []

def execute_mcp_tool(tool_name: str, params: Dict[str, Any], require_confirmation: bool = False) -> Optional[Any]:
    """Exécuter un tool MCP v4.0"""
    if not USE_MCP:
        return None
    
    try:
        # Confirmation UI si nécessaire
        if require_confirmation:
            with st.expander("🔧 Action MCP à confirmer", expanded=True):
                st.write(f"**Tool:** `{tool_name}`")
                st.json(params)
                
                col1, col2 = st.columns(2)
                with col1:
                    if not st.button("✅ Confirmer", key=f"mcp_{tool_name}_{hash(str(params))}", use_container_width=True):
                        st.info("⏳ En attente de confirmation...")
                        st.stop()
                with col2:
                    if st.button("❌ Annuler", key=f"mcp_cancel_{tool_name}", use_container_width=True):
                        st.warning("Action annulée")
                        return None
        
        # Exécution
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json={"tool": tool_name, "params": params},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data")
            else:
                st.error(f"Erreur MCP: {result.get('error', 'Unknown error')}")
                return None
        
        return None
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout MCP Server")
        return None
    except Exception as e:
        st.error(f"❌ Erreur MCP: {e}")
        return None

# =============================================================================
# HANDLERS - Portfolio Management
# =============================================================================

def handle_portfolio_query(prompt: str) -> str:
    """Gérer requêtes portfolio"""
    try:
        # Récupérer portfolios via MCP
        data = execute_mcp_tool("get_portfolios", {"user_id": str(user_id)})
        
        if not data or not data.get('portfolios'):
            return """📁 **Aucun portfolio trouvé**
            
Vous n'avez pas encore de portfolio. Créez-en un via le menu "Portfolio Manager" !
"""
        
        portfolios = data['portfolios']
        aggregates = data.get('aggregates', {})
        
        # Construire réponse formatée
        response = f"""📊 **Vos Portfolios** ({len(portfolios)} total)

**Vue d'ensemble:**
- 💰 Valeur totale: ${aggregates.get('total_value', 0):,.2f}
- 📈 P&L total: ${aggregates.get('total_pnl', 0):,.2f} ({aggregates.get('total_pnl_pct', 0):.2f}%)
- 💵 Investi: ${aggregates.get('total_invested', 0):,.2f}

**Détails par portfolio:**
"""
        
        for pf in portfolios[:5]:  # Limiter à 5
            name = pf.get('name', 'Unknown')
            value = pf.get('current_value', 0)
            pnl = pf.get('pnl', 0)
            pnl_pct = pf.get('pnl_pct', 0)
            holdings_count = len(pf.get('holdings', []))
            
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            response += f"""
{emoji} **{name}**
   - Valeur: ${value:,.2f}
   - P&L: ${pnl:,.2f} ({pnl_pct:.2f}%)
   - Assets: {holdings_count}
"""
        
        if len(portfolios) > 5:
            response += f"\n*... et {len(portfolios) - 5} autres portfolios*"
        
        return response
    
    except Exception as e:
        return f"❌ Erreur lors de la récupération des portfolios: {e}"
        

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

def handle_portfolio_analysis(portfolio_name: str) -> str:
    """Analyser un portfolio spécifique avec risque"""
    try:
        # 1. Détails du portfolio
        details = execute_mcp_tool("get_portfolio_details", {
            "user_id": str(user_id),
            "portfolio_name": portfolio_name
        })
        
        if not details:
            return f"❌ Portfolio '{portfolio_name}' non trouvé"
        
        # 2. Analyse de risque
        risk = execute_mcp_tool("analyze_portfolio_risk", {
            "user_id": str(user_id),
            "portfolio_name": portfolio_name
        })
        
        # Construire réponse
        response = f"""📊 **Analyse: {portfolio_name}**

**Holdings:**
"""
        
        for holding in details.get('holdings', [])[:10]:
            symbol = holding.get('symbol', 'N/A')
            weight = holding.get('weight', 0) * 100
            value = holding.get('value', 0)
            pnl = holding.get('pnl', 0)
            
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            response += f"{emoji} {symbol}: {weight:.1f}% (${value:,.2f})\n"
        
        # Métriques de risque
        if risk and 'risk_metrics' in risk:
            metrics = risk['risk_metrics']
            interpretation = risk.get('interpretation', {})
            
            response += f"""

**📉 Métriques de Risque:**
- Volatilité annuelle: {metrics.get('volatility_annual', 0)*100:.2f}%
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f} ({interpretation.get('sharpe_quality', 'N/A')})
- Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%
- VaR 95%: {metrics.get('var_95', 0)*100:.2f}%
- CVaR 95%: {metrics.get('cvar_95', 0)*100:.2f}%

**💡 Niveau de risque: {interpretation.get('risk_level', 'N/A').upper()}**
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur analyse: {e}"

# =============================================================================
# HANDLERS - Market Intelligence (NEW in v4.0)
# =============================================================================

def handle_market_overview(region: str = "US") -> str:
    """Vue d'ensemble du marché"""
    try:
        data = execute_mcp_tool("get_market_overview", {
            "region": region,
            "include_sectors": True,
            "period": "1mo"
        })
        
        if not data:
            return "❌ Données marché indisponibles"
        
        indices = data.get('indices', [])
        sentiment = data.get('market_sentiment', {})
        sectors = data.get('sectors', [])
        
        response = f"""📊 **Market Overview - {region}**

**Sentiment global: {sentiment.get('label', 'N/A').upper()} ({sentiment.get('score', 0):.2f}%)**

**Indices principaux:**
"""
        
        for idx in indices:
            name = idx.get('name', 'N/A')
            price = idx.get('price', 0)
            change = idx.get('change_1mo', 0)
            sentiment_idx = idx.get('sentiment', 'neutral')
            
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            response += f"{emoji} **{name}**: ${price:,.2f} ({change:+.2f}%) - {sentiment_idx}\n"
        
        # Secteurs
        if sectors:
            response += "\n**Performance par secteur (1 mois):**\n"
            for sector in sectors[:5]:
                perf = sector.get('performance', 0)
                emoji = "🟢" if perf > 0 else "🔴"
                response += f"{emoji} {sector.get('sector', 'N/A').title()}: {perf:+.2f}%\n"
        
        return response
    
    except Exception as e:
        return f"❌ Erreur market overview: {e}"

def handle_sector_analysis(sector: str, subsector: Optional[str] = None) -> str:
    """Analyse sectorielle détaillée"""
    try:
        data = execute_mcp_tool("analyze_sector", {
            "sector": sector.lower(),
            "subsector": subsector,
            "metrics": ["performance", "sentiment", "top_stocks", "correlations"],
            "period": "3mo"
        })
        
        if not data:
            return f"❌ Analyse secteur '{sector}' indisponible"
        
        performance = data.get('performance', {})
        sentiment = data.get('sentiment', {})
        top_performers = data.get('top_performers', [])
        correlations = data.get('correlations', {})
        
        response = f"""🔬 **Analyse Secteur: {sector.title()}**
{f'({subsector})' if subsector else ''}

**Performance moyenne (3 mois): {performance.get('average', 0):.2f}%**
**Sentiment: {sentiment.get('label', 'N/A').upper()} (score: {sentiment.get('score', 0):.2f})**

**Top Performers:**
"""
        
        for stock in top_performers[:5]:
            ticker = stock.get('ticker', 'N/A')
            perf = stock.get('performance_3mo', 0)
            name = stock.get('name', ticker)
            
            emoji = "🟢" if perf > 0 else "🔴"
            response += f"{emoji} **{ticker}** ({name[:30]}): {perf:+.2f}%\n"
        
        # Corrélation
        if correlations:
            avg_corr = correlations.get('average', 0)
            diversif = correlations.get('diversification', 'N/A')
            response += f"\n**Diversification: {diversif.upper()}** (corrélation moyenne: {avg_corr:.2f})"
        
        return response
    
    except Exception as e:
        return f"❌ Erreur analyse secteur: {e}"

def handle_sentiment_analysis(target: str) -> str:
    """Analyse sentiment pour un ticker/secteur"""
    try:
        data = execute_mcp_tool("get_market_sentiment", {
            "target": target.upper(),
            "period": "1mo",
            "include_news": False
        })
        
        if not data:
            return f"❌ Sentiment '{target}' indisponible"
        
        sentiment = data.get('sentiment', {})
        metrics = data.get('metrics', {})
        
        label = sentiment.get('label', 'N/A')
        score = sentiment.get('score', 0)
        confidence = sentiment.get('confidence', 0)
        
        emoji_map = {
            'very bullish': '🚀',
            'bullish': '🟢',
            'neutral': '⚪',
            'bearish': '🔴',
            'very bearish': '💥'
        }
        
        emoji = emoji_map.get(label, '⚪')
        
        response = f"""{emoji} **Sentiment Analysis: {target.upper()}**

**Sentiment: {label.upper()}**
- Score: {score:.2f}
- Confiance: {confidence:.2f}

**Indicateurs techniques:**
- RSI (14): {metrics.get('rsi', 0):.1f}
- Au-dessus SMA 20: {'✅' if metrics.get('above_sma20') else '❌'}
- Au-dessus SMA 50: {'✅' if metrics.get('above_sma50') else '❌'}
- Jours positifs: {metrics.get('positive_days', 0)}
- Jours négatifs: {metrics.get('negative_days', 0)}

**Volatilité:** {data.get('volatility', 0)*100:.2f}%
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur sentiment: {e}"

def handle_compare_markets(targets: List[str]) -> str:
    """Comparer plusieurs marchés/assets"""
    try:
        data = execute_mcp_tool("compare_markets", {
            "targets": [t.upper() for t in targets],
            "period": "1y"
        })
        
        if not data:
            return "❌ Comparaison indisponible"
        
        comparison = data.get('comparison', [])
        rankings = data.get('rankings', {})
        
        response = f"""📊 **Comparaison: {', '.join(targets)}**

**Rankings:**
- 🏆 Meilleure performance: {rankings.get('best_performer', 'N/A')}
- 💎 Meilleur Sharpe: {rankings.get('best_risk_adjusted', 'N/A')}
- 🛡️ Moins volatile: {rankings.get('lowest_volatility', 'N/A')}

**Détails:**
"""
        
        for asset in comparison:
            target = asset.get('target', 'N/A')
            perf = asset.get('performance', 0)
            vol = asset.get('volatility', 0)
            sharpe = asset.get('sharpe_ratio', 0)
            
            emoji = "🟢" if perf > 0 else "🔴"
            response += f"""
{emoji} **{target}**
   - Performance: {perf:+.2f}%
   - Volatilité: {vol*100:.2f}%
   - Sharpe: {sharpe:.2f}
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur comparaison: {e}"

# =============================================================================
# HANDLERS - Backtesting & Predictions (NEW in v4.0)
# =============================================================================

def handle_backtest(portfolio_name: str, start_date: str, end_date: str) -> str:
    """Backtester un portfolio"""
    try:
        data = execute_mcp_tool("backtest_portfolio", {
            "user_id": str(user_id),
            "portfolio_name": portfolio_name,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 10000,
            "rebalance_frequency": "monthly"
        })
        
        if not data:
            return "❌ Backtest échoué"
        
        results = data.get('backtest_results', {})
        config = data.get('configuration', {})
        
        return_pct = results.get('total_return_pct', 0)
        emoji = "🟢" if return_pct > 0 else "🔴"
        
        response = f"""{emoji} **Backtest Results: {portfolio_name}**

**Période:** {start_date} → {end_date}

**Performance:**
- Capital initial: ${results.get('initial_capital', 0):,.2f}
- Valeur finale: ${results.get('final_value', 0):,.2f}
- Rendement total: {return_pct:+.2f}%
- Rendement annualisé: {results.get('annualized_return', 0)*100:+.2f}%

**Métriques de risque:**
- Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
- Sortino Ratio: {results.get('sortino_ratio', 0):.2f}
- Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%
- Volatilité: {results.get('volatility', 0)*100:.2f}%
- Win Rate: {results.get('win_rate', 0)*100:.1f}%

**Configuration:**
- Rebalancing: {config.get('rebalance_frequency', 'N/A')}
- Coûts transaction: {config.get('transaction_cost', 0)*100:.2f}%
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur backtest: {e}"

def handle_prediction(portfolio_name: str, horizon: str = "3mo") -> str:
    """Prédire performance d'un portfolio"""
    try:
        data = execute_mcp_tool("predict_performance", {
            "user_id": str(user_id),
            "portfolio_name": portfolio_name,
            "horizon": horizon,
            "model": "ensemble",
            "confidence_level": 0.95
        })
        
        if not data:
            return "❌ Prédiction échouée"
        
        prediction = data.get('prediction', {})
        
        expected = prediction.get('expected_return_pct', 0)
        emoji = "🟢" if expected > 0 else "🔴"
        
        response = f"""{emoji} **Prédiction: {portfolio_name}**

**Horizon:** {horizon}

**Rendement attendu: {expected:+.2f}%**

**Intervalle de confiance (95%):**
- Borne inférieure: {prediction.get('confidence_lower', 0)*100:+.2f}%
- Borne supérieure: {prediction.get('confidence_upper', 0)*100:+.2f}%

**Modèle:** {data.get('model', 'N/A')}

⚠️ {data.get('disclaimer', 'Les performances passées ne garantissent pas les résultats futurs.')}
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur prédiction: {e}"

# =============================================================================
# HANDLERS - Research & Education
# =============================================================================

def handle_research_query(prompt: str) -> str:
    """Rechercher info sur un ticker"""
    ticker = extract_ticker_from_prompt(prompt)
    if not ticker:
        return "🔍 **Précisez un ticker** (ex: AAPL, MSFT, GOOGL)"
    
    try:
        data = yahoo.get_ticker_data(ticker, period='1y')
        info = yahoo.get_ticker_info(ticker)
        
        if data is None or data.empty:
            return f"❌ Données indisponibles pour {ticker}"
        
        current = float(data['Close'].iloc[-1])
        start = float(data['Close'].iloc[0])
        perf_1y = (current - start) / start * 100
        
        name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        emoji = "🟢" if perf_1y > 0 else "🔴"
        
        response = f"""{emoji} **{ticker} - {name}**

**Prix actuel:** ${current:.2f}
**Performance 1 an:** {perf_1y:+.2f}%

**Informations:**
- Secteur: {sector}
- Capitalisation: ${market_cap/1e9:.2f}B
- P/E Ratio: {pe_ratio:.2f}

💡 *Utilisez "Analyse le sentiment sur {ticker}" pour plus de détails*
"""
        
        return response
    
    except Exception as e:
        return f"❌ Erreur recherche {ticker}: {e}"
        
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
         'amd': 'AMD',
        'intel': 'INTC',
        'ibm': 'IBM',
        'ionq': 'IONQ',
        'rigetti': 'RGTI',
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
# ROUTER PRINCIPAL avec Claude AI
# =============================================================================

def route_query(prompt: str) -> str:
    """Router intelligent avec détection d'intention"""
    p = prompt.lower()
    
    # 1. Portfolio queries
    if any(w in p for w in ['mon portfolio', 'mes portfolios', 'portefeuille']):
        if 'analyse' in p or 'détail' in p or 'risque' in p:
            # Extraire nom de portfolio si mentionné
            words = prompt.split()
            for i, word in enumerate(words):
                if word.lower() in ['portfolio', 'portefeuille'] and i + 1 < len(words):
                    portfolio_name = words[i + 1]
                    return handle_portfolio_analysis(portfolio_name)
        return handle_portfolio_query(prompt)
    
    # 2. Market Intelligence (NEW v4.0)
    if any(w in p for w in ['marché', 'market', 'indices', 'vue d\'ensemble']):
        region = 'US'  # Default
        if 'europ' in p or 'eu' in p:
            region = 'EU'
        elif 'asi' in p:
            region = 'ASIA'
        elif 'global' in p:
            region = 'GLOBAL'
        return handle_market_overview(region)
    
    # 3. Sector Analysis (NEW v4.0)
    if any(w in p for w in ['secteur', 'sector', 'industrie']):
        sectors_map = {
            'tech': 'technology',
            'technolog': 'technology',
            'semicond': 'semiconductors',
            'quantum': 'quantum',
            'ai': 'ai_ml',
            'ml': 'ai_ml',
            'santé': 'healthcare',
            'health': 'healthcare',
            'finance': 'finance',
            'énergie': 'energy',
            'energy': 'energy',
            'consommat': 'consumer',
            'consumer': 'consumer',
        }
        
        sector = None
        for key, value in sectors_map.items():
            if key in p:
                sector = value
                break
        
        if sector:
            return handle_sector_analysis(sector)
        else:
            return "🔍 Secteurs disponibles: technology, semiconductors, quantum, ai_ml, healthcare, finance, energy, consumer"
    
    # 4. Sentiment Analysis (NEW v4.0)
    if 'sentiment' in p:
        ticker = extract_ticker_from_prompt(prompt)
        if ticker:
            return handle_sentiment_analysis(ticker)
        else:
            return "🔍 Précisez un ticker pour l'analyse sentiment (ex: 'sentiment sur AAPL')"
    
    # 5. Comparison (NEW v4.0)
    if any(w in p for w in ['compare', 'comparaison', 'vs', 'versus']):
        # Extraire tickers à comparer
        tickers = re.findall(r'\b([A-Z]{2,5})\b', prompt.upper())
        if len(tickers) >= 2:
            return handle_compare_markets(tickers[:5])  # Max 5
        else:
            return "🔍 Précisez au moins 2 tickers à comparer (ex: 'Compare AAPL et MSFT')"
    
    # 6. Backtesting (NEW v4.0)
    if any(w in p for w in ['backtest', 'test', 'historique']):
        # Extraire dates si possible
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Extraire nom portfolio
        words = prompt.split()
        portfolio_name = None
        for i, word in enumerate(words):
            if word.lower() in ['portfolio', 'portefeuille'] and i + 1 < len(words):
                portfolio_name = words[i + 1]
                break
        
        if portfolio_name:
            return handle_backtest(portfolio_name, start_date, end_date)
        else:
            return "🔍 Précisez le nom du portfolio à backtester"
    
    # 7. Predictions (NEW v4.0)
    if any(w in p for w in ['prédis', 'predict', 'prévision', 'futur']):
        words = prompt.split()
        portfolio_name = None
        horizon = '3mo'
        
        for i, word in enumerate(words):
            if word.lower() in ['portfolio', 'portefeuille'] and i + 1 < len(words):
                portfolio_name = words[i + 1]
        
        if '1 mois' in p or '1mo' in p:
            horizon = '1mo'
        elif '6 mois' in p or '6mo' in p:
            horizon = '6mo'
        elif '1 an' in p or '1y' in p:
            horizon = '1y'
        
        if portfolio_name:
            return handle_prediction(portfolio_name, horizon)
        else:
            return "🔍 Précisez le nom du portfolio pour la prédiction"
    
    # 8. Research (ticker info)
    if any(w in p for w in ['recherche', 'rechercher', 'analyse', 'info', 'action']):
        return handle_research_query(prompt)
    
    # 9. Education
    if any(w in p for w in ['explique', 'qu\'est-ce', 'comment', 'définition', 'c\'est quoi']):
        return handle_education_query(prompt)
    
    # 10. RAG Search (si disponible)
    rag = get_rag_engine()
    if rag and any(w in p for w in ['cherche dans', 'trouve dans', 'documents']):
        try:
            results = rag.search(prompt, top_k=3)
            if results:
                response = "📚 **Résultats de recherche:**\n\n"
                for i, result in enumerate(results, 1):
                    response += f"{i}. {result.get('content', '')[:200]}...\n\n"
                return response
        except:
            pass
    
    # 11. Web Search (si disponible)
    web = get_web_search_engine()
    if web and any(w in p for w in ['cherche sur le web', 'web search', 'internet']):
        try:
            results = web.search(prompt, num_results=3)
            if results:
                response = "🌐 **Résultats web:**\n\n"
                for result in results:
                    response += f"**{result.get('title', 'N/A')}**\n"
                    response += f"{result.get('snippet', 'N/A')}\n"
                    response += f"🔗 {result.get('url', 'N/A')}\n\n"
                return response
        except:
            pass
    
    # 12. Fallback: Claude AI généraliste
    return handle_general_query_with_claude(prompt)

def handle_general_query_with_claude(prompt: str) -> str:
    """Utiliser Claude AI pour questions générales"""
    if not ANTHROPIC_API_KEY:
        return """🤖 **Configuration requise**
        
Pour utiliser l'assistant IA, configurez votre clé API Anthropic dans `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

Obtenez votre clé sur https://console.anthropic.com/
"""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Contexte système
        system_prompt = f"""Tu es Φ AI Assistant, un assistant financier expert pour la plateforme PyManager.

**Capacités:**
- Analyse de portfolios et gestion de risques
- Market intelligence (US, EU, ASIA, GLOBAL)
- Analyse sectorielle (tech, semiconductors, quantum computing, AI/ML, etc.)
- Backtesting et prédictions ML
- Éducation financière
- Sentiment analysis

**User ID:** {user_id}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

**Instructions:**
1. Réponds de manière concise et professionnelle
2. Utilise des emojis pour la lisibilité
3. Cite tes sources si tu utilises des données
4. Propose des actions concrètes quand pertinent
5. Si la question nécessite des données MCP, suggère la commande appropriée

**Important:** Tu ne peux pas exécuter directement les tools MCP, mais tu peux suggérer à l'utilisateur d'utiliser les commandes appropriées."""

        # Préparer messages avec contexte
        messages = []
        
        # Ajouter contexte conversation (derniers 5 messages)
        context_messages = st.session_state.conversation_context[-10:]
        for msg in context_messages:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Ajouter prompt actuel s'il n'est pas déjà dans le contexte
        if not messages or messages[-1]['content'] != prompt:
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        # Appeler Claude
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
            temperature=0.7
        )
        
        return response.content[0].text
    
    except anthropic.AuthenticationError:
        return "❌ **Erreur d'authentification**\n\nVérifiez votre clé API Anthropic dans secrets.toml"
    except anthropic.RateLimitError:
        return "⏱️ **Rate limit atteint**\n\nEssayez à nouveau dans quelques instants"
    except Exception as e:
        return f"⚠️ **Erreur IA:** {str(e)}"


