# pages/ai_assistant.py
"""
AI Assistant - Intelligent Portfolio Advisor
Multi-agent system with MCP integration
"""

import streamlit as st
import anthropic
import json
from datetime import datetime
from uiconfig import get_theme_colors
from dataprovider import yahoo

try:
    from database import get_portfolios
except:
    def get_portfolios():
        return []


# Configuration
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")  # À configurer dans .streamlit/secrets.toml


def render_ai_assistant():
    """AI Assistant principal avec multi-agent orchestration"""
    theme = get_theme_colors()
    
    # Header
    st.html(f"""
    <div style="
        background: {theme['gradient_primary']};
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">
            🤖 AI Assistant
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            Your intelligent portfolio advisor powered by Claude
        </p>
    </div>
    """)
    
    # Initialiser l'historique
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar avec capacités
    with st.sidebar:
        st.markdown("### 🎯 AI Capabilities")
        
        capabilities = {
            "📊 Portfolio Analysis": "Analyze your portfolios, performance, and allocation",
            "⭐ Watchlist Insights": "Get insights on your watchlist stocks",
            "🔍 Company Research": "Deep dive research on any company",
            "📈 Market Screening": "Find stocks matching your criteria",
            "📝 Report Generation": "Generate detailed financial reports",
            "🎓 Finance Education": "Learn about investment concepts",
            "💡 Recommendations": "Get personalized investment advice"
        }
        
        for icon_title, desc in capabilities.items():
            with st.expander(icon_title):
                st.caption(desc)
        
        st.divider()
        
        # Bouton reset
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Stats
        if st.session_state.chat_history:
            st.metric("Messages", len(st.session_state.chat_history))
    
    # Écran d'accueil si pas d'historique
    if not st.session_state.chat_history:
        render_welcome_screen(theme)
    else:
        # Afficher l'historique
        render_chat_history(theme)
    
    # Input utilisateur (toujours visible en bas)
    render_chat_input(theme)


def render_welcome_screen(theme):
    """Écran d'accueil avec suggestions"""
    
    st.html(f"""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h2 style="color: {theme['text_primary']}; margin-bottom: 0.5rem;">
            Φ AI Assistant
        </h2>
        <p style="color: {theme['text_secondary']}; font-size: 1.1rem;">
            Your intelligent portfolio advisor
        </p>
    </div>
    """)
    
    st.markdown("### 💡 Quick Suggestions")
    
    suggestions = [
        {
            "icon": "📊",
            "title": "Analyze my portfolio",
            "prompt": "Analyze my current portfolio performance and give me recommendations"
        },
        {
            "icon": "🔍",
            "title": "Research a company",
            "prompt": "Research Apple Inc. (AAPL) and tell me if it's a good investment"
        },
        {
            "icon": "📈",
            "title": "Screen for growth stocks",
            "prompt": "Find me high-growth technology stocks with P/E < 30"
        },
        {
            "icon": "📝",
            "title": "Generate a report",
            "prompt": "Generate a detailed financial report for my portfolio"
        },
        {
            "icon": "🎓",
            "title": "Explain a concept",
            "prompt": "Explain the Sharpe Ratio and how to use it"
        },
        {
            "icon": "⭐",
            "title": "Watchlist insights",
            "prompt": "Give me insights on my watchlist stocks"
        }
    ]
    
    cols = st.columns(2)
    
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(
                f"{suggestion['icon']} {suggestion['title']}", 
                key=f"sug_{idx}",
                use_container_width=True
            ):
                process_user_message(suggestion['prompt'])


def render_chat_history(theme):
    """Affiche l'historique du chat"""
    
    for idx, message in enumerate(st.session_state.chat_history):
        role = message['role']
        content = message['content']
        
        if role == 'user':
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Si le message contient des données structurées
                if isinstance(content, dict):
                    render_structured_response(content, theme)
                else:
                    st.markdown(content)
                
                # Boutons d'action si disponibles
                if 'actions' in message:
                    render_action_buttons(message['actions'], idx)


def render_chat_input(theme):
    """Input utilisateur"""
    
    if prompt := st.chat_input("Ask me anything about investing, portfolios, or markets...", key="ai_input"):
        process_user_message(prompt)


def process_user_message(prompt):
    """Traite le message utilisateur avec orchestration intelligente"""
    
    # Ajouter le message utilisateur
    st.session_state.chat_history.append({
        'role': 'user',
        'content': prompt
    })
    
    # Afficher un spinner pendant le traitement
    with st.spinner("🤖 AI is thinking..."):
        # Déterminer l'intent et router vers le bon agent
        response = orchestrate_request(prompt)
    
    # Ajouter la réponse
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response
    })
    
    st.rerun()


def orchestrate_request(prompt):
    """
    Orchestrateur intelligent qui route les requêtes vers les bons agents
    """
    
    prompt_lower = prompt.lower()
    
    # 1. Requêtes sur données internes (Portfolio, Watchlist)
    if any(word in prompt_lower for word in ['portfolio', 'my stocks', 'my investments', 'watchlist']):
        return handle_internal_data_query(prompt)
    
    # 2. Recherche sur entreprise spécifique
    elif any(word in prompt_lower for word in ['research', 'analyze', 'tell me about']) and \
         any(word in prompt_lower for word in ['company', 'stock', 'aapl', 'msft', 'tsla', 'googl']):
        return handle_company_research(prompt)
    
    # 3. Screening de marché
    elif any(word in prompt_lower for word in ['find', 'screen', 'search for', 'discover']):
        return handle_market_screening(prompt)
    
    # 4. Génération de rapport
    elif any(word in prompt_lower for word in ['report', 'generate', 'write', 'create report']):
        return handle_report_generation(prompt)
    
    # 5. Questions éducatives
    elif any(word in prompt_lower for word in ['explain', 'what is', 'how to', 'teach me']):
        return handle_educational_query(prompt)
    
    # 6. Requête générale - utiliser Claude avec tous les tools
    else:
        return handle_general_query(prompt)


# =============================================================================
# Handlers pour chaque type de requête
# =============================================================================

def handle_internal_data_query(prompt):
    """Handler pour les requêtes sur données internes"""
    
    try:
        # Récupérer les portfolios
        portfolios = list(get_portfolios())
        watchlist = st.session_state.get('watchlist', [])
        
        # Construire le contexte
        context = {
            'portfolios': portfolios,
            'watchlist': watchlist,
            'total_value': sum([p.get('amount', 0) for p in portfolios]),
            'num_portfolios': len(portfolios)
        }
        
        # Appeler Claude avec le contexte
        if not ANTHROPIC_API_KEY:
            return generate_fallback_response(prompt, context)
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a financial advisor. Here is the user's portfolio data:

Portfolio Summary:
- Number of portfolios: {context['num_portfolios']}
- Total value: ${context['total_value']:,.2f}
- Portfolios: {json.dumps(portfolios, indent=2)}
- Watchlist: {watchlist}

User question: {prompt}

Provide a detailed, helpful response with specific insights and recommendations."""
                }
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Error accessing portfolio data: {str(e)}\n\nPlease try again or contact support."


def handle_company_research(prompt):
    """Handler pour la recherche d'entreprise"""
    
    # Extraire le ticker si mentionné
    ticker = extract_ticker_from_prompt(prompt)
    
    if not ticker:
        return "🔍 Please specify a company ticker (e.g., AAPL, MSFT, TSLA) for detailed research."
    
    try:
        # Récupérer les données
        data = yahoo.get_ticker_data(ticker, period='1y')
        info = yahoo.get_ticker_info(ticker)
        
        if data is None or data.empty:
            return f"❌ Could not fetch data for {ticker}. Please verify the ticker symbol."
        
        # Calculer des métriques
        current_price = float(data['Close'].iloc[-1])
        ytd_return = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
        volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
        
        # Construire le contexte
        company_context = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'ytd_return': ytd_return,
            'volatility': volatility,
            'description': info.get('longBusinessSummary', '')[:500]
        }
        
        # Utiliser Claude pour l'analyse
        if not ANTHROPIC_API_KEY:
            return format_company_research_fallback(company_context)
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2500,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a financial analyst. Provide a comprehensive research report on this company:

Company: {company_context['name']} ({ticker})
Sector: {company_context['sector']} | Industry: {company_context['industry']}

Financial Metrics:
- Current Price: ${company_context['price']:.2f}
- Market Cap: ${company_context['market_cap']:,.0f}
- P/E Ratio: {company_context['pe_ratio']:.2f}
- Dividend Yield: {company_context['dividend_yield']*100:.2f}%
- YTD Return: {company_context['ytd_return']:.2f}%
- Volatility: {company_context['volatility']:.2f}%

Description: {company_context['description']}

User question: {prompt}

Provide:
1. Business overview
2. Financial health analysis
3. Valuation assessment
4. Investment recommendation (Buy/Hold/Sell)
5. Risk factors
6. Growth catalysts"""
                }
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"❌ Error researching {ticker}: {str(e)}"


def handle_market_screening(prompt):
    """Handler pour le screening de marché"""
    
    screening_prompt = f"""Based on this request: "{prompt}"

I can help you screen the market. Here's what I found:

🔍 **Screening Criteria Detected:**
"""
    
    # Extraire les critères du prompt
    criteria = []
    
    if 'growth' in prompt.lower():
        criteria.append("- High growth stocks (revenue growth > 20%)")
    
    if 'value' in prompt.lower():
        criteria.append("- Value stocks (low P/E ratio < 15)")
    
    if 'dividend' in prompt.lower():
        criteria.append("- Dividend stocks (yield > 3%)")
    
    if 'tech' in prompt.lower() or 'technology' in prompt.lower():
        criteria.append("- Technology sector")
    
    if criteria:
        screening_prompt += "\n".join(criteria)
    else:
        screening_prompt += "- General market scan"
    
    screening_prompt += """

📊 **Recommended Actions:**
1. Go to the **Screener** page for advanced filtering
2. Set your specific criteria (P/E, market cap, sector, etc.)
3. Run the scan to get real-time results

Would you like me to help you refine your screening criteria?"""
    
    return screening_prompt


def handle_report_generation(prompt):
    """Handler pour la génération de rapports"""
    
    try:
        portfolios = list(get_portfolios())
        
        if not portfolios:
            return "📝 No portfolios found. Please create a portfolio first to generate a report."
        
        total_value = sum([p.get('amount', 0) for p in portfolios])
        
        report = f"""# 📊 Portfolio Report
*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*

---

## Executive Summary

**Total Assets Under Management:** ${total_value:,.2f}  
**Number of Portfolios:** {len(portfolios)}  
**Report Period:** Last 30 days

---

## Portfolio Overview

"""
        
        for idx, pf in enumerate(portfolios, 1):
            name = pf.get('name', f'Portfolio {idx}')
            value = pf.get('amount', 0)
            model = pf.get('model', 'N/A')
            
            report += f"""### {idx}. {name}
- **Value:** ${value:,.2f}
- **Model:** {model.title()}
- **Allocation:** {(value/total_value*100):.1f}% of total

"""
        
        report += """---

## Recommendations

Based on your current portfolio:

1. **Diversification:** Consider adding international exposure
2. **Risk Management:** Review your position sizes
3. **Rebalancing:** Consider quarterly rebalancing
4. **Tax Efficiency:** Optimize for tax-loss harvesting

---

## Next Steps

- Review individual holdings performance
- Assess risk metrics (Sharpe ratio, Beta)
- Consider adding protective positions
- Schedule quarterly review

---

*This report is generated automatically and should not be considered as financial advice.*
"""
        
        return report
        
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"


def handle_educational_query(prompt):
    """Handler pour les questions éducatives"""
    
    # Base de connaissances simplifiée
    knowledge_base = {
        'sharpe ratio': """📊 **Sharpe Ratio Explained**

The Sharpe Ratio measures **risk-adjusted returns**. It answers: "How much return am I getting per unit of risk?"

**Formula:**
```
Sharpe Ratio = (Portfolio Return - Risk-free Rate) / Standard Deviation
```

**Interpretation:**
- **> 2.0**: Excellent ⭐⭐⭐
- **1.0 - 2.0**: Very Good ⭐⭐
- **0.5 - 1.0**: Acceptable ⭐
- **< 0.5**: Poor ❌

**Example:**
- Portfolio A: 15% return, 10% volatility → Sharpe = 1.3
- Portfolio B: 20% return, 25% volatility → Sharpe = 0.7
- **Portfolio A is better** (higher risk-adjusted return)

**Key Insight:** Higher Sharpe = Better risk-adjusted performance""",
        
        'diversification': """🎯 **Portfolio Diversification**

Diversification = "Don't put all your eggs in one basket"

**Why Diversify?**
- Reduces overall portfolio risk
- Smooths out returns
- Protects against sector-specific shocks

**How to Diversify:**

1. **Asset Classes**
   - Stocks (60%)
   - Bonds (30%)
   - Commodities/Cash (10%)

2. **Sectors**
   - Technology, Healthcare, Finance, etc.
   - No single sector > 25%

3. **Geography**
   - US (60%), International (30%), Emerging (10%)

4. **Market Cap**
   - Large-cap (50%), Mid-cap (30%), Small-cap (20%)

**Rule of Thumb:** 15-30 different stocks is optimal""",
        
        'beta': """📈 **Beta Explained**

Beta measures a stock's **volatility relative to the market**.

**Understanding Beta:**
- **Beta = 1.0**: Moves with the market
- **Beta > 1.0**: More volatile than market (aggressive)
- **Beta < 1.0**: Less volatile than market (defensive)
- **Beta < 0**: Moves opposite to market (rare)

**Examples:**
- Tech stocks: Beta ~ 1.2-1.5 (high volatility)
- Utilities: Beta ~ 0.5-0.8 (low volatility)
- Gold: Beta ~ 0 or negative (hedge)

**Practical Use:**
- High Beta = Higher risk & reward potential
- Low Beta = Defensive, stable returns""",
        
        'p/e ratio': """💰 **P/E Ratio (Price-to-Earnings)**

P/E Ratio = Stock Price / Earnings Per Share

**What it tells you:**
How much investors pay for each dollar of earnings.

**Interpretation:**
- **Low P/E (< 15)**: Potentially undervalued or slow growth
- **Medium P/E (15-25)**: Fair valuation
- **High P/E (> 25)**: Growth expectations or overvalued

**Important:**
- Compare within same sector
- Tech stocks typically have higher P/E
- Consider PEG ratio (P/E / Growth rate)

**Example:**
- Stock A: Price $100, EPS $5 → P/E = 20
- Stock B: Price $50, EPS $5 → P/E = 10
- Stock B is "cheaper" relatively"""
    }
    
    # Chercher dans la base de connaissances
    for keyword, explanation in knowledge_base.items():
        if keyword in prompt.lower():
            return explanation
    
    # Réponse générique si pas trouvé
    return """🎓 **Finance Education**

I can explain many financial concepts! Try asking about:

- **Risk Metrics**: Sharpe Ratio, Beta, Standard Deviation
- **Valuation**: P/E Ratio, P/B Ratio, PEG Ratio
- **Portfolio Theory**: Diversification, Modern Portfolio Theory, Efficient Frontier
- **Technical Analysis**: Moving Averages, RSI, MACD
- **Bonds**: Yield, Duration, Convexity
- **Options**: Calls, Puts, Greeks

What would you like to learn about?"""


def handle_general_query(prompt):
    """Handler pour les requêtes générales"""
    
    if not ANTHROPIC_API_KEY:
        return """🤖 **AI Assistant**

I'm here to help! I can assist with:

- 📊 Portfolio analysis and optimization
- 🔍 Company research and analysis
- 📈 Market screening and stock discovery
- 📝 Financial report generation
- 🎓 Investment education
- 💡 Personalized recommendations

**Note:** Full AI capabilities require API configuration.

Try asking specific questions about your portfolios, stocks, or financial concepts!"""
    
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a helpful financial advisor and portfolio manager.

User question: {prompt}

Provide a clear, helpful response. If the question is about investing, portfolios, or finance, give actionable advice. If it's outside your domain, politely redirect to financial topics."""
                }
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nPlease try rephrasing your question."


# =============================================================================
# Helper Functions
# =============================================================================

def extract_ticker_from_prompt(prompt):
    """Extrait un ticker symbol du prompt"""
    
    # Liste de tickers communs
    common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                      'JPM', 'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD']
    
    prompt_upper = prompt.upper()
    
    for ticker in common_tickers:
        if ticker in prompt_upper:
            return ticker
    
    # Chercher pattern (XXX) ou $XXX
    import re
    pattern = r'\b([A-Z]{1,5})\b|\$([A-Z]{1,5})'
    matches = re.findall(pattern, prompt_upper)
    
    if matches:
        return matches[0][0] or matches[0][1]
    
    return None


def generate_fallback_response(prompt, context):
    """Génère une réponse de fallback sans API"""
    
    num_portfolios = context.get('num_portfolios', 0)
    total_value = context.get('total_value', 0)
    portfolios = context.get('portfolios', [])
    
    if num_portfolios == 0:
        return """📁 **No Portfolios Found**

You don't have any portfolios yet. Let me help you get started!

**To create a portfolio:**
1. Go to the **Portfolio** page
2. Click "Create New Portfolio"
3. Choose a strategy (Growth, Income, Balanced)
4. Add your assets

Would you like guidance on portfolio construction?"""
    
    response = f"""📊 **Portfolio Overview**

You currently have **{num_portfolios} portfolio(s)** with a total value of **${total_value:,.2f}**.

**Your Portfolios:**
"""
    
    for idx, pf in enumerate(portfolios, 1):
        name = pf.get('name', f'Portfolio {idx}')
        value = pf.get('amount', 0)
        model = pf.get('model', 'N/A')
        
        response += f"""
{idx}. **{name}**
   - Value: ${value:,.2f}
   - Model: {model.title()}
   - Allocation: {(value/total_value*100):.1f}%
"""
    
    response += """
**💡 Recommendations:**
- Review your asset allocation
- Consider rebalancing if needed
- Monitor individual positions
- Set up alerts for major changes

Would you like a detailed analysis of any specific portfolio?"""
    
    return response


def format_company_research_fallback(context):
    """Formate la recherche d'entreprise sans API"""
    
    return f"""🔍 **Company Research: {context['name']}**

**Ticker:** {context['ticker']}  
**Sector:** {context['sector']} | **Industry:** {context['industry']}

---

**📊 Current Metrics:**
- **Price:** ${context['price']:.2f}
- **Market Cap:** ${context['market_cap']:,.0f}
- **P/E Ratio:** {context['pe_ratio']:.2f}
- **Dividend Yield:** {context['dividend_yield']*100:.2f}%

**📈 Performance:**
- **YTD Return:** {context['ytd_return']:.2f}%
- **Volatility:** {context['volatility']:.2f}%

---

**🏢 Business Overview:**
{context['description']}

---

**💡 Quick Assessment:**

**Valuation:** {'Expensive' if context['pe_ratio'] > 25 else 'Fair' if context['pe_ratio'] > 15 else 'Cheap'}  
**Volatility:** {'High' if context['volatility'] > 30 else 'Moderate' if context['volatility'] > 20 else 'Low'}  
**Income:** {'Yes' if context['dividend_yield'] > 0.02 else 'No'} dividend

---

For detailed analysis, consider:
- Revenue growth trends
- Competitive position
- Future catalysts
- Risk factors

Would you like me to analyze specific aspects?"""


def render_structured_response(data, theme):
    """Affiche une réponse structurée"""
    
    if 'metrics' in data:
        cols = st.columns(len(data['metrics']))
        for idx, (label, value) in enumerate(data['metrics'].items()):
            with cols[idx]:
                st.metric(label, value)
    
    if 'content' in data:
        st.markdown(data['content'])


def render_action_buttons(actions, message_idx):
    """Affiche des boutons d'action"""
    
    cols = st.columns(len(actions))
    
    for idx, action in enumerate(actions):
        with cols[idx]:
            if st.button(action['label'], key=f"action_{message_idx}_{idx}"):
                # Exécuter l'action
                if action['type'] == 'navigate':
                    st.session_state.current_page = action['target']
                    st.rerun()
