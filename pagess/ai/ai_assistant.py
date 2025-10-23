# pages/ai_assistant.py
"""
Page AI Assistant style ChatGPT
"""

import streamlit as st
from database import get_portfolios


def render_ai_assistant():
    """Page AI Assistant style ChatGPT"""
    
    # Si pas d'historique, afficher l'écran d'accueil
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="ai-welcome">
            <div class="ai-logo">🤖</div>
            <div class="ai-title">Φ AI Assistant</div>
            <div class="ai-subtitle">Your intelligent portfolio advisor</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggestions de prompts
        st.markdown("### Start with a suggestion")
        
        col1, col2 = st.columns(2)
        
        suggestions = [
            "Analyze my portfolio performance",
            "What are the best growth stocks?",
            "Explain the Sharpe ratio",
            "How to diversify my investments?"
        ]
        
        for i, suggestion in enumerate(suggestions):
            with col1 if i % 2 == 0 else col2:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.chat_history.append({'role': 'user', 'content': suggestion})
                    response = generate_ai_response(suggestion)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
    
    else:
        # Afficher l'historique du chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
    
    # Input fixé en bas
    if prompt := st.chat_input("Message Φ Assistant...", key="ai_chat_input"):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        # Réponse basique (à remplacer par OpenAI)
        response = generate_ai_response(prompt)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()


def generate_ai_response(prompt):
    """Génère une réponse AI (placeholder pour OpenAI integration)"""
    
    # Réponses contextuelles basiques
    prompt_lower = prompt.lower()
    
    if "portfolio" in prompt_lower or "portfolios" in prompt_lower:
        try:
            portfolios = list(get_portfolios())
            if portfolios:
                total_value = sum([p.get('amount', 0) for p in portfolios])
                portfolio_list = ", ".join([p['name'] for p in portfolios])
                return f"""📊 **Portfolio Overview**

You currently have **{len(portfolios)} portfolio(s)** with a total value of **${total_value:,.2f}**.

Your portfolios: {portfolio_list}

Would you like me to analyze a specific portfolio or provide recommendations?"""
            else:
                return """📁 **No Portfolios Found**

You don't have any portfolios yet. Would you like help creating one?

I can guide you through:
- Building a diversified portfolio using Modern Portfolio Theory
- Selecting optimal assets based on risk tolerance
- Calculating expected returns and volatility"""
        except:
            pass
    
    if "stock" in prompt_lower or "ticker" in prompt_lower or "analyze" in prompt_lower:
        return """📈 **Stock Analysis**

I can help you analyze stocks! Here's what I can do:

- **Fundamental Analysis**: P/E ratio, earnings, dividends
- **Technical Indicators**: Moving averages, RSI, MACD
- **Price Trends**: Historical performance and patterns
- **Comparison**: Compare multiple stocks

Try asking:
- "Analyze AAPL"
- "Compare MSFT and GOOGL"
- "What's the RSI for TSLA?"
"""
    
    if "sharpe" in prompt_lower or "ratio" in prompt_lower:
        return """📊 **Sharpe Ratio Explained**

The Sharpe Ratio measures **risk-adjusted returns**. It tells you how much return you're getting for each unit of risk taken.

**Formula**: (Return - Risk-free rate) / Standard Deviation

**Interpretation**:
- **> 1.0**: Good risk-adjusted performance ✅
- **0.5 - 1.0**: Acceptable but could be better ⚠️
- **< 0.5**: Poor risk-adjusted returns ❌

Higher is better! It helps you compare investments with different risk levels."""
    
    if "diversif" in prompt_lower:
        return """🎯 **Portfolio Diversification**

Diversification reduces risk by spreading investments across different assets.

**Key Strategies**:
1. **Asset Classes**: Mix stocks, bonds, commodities
2. **Sectors**: Technology, Healthcare, Finance, etc.
3. **Geography**: Domestic and international markets
4. **Market Cap**: Large, mid, and small-cap stocks

**Benefits**:
- Reduces portfolio volatility
- Protects against sector-specific risks
- Improves risk-adjusted returns

I can help you build a diversified portfolio using Modern Portfolio Theory!"""
    
    if "risk" in prompt_lower:
        return """⚠️ **Investment Risk**

Understanding risk is crucial for successful investing:

**Types of Risk**:
- **Market Risk**: Overall market movements
- **Volatility**: Price fluctuations
- **Company Risk**: Specific to individual stocks
- **Sector Risk**: Industry-specific factors

**Measuring Risk**:
- **Standard Deviation**: Volatility measure
- **Beta**: Sensitivity to market movements
- **VaR (Value at Risk)**: Potential losses

Would you like to assess the risk of your current portfolio?"""
    
    if "help" in prompt_lower or "what can you do" in prompt_lower:
        return """🤖 **How I Can Help**

I'm your intelligent portfolio advisor! Here's what I can do:

📊 **Portfolio Management**
- Analyze portfolio performance
- Optimize asset allocation
- Calculate risk metrics (Sharpe, Beta, VaR)

📈 **Stock Analysis**
- Real-time stock data
- Technical indicators
- Fundamental analysis

💡 **Investment Guidance**
- Diversification strategies
- Risk assessment
- Market insights

🎓 **Education**
- Explain financial concepts
- Investment terminology
- Strategy recommendations

Just ask me anything about investing, portfolios, or stocks!"""
    
    # Réponse par défaut
    return f"""Thank you for your question! 

I received: *"{prompt}"*

**Note**: Full AI capabilities require OpenAI API integration. 

For now, I can help with:
- Portfolio analysis and management
- Stock information and trends
- Investment concepts and strategies
- Risk assessment

Try asking about specific portfolios, stocks, or financial concepts!"""
