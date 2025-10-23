# pages/fund_explorer.py
"""
Fund Explorer - Analyse de fonds (ETFs, Mutual Funds)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataprovider import yahoo
from uiconfig import get_theme_colors


POPULAR_ETFS = {
    "📈 Broad Market": {
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ (NASDAQ-100)',
        'IWM': 'iShares Russell 2000 ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
    },
    "🌍 International": {
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF',
        'VEA': 'Vanguard FTSE Developed Markets ETF',
    },
    "🏭 Sector": {
        'XLK': 'Technology Select Sector SPDR',
        'XLF': 'Financial Select Sector SPDR',
        'XLE': 'Energy Select Sector SPDR',
        'XLV': 'Health Care Select Sector SPDR',
    },
    "💰 Bond": {
        'AGG': 'iShares Core US Aggregate Bond ETF',
        'BND': 'Vanguard Total Bond Market ETF',
        'TLT': 'iShares 20+ Year Treasury Bond ETF',
    }
}


def render_fund_explorer():
    """Explorer pour les fonds"""
    theme = get_theme_colors()
    
    st.markdown("### 🏦 Fund Explorer")
    st.caption("Analyze ETFs and Mutual Funds")
    
    # Recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "🔍 Search Fund (ETF/Mutual Fund)",
            placeholder="Enter ticker (e.g., SPY, QQQ, VTI)",
            key="fund_search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Analyze", use_container_width=True, type="primary")
    
    # Suggestions populaires
    if not ticker:
        st.markdown("#### 💡 Popular ETFs")
        
        for category, funds in POPULAR_ETFS.items():
            with st.expander(category):
                for symbol, name in funds.items():
                    if st.button(f"{symbol} - {name}", key=f"etf_{symbol}"):
                        st.session_state.fund_search_value = symbol
                        st.rerun()
        return
    
    if ticker and (search_button or ticker):
        ticker = ticker.upper().strip()
        
        try:
            with st.spinner(f"Loading {ticker}..."):
                # Charger données
                data = yahoo.get_ticker_data(ticker, period='1y')
                info = yahoo.get_ticker_info(ticker)
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return
                
                # Header
                render_fund_header(ticker, data, info, theme)
                
                st.divider()
                
                # Tabs
                tabs = st.tabs([
                    "📈 Chart",
                    "ℹ️ Overview",
                    "📊 Holdings",
                    "📉 Performance",
                    "💰 Distributions"
                ])
                
                with tabs[0]:
                    render_fund_chart_tab(ticker, data, theme)
                
                with tabs[1]:
                    render_fund_overview_tab(ticker, info, theme)
                
                with tabs[2]:
                    render_fund_holdings_tab(ticker, info, theme)
                
                with tabs[3]:
                    render_fund_performance_tab(ticker, data, theme)
                
                with tabs[4]:
                    render_fund_distributions_tab(ticker, theme)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def render_fund_header(ticker, data, info, theme):
    """Header du fonds"""
    
    current_price = float(data['Close'].iloc[-1])
    
    if len(data) > 1:
        prev_price = float(data['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price) * 100
    else:
        change_pct = 0
    
    fund_name = info.get('longName', ticker)
    category = info.get('category', 'N/A')
    aum = info.get('totalAssets', 0)
    expense_ratio = info.get('annualReportExpenseRatio', 0) * 100 if info.get('annualReportExpenseRatio') else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"### {ticker}")
        st.caption(fund_name)
        st.metric(
            label="NAV / Price",
            value=f"${current_price:.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="AUM",
            value=format_large_number(aum)
        )
        st.caption(f"Category: {category}")
    
    with col3:
        st.metric(
            label="Expense Ratio",
            value=f"{expense_ratio:.2f}%"
        )
    
    with col4:
        ytd_return = calculate_ytd_return(data)
        st.metric(
            label="YTD Return",
            value=f"{ytd_return:+.2f}%"
        )


def render_fund_chart_tab(ticker, data, theme):
    """Graphique du fonds"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            key="fund_period"
        )
    
    # Recharger
    data = yahoo.get_ticker_data(ticker, period=period)
    
    if data is not None and not data.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#6366F1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        fig.update_layout(
            
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=500,
             plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary']),
           title=dict(text=f"{ticker} Price Chart", font=dict(size=18, color=theme['text_primary'])),
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_fund_overview_tab(ticker, info, theme):
    """Overview du fonds"""
    
    st.markdown("### 📋 Fund Overview")
    
    description = info.get('longBusinessSummary', 'No description available')
    st.write(description)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏦 Fund Details")
        st.markdown(f"**Fund Family:** {info.get('fundFamily', 'N/A')}")
        st.markdown(f"**Category:** {info.get('category', 'N/A')}")
        st.markdown(f"**Inception Date:** {info.get('fundInceptionDate', 'N/A')}")
        st.markdown(f"**Legal Type:** {info.get('legalType', 'N/A')}")
    
    with col2:
        st.markdown("#### 💰 Costs & Minimums")
        expense_ratio = info.get('annualReportExpenseRatio', 0) * 100 if info.get('annualReportExpenseRatio') else 0
        st.markdown(f"**Expense Ratio:** {expense_ratio:.2f}%")
        st.markdown(f"**Min Investment:** ${info.get('minInvestment', 0):,.0f}")
        st.markdown(f"**Yield:** {info.get('yield', 0) * 100:.2f}%" if info.get('yield') else "**Yield:** N/A")


def render_fund_holdings_tab(ticker, info, theme):
    """Holdings du fonds"""
    
    st.markdown("### 📊 Top Holdings")
    st.info("ℹ️ Detailed holdings data requires premium API access")
    
    # Répartition sectorielle si disponible
    if 'sectorWeightings' in info and info['sectorWeightings']:
        st.markdown("#### 🏭 Sector Allocation")
        
        sectors = info['sectorWeightings']
        if sectors:
            df_sectors = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Weight'])
            
            fig = go.Figure(data=[go.Pie(
                labels=df_sectors['Sector'],
                values=df_sectors['Weight'],
                hole=0.4
            )])
            
            fig.update_layout(
                template=theme.get('plotly_template', 'plotly_dark'),
                height=400,
                 plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary'])
            )
            
            st.plotly_chart(fig, use_container_width=True)


def render_fund_performance_tab(ticker, data, theme):
    """Performance du fonds"""
    
    st.markdown("### 📉 Performance Analysis")
    
    returns = data['Close'].pct_change().dropna()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        perf_1m = ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100 if len(data) >= 21 else 0
        st.metric("1 Month", f"{perf_1m:+.2f}%")
    
    with col2:
        perf_3m = ((data['Close'].iloc[-1] / data['Close'].iloc[-63]) - 1) * 100 if len(data) >= 63 else 0
        st.metric("3 Months", f"{perf_3m:+.2f}%")
    
    with col3:
        perf_ytd = calculate_ytd_return(data)
        st.metric("YTD", f"{perf_ytd:+.2f}%")
    
    with col4:
        perf_1y = ((data['Close'].iloc[-1] / data['Close'].iloc[-252]) - 1) * 100 if len(data) >= 252 else 0
        st.metric("1 Year", f"{perf_1y:+.2f}%")


def render_fund_distributions_tab(ticker, theme):
    """Distributions du fonds"""
    
    st.markdown("### 💵 Distribution History")
    
    try:
        divs = yahoo.get_ticker_dividends(ticker)
        
        if not divs.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=divs.index,
                y=divs.values,
                marker_color='#10B981'
            ))
            
            fig.update_layout(
               
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                template=theme.get('plotly_template', 'plotly_dark'),
                height=400,
                 plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary']),
           title=dict(text="Distribution Payments", font=dict(size=18, color=theme['text_primary'])),
           
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(divs.to_frame('Distribution'), use_container_width=True)
        else:
            st.info("No distribution history available")
    
    except:
        st.info("Distribution data not available")



# Fonctions utilitaires communes
def format_large_number(num):
    """Formate les grands nombres"""
    if not num or num == 0:
        return "N/A"
    
    num = float(num)
    
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"
        
        
        
def calculate_ytd_return(data):
    """Calcule le retour YTD"""
    import datetime
    
    current_year = datetime.datetime.now().year
    ytd_data = data[data.index.year == current_year]
    
    if len(ytd_data) < 2:
        return 0
    
    start_price = float(ytd_data['Close'].iloc[0])
    end_price = float(ytd_data['Close'].iloc[-1])
    
    return ((end_price - start_price) / start_price) * 100

