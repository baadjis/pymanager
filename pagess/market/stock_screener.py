# pages/screener.py
"""
Advanced Multi-Asset Screener
Stocks, ETFs, Funds, Commodities, Currencies, Bonds
With Geography, Fundamentals, Technical, and Custom Filters
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from uiconfig import get_theme_colors, apply_plotly_theme
from dataprovider import yahoo
import plotly.graph_objects as go


# Base de données de tickers par catégorie et géographie
TICKERS_DATABASE = {
    'stocks': {
        'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'WMT', 
               'JNJ', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'DIS', 'ADBE', 'NFLX', 'CRM'],
        'Europe': ['ASML', 'LVMH.PA', 'NVO', 'SAP', 'TM', 'SHEL', 'AZN', 'OR.PA', 'MC.PA', 'SAN.PA'],
        'Asia': ['TSM', 'BABA', '2222.SR', '005930.KS', '7203.T', '6758.T', '9988.HK', '0700.HK'],
        'Global': ['NESN.SW', 'RHHBY', 'BHP', 'RIO']
    },
    'etfs': {
        'US Broad': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO'],
        'Sector': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLU', 'XLB', 'XLC'],
        'International': ['EFA', 'EEM', 'VEA', 'IEMG', 'VWO', 'ACWI'],
        'Bond': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'MUB']
    },
    'commodities': {
        'Energy': ['CL=F', 'BZ=F', 'NG=F', 'RB=F', 'HO=F'],
        'Metals': ['GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F'],
        'Agriculture': ['ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'SB=F', 'CC=F', 'CT=F']
    },
    'currencies': {
        'Major': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X', 'AUDUSD=X', 'NZDUSD=X'],
        'Crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD']
    },
    'bonds': {
        'Treasury': ['^IRX', '^FVX', '^TNX', '^TYX'],
        'ETFs': ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'MUB', 'VCIT', 'VCLT']
    }
}


def render_screener():
    """Screener principal multi-asset"""
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
            🎯 Advanced Multi-Asset Screener
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            Filter and discover assets across all markets with advanced criteria
        </p>
    </div>
    """)
    
    # Tabs par type d'actif
    tabs = st.tabs(['📈 Stocks', '🏦 ETFs & Funds', '🏭 Commodities', '💱 Currencies', '📉 Bonds'])
    
    with tabs[0]:
        render_stock_screener(theme)
    
    with tabs[1]:
        render_etf_screener(theme)
    
    with tabs[2]:
        render_commodity_screener(theme)
    
    with tabs[3]:
        render_currency_screener(theme)
    
    with tabs[4]:
        render_bond_screener(theme)


def render_stock_screener(theme):
    """Screener pour actions"""
    
    st.markdown("### 📈 Stock Screener")
    
    # Section filtres
    with st.expander("🔍 Filters", expanded=True):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌍 Geography**")
            geography = st.multiselect(
                "Select Markets",
                ['US', 'Europe', 'Asia', 'Global'],
                default=['US'],
                key='stock_geo'
            )
        
        with col2:
            st.markdown("**💼 Market Cap**")
            market_cap_range = st.select_slider(
                "Range",
                options=['Nano (<$50M)', 'Micro ($50M-$300M)', 'Small ($300M-$2B)', 
                        'Mid ($2B-$10B)', 'Large ($10B-$200B)', 'Mega (>$200B)'],
                value=('Small ($300M-$2B)', 'Mega (>$200B)'),
                key='stock_mcap'
            )
        
        with col3:
            st.markdown("**📊 Sector**")
            sectors = st.multiselect(
                "Select Sectors",
                ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 
                 'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication'],
                key='stock_sector'
            )
        
        st.divider()
        
        # Filtres fondamentaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**💰 P/E Ratio**")
            pe_min = st.number_input("Min", value=0.0, key='pe_min')
            pe_max = st.number_input("Max", value=50.0, key='pe_max')
        
        with col2:
            st.markdown("**📈 Dividend Yield**")
            div_min = st.number_input("Min %", value=0.0, key='div_min')
            div_max = st.number_input("Max %", value=10.0, key='div_max')
        
        with col3:
            st.markdown("**💵 Price Range**")
            price_min = st.number_input("Min $", value=0.0, key='price_min')
            price_max = st.number_input("Max $", value=1000.0, key='price_max')
        
        with col4:
            st.markdown("**📊 Volume**")
            volume_min = st.number_input("Min (M)", value=0.0, key='vol_min')
        
        st.divider()
        
        # Filtres techniques
        st.markdown("**⚡ Technical Filters**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_filter = st.checkbox("RSI < 30 (Oversold)", key='rsi_os')
            rsi_filter2 = st.checkbox("RSI > 70 (Overbought)", key='rsi_ob')
        
        with col2:
            ma_filter = st.checkbox("Price > MA 50", key='ma50')
            ma_filter2 = st.checkbox("Price > MA 200", key='ma200')
        
        with col3:
            perf_filter = st.checkbox("52W High", key='52h')
            perf_filter2 = st.checkbox("52W Low", key='52l')
        
        with col4:
            volume_filter = st.checkbox("High Volume (>Avg)", key='hvol')
    
    # Bouton scan
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔎 Scan Market", type="primary", use_container_width=True):
            st.session_state.stock_scan_triggered = True
    
    with col2:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()
    
    # Résultats
    if st.session_state.get('stock_scan_triggered', False):
        render_stock_results(geography, theme)


def render_etf_screener(theme):
    """Screener pour ETFs"""
    
    st.markdown("### 🏦 ETF & Fund Screener")
    
    with st.expander("🔍 Filters", expanded=True):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📂 Category**")
            etf_category = st.multiselect(
                "Select",
                ['US Broad', 'Sector', 'International', 'Bond', 'Commodity', 'Currency'],
                default=['US Broad'],
                key='etf_cat'
            )
        
        with col2:
            st.markdown("**💰 Expense Ratio**")
            expense_max = st.slider("Max %", 0.0, 1.0, 0.5, 0.05, key='expense')
        
        with col3:
            st.markdown("**📊 AUM**")
            aum_min = st.selectbox(
                "Minimum",
                ['Any', '$100M', '$500M', '$1B', '$5B', '$10B'],
                key='aum_min'
            )
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📈 Performance**")
            perf_period = st.selectbox("Period", ['1M', '3M', '6M', '1Y', 'YTD'], key='etf_perf')
            perf_min = st.number_input("Min Return %", value=-100.0, key='etf_perf_min')
        
        with col2:
            st.markdown("**📉 Volatility**")
            vol_max = st.slider("Max %", 0.0, 50.0, 30.0, key='etf_vol')
        
        with col3:
            st.markdown("**💵 Dividend Yield**")
            etf_div_min = st.number_input("Min %", value=0.0, key='etf_div')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Scan ETFs", type="primary", use_container_width=True, key='scan_etf'):
            st.session_state.etf_scan_triggered = True
    
    if st.session_state.get('etf_scan_triggered', False):
        render_etf_results(etf_category, theme)


def render_commodity_screener(theme):
    """Screener pour commodities"""
    
    st.markdown("### 🏭 Commodity Screener")
    
    with st.expander("🔍 Filters", expanded=True):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏷️ Category**")
            commodity_cat = st.multiselect(
                "Select",
                ['Energy', 'Metals', 'Agriculture', 'Livestock'],
                default=['Energy', 'Metals'],
                key='comm_cat'
            )
        
        with col2:
            st.markdown("**📊 Price Change**")
            change_period = st.selectbox("Period", ['1D', '1W', '1M', '3M', '1Y'], key='comm_period')
            change_min = st.number_input("Min %", value=-100.0, key='comm_change_min')
            change_max = st.number_input("Max %", value=100.0, key='comm_change_max')
        
        with col3:
            st.markdown("**📈 Trend**")
            trend = st.radio(
                "Direction",
                ['Any', 'Uptrend', 'Downtrend', 'Sideways'],
                key='comm_trend'
            )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Scan Commodities", type="primary", use_container_width=True, key='scan_comm'):
            st.session_state.comm_scan_triggered = True
    
    if st.session_state.get('comm_scan_triggered', False):
        render_commodity_results(commodity_cat, theme)


def render_currency_screener(theme):
    """Screener pour devises"""
    
    st.markdown("### 💱 Currency Screener")
    
    with st.expander("🔍 Filters", expanded=True):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💱 Type**")
            curr_type = st.multiselect(
                "Select",
                ['Major Pairs', 'Cross Pairs', 'Crypto'],
                default=['Major Pairs'],
                key='curr_type'
            )
        
        with col2:
            st.markdown("**📊 Volatility**")
            curr_vol_min = st.number_input("Min %", value=0.0, key='curr_vol_min')
            curr_vol_max = st.number_input("Max %", value=50.0, key='curr_vol_max')
        
        with col3:
            st.markdown("**📈 Performance**")
            curr_perf_period = st.selectbox("Period", ['1D', '1W', '1M', '1Y'], key='curr_perf')
            curr_perf_min = st.number_input("Min %", value=-50.0, key='curr_perf_min')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Scan Currencies", type="primary", use_container_width=True, key='scan_curr'):
            st.session_state.curr_scan_triggered = True
    
    if st.session_state.get('curr_scan_triggered', False):
        render_currency_results(curr_type, theme)


def render_bond_screener(theme):
    """Screener pour obligations"""
    
    st.markdown("### 📉 Bond Screener")
    
    with st.expander("🔍 Filters", expanded=True):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏦 Type**")
            bond_type = st.multiselect(
                "Select",
                ['Treasury', 'Investment Grade', 'High Yield', 'Municipal', 'International'],
                default=['Treasury'],
                key='bond_type'
            )
        
        with col2:
            st.markdown("**⏰ Maturity**")
            maturity = st.multiselect(
                "Duration",
                ['Short (1-3Y)', 'Intermediate (3-10Y)', 'Long (10-30Y)'],
                key='bond_mat'
            )
        
        with col3:
            st.markdown("**📊 Yield**")
            yield_min = st.number_input("Min %", value=0.0, key='yield_min')
            yield_max = st.number_input("Max %", value=10.0, key='yield_max')
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Price Change**")
            bond_change_min = st.number_input("Min % (1M)", value=-10.0, key='bond_ch_min')
        
        with col2:
            st.markdown("**⚡ Duration**")
            duration_max = st.slider("Max Duration", 0.0, 20.0, 10.0, key='duration')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Scan Bonds", type="primary", use_container_width=True, key='scan_bond'):
            st.session_state.bond_scan_triggered = True
    
    if st.session_state.get('bond_scan_triggered', False):
        render_bond_results(bond_type, theme)


def render_stock_results(geography, theme):
    """Affiche les résultats pour actions"""
    
    st.markdown("### 📊 Scan Results")
    
    with st.spinner("Scanning market..."):
        # Collecter les tickers selon géographie
        tickers = []
        for geo in geography:
            tickers.extend(TICKERS_DATABASE['stocks'].get(geo, []))
        
        # Scanner les tickers
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(tickers[:20]):  # Limiter à 20 pour démo
            try:
                data = yahoo.get_ticker_data(ticker, period='1mo')
                info = yahoo.get_ticker_info(ticker)
                
                if data is not None and not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    change_30d = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
                    
                    results.append({
                        'Symbol': ticker,
                        'Name': info.get('longName', ticker)[:30],
                        'Price': current_price,
                        'Change 30D': change_30d,
                        'Market Cap': info.get('marketCap', 0),
                        'P/E': info.get('trailingPE', 0),
                        'Div Yield': (info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                        'Sector': info.get('sector', 'N/A')
                    })
            except:
                continue
            
            progress_bar.progress((idx + 1) / len(tickers[:20]))
        
        progress_bar.empty()
    
    if results:
        df = pd.DataFrame(results)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Results Found", len(df))
        with col2:
            avg_pe = df[df['P/E'] > 0]['P/E'].mean()
            st.metric("Avg P/E", f"{avg_pe:.1f}" if not pd.isna(avg_pe) else "N/A")
        with col3:
            avg_div = df['Div Yield'].mean()
            st.metric("Avg Div Yield", f"{avg_div:.2f}%")
        with col4:
            avg_change = df['Change 30D'].mean()
            st.metric("Avg 30D Change", f"{avg_change:+.2f}%")
        
        st.divider()
        
        # Tableau résultats
        st.dataframe(
            df.style.format({
                'Price': '${:.2f}',
                'Change 30D': '{:+.2f}%',
                'Market Cap': '${:,.0f}',
                'P/E': '{:.2f}',
                'Div Yield': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export Results (CSV)",
            csv,
            f"stock_screen_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No results found. Try adjusting your filters.")


def render_etf_results(categories, theme):
    """Affiche les résultats pour ETFs"""
    
    st.markdown("### 📊 ETF Results")
    
    with st.spinner("Scanning ETFs..."):
        tickers = []
        for cat in categories:
            tickers.extend(TICKERS_DATABASE['etfs'].get(cat, []))
        
        results = []
        for ticker in tickers[:15]:
            try:
                data = yahoo.get_ticker_data(ticker, period='1y')
                info = yahoo.get_ticker_info(ticker)
                
                if data is not None and not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    ytd_return = ((current_price - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
                    volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
                    
                    results.append({
                        'Symbol': ticker,
                        'Name': info.get('longName', ticker)[:40],
                        'Price': current_price,
                        'YTD Return': ytd_return,
                        'Volatility': volatility,
                        'Expense Ratio': (info.get('annualReportExpenseRatio', 0) * 100) if info.get('annualReportExpenseRatio') else 0,
                        'AUM': info.get('totalAssets', 0)
                    })
            except:
                continue
    
    if results:
        df = pd.DataFrame(results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 ETFs Found", len(df))
        with col2:
            avg_return = df['YTD Return'].mean()
            st.metric("Avg YTD Return", f"{avg_return:+.2f}%")
        with col3:
            avg_expense = df['Expense Ratio'].mean()
            st.metric("Avg Expense", f"{avg_expense:.2f}%")
        
        st.divider()
        
        st.dataframe(
            df.style.format({
                'Price': '${:.2f}',
                'YTD Return': '{:+.2f}%',
                'Volatility': '{:.2f}%',
                'Expense Ratio': '{:.2f}%',
                'AUM': '${:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No ETFs found.")


def render_commodity_results(categories, theme):
    """Affiche les résultats pour commodities"""
    
    st.markdown("### 📊 Commodity Results")
    
    with st.spinner("Scanning commodities..."):
        tickers = []
        for cat in categories:
            tickers.extend(TICKERS_DATABASE['commodities'].get(cat, []))
        
        results = []
        for ticker in tickers:
            try:
                data = yahoo.get_ticker_data(ticker, period='3mo')
                
                if data is not None and not data.empty and len(data) > 1:
                    current_price = float(data['Close'].iloc[-1])
                    change_1m = ((current_price - float(data['Close'].iloc[-21])) / float(data['Close'].iloc[-21])) * 100 if len(data) >= 21 else 0
                    volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
                    
                    results.append({
                        'Symbol': ticker,
                        'Price': current_price,
                        '1M Change': change_1m,
                        '3M High': float(data['High'].max()),
                        '3M Low': float(data['Low'].min()),
                        'Volatility': volatility
                    })
            except:
                continue
    
    if results:
        df = pd.DataFrame(results)
        
        st.dataframe(
            df.style.format({
                'Price': '${:.2f}',
                '1M Change': '{:+.2f}%',
                '3M High': '${:.2f}',
                '3M Low': '${:.2f}',
                'Volatility': '{:.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("No commodities found.")


def render_currency_results(types, theme):
    """Affiche les résultats pour devises"""
    
    st.markdown("### 📊 Currency Results")
    
    with st.spinner("Scanning currencies..."):
        tickers = []
        if 'Major Pairs' in types:
            tickers.extend(TICKERS_DATABASE['currencies']['Major'])
        if 'Crypto' in types:
            tickers.extend(TICKERS_DATABASE['currencies']['Crypto'])
        
        results = []
        for ticker in tickers:
            try:
                data = yahoo.get_ticker_data(ticker, period='1mo')
                
                if data is not None and not data.empty:
                    current_rate = float(data['Close'].iloc[-1])
                    change_1w = ((current_rate - float(data['Close'].iloc[-7])) / float(data['Close'].iloc[-7])) * 100 if len(data) >= 7 else 0
                    volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
                    
                    results.append({
                        'Pair': ticker,
                        'Rate': current_rate,
                        '1W Change': change_1w,
                        'Volatility': volatility,
                        'Type': 'Crypto' if 'USD' in ticker and '-' in ticker else 'Forex'
                    })
            except:
                continue
    
    if results:
        df = pd.DataFrame(results)
        
        st.dataframe(
            df.style.format({
                'Rate': '{:.4f}',
                '1W Change': '{:+.2f}%',
                'Volatility': '{:.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("No currencies found.")


def render_bond_results(types, theme):
    """Affiche les résultats pour bonds"""
    
    st.markdown("### 📊 Bond Results")
    
    with st.spinner("Scanning bonds..."):
        tickers = []
        if 'Treasury' in types:
            tickers.extend(TICKERS_DATABASE['bonds']['Treasury'])
        tickers.extend(TICKERS_DATABASE['bonds']['ETFs'][:10])
        
        results = []
        for ticker in tickers:
            try:
                data = yahoo.get_ticker_data(ticker, period='1mo')
                
                if data is not None and not data.empty:
                    current_value = float(data['Close'].iloc[-1])
                    change_1m = ((current_value - float(data['Close'].iloc[0])) / float(data['Close'].iloc[0])) * 100
                    
                    results.append({
                        'Symbol': ticker,
                        'Value/Yield': current_value,
                        '1M Change': change_1m,
                        'Type': 'Yield' if ticker.startswith('^') else 'ETF'
                    })
            except:
                continue
    
    if results:
        df = pd.DataFrame(results)
        
        st.dataframe(
            df.style.format({
                'Value/Yield': '{:.3f}',
                '1M Change': '{:+.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("No bonds found.")
