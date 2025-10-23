# pages/currency_explorer.py
"""
Currency Explorer - Analyse de paires de devises
"""


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataprovider import yahoo
from uiconfig import get_theme_colors
import numpy as np

MAJOR_PAIRS = {
    "💵 USD Pairs": {
        'EURUSD=X': 'EUR/USD - Euro vs US Dollar',
        'GBPUSD=X': 'GBP/USD - British Pound vs US Dollar',
        'USDJPY=X': 'USD/JPY - US Dollar vs Japanese Yen',
        'USDCHF=X': 'USD/CHF - US Dollar vs Swiss Franc',
        'USDCAD=X': 'USD/CAD - US Dollar vs Canadian Dollar',
    },
    "🌍 Cross Pairs": {
        'EURGBP=X': 'EUR/GBP - Euro vs British Pound',
        'EURJPY=X': 'EUR/JPY - Euro vs Japanese Yen',
        'GBPJPY=X': 'GBP/JPY - British Pound vs Japanese Yen',
        'AUDJPY=X': 'AUD/JPY - Australian Dollar vs Japanese Yen',
    },
    "🪙 Cryptocurrencies": {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'BNB-USD': 'Binance Coin',
        'XRP-USD': 'Ripple',
    }
}
def get_values(dt):
        return [t[0] for t in dt.values]

def render_currency_explorer():
    """Explorer pour les devises"""
    theme = get_theme_colors()
    
    st.markdown("### 💱 Currency Explorer")
    st.caption("Analyze forex pairs and cryptocurrencies")
    
    # Recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "🔍 Search Currency Pair",
            placeholder="Enter pair (e.g., EURUSD=X, BTC-USD)",
            key="currency_search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Analyze", use_container_width=True, type="primary")
    
    # Suggestions populaires
    if not ticker:
        st.markdown("#### 💡 Popular Currency Pairs")
        
        for category, pairs in MAJOR_PAIRS.items():
            with st.expander(category):
                for symbol, name in pairs.items():
                    if st.button(f"{symbol} - {name}", key=f"pair_{symbol}"):
                        st.session_state.currency_search_value = symbol
                        st.rerun()
        return
    
    if ticker and (search_button or ticker):
        ticker = ticker.upper().strip()
        
        try:
            with st.spinner(f"Loading {ticker}..."):
                # Charger données
                data = yahoo.get_ticker_data(ticker, period='1y')
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return
                
                # Header
                render_currency_header(ticker, data, theme)
                
                st.divider()
                
                # Tabs
                tabs = st.tabs([
                    "📈 Chart",
                    "📊 Analysis",
                    "📉 Volatility",
                    "🔄 Correlations"
                ])
                
                with tabs[0]:
                    render_currency_chart_tab(ticker, data, theme)
                
                with tabs[1]:
                    render_currency_analysis_tab(ticker, data, theme)
                    
                
                with tabs[2]:
                    render_currency_volatility_tab(ticker, data, theme)
                    
                
                with tabs[3]:
                    render_currency_correlations_tab(ticker, data, theme)
                    
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def render_currency_header(ticker, data, theme):
    """Header de la paire de devises"""
    
    current_rate = float(data['Close'].iloc[-1])
    
    if len(data) > 1:
        prev_rate = float(data['Close'].iloc[-2])
        change = current_rate - prev_rate
        change_pct = (change / prev_rate) * 100
    else:
        change = 0
        change_pct = 0
    
    # Calculer performances
    perf_1w = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0
    perf_1m = ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100 if len(data) >= 21 else 0
    perf_ytd = calculate_ytd_return(data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"### {ticker}")
        
        # Déterminer le nombre de décimales selon le type
        if 'BTC' in ticker or 'ETH' in ticker:
            decimals = 2
        else:
            decimals = 4 if current_rate < 100 else 2
        
        st.metric(
            label="Current Rate",
            value=f"{current_rate:.{decimals}f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="1 Week",
            value=f"{perf_1w[0]:+.2f}%"
        )
    
    with col3:
        st.metric(
            label="1 Month",
            value=f"{perf_1m[0]:+.2f}%"
        )
    
    with col4:
        st.metric(
            label="YTD",
            value=f"{perf_ytd:+.2f}%"
        )


def render_currency_chart_tab(ticker, data, theme):
    """Graphique de la paire"""
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"],
            index=5,
            key="currency_period"
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Candlestick", "Area"],
            key="currency_chart_type"
        )
    
    with col3:
        show_ma = st.checkbox("Moving Averages", value=True)
    
    # Recharger
    data = yahoo.get_ticker_data(ticker, period=period)
    data_x=data.index
    data_y=get_values(data['Close'])
    if data is not None and not data.empty:
        fig = go.Figure()
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=data_x,
                open=get_values(data['Open']),
                high=get_values(data['High']),
                low=get_values(data['Low']),
                close=data_y,
                name=ticker
            ))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(
                x=data_x,
                y=data_y,
                mode='lines',
                name=ticker,
                line=dict(color='#6366F1', width=2),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data_x,
                y=data_y,
                mode='lines',
                name=ticker,
                line=dict(color='#6366F1', width=2)
            ))
        
        # Moyennes mobiles
        if show_ma:
            if len(data) >= 20:
                ma20 = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=data_x,
                    y=get_values(ma20),
                    mode='lines',
                    name='MA 20',
                    line=dict(color='#F59E0B', width=1, dash='dash')
                ))
            
            if len(data) >= 50:
                ma50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=get_values(ma50),
                    mode='lines',
                    name='MA 50',
                    line=dict(color='#EF4444', width=1, dash='dash')
                ))
        
        fig.update_layout(
            title=f"{ticker} Exchange Rate",
            xaxis_title="Date",
            yaxis_title="Rate",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=500,
            hovermode='x unified',
             plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_currency_analysis_tab(ticker, data, theme):
    """Analyse de la paire"""
    
    st.markdown("### 📊 Statistical Analysis")
    
    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_52w = data['High'].tail(252) if len(data) >= 252 else data['High']
        print(max(get_values(high_52w)))
        st.metric("52W High", f"{max(get_values(high_52w)):.4f}")
    
    with col2:
        low_52w = data['Low'].tail(252) if len(data) >= 252 else data['Low']
        st.metric("52W Low", f"{min(get_values(low_52w)):.4f}")
    
    with col3:
        vals = data['Volume']
        
        avg_volume = vals.mean().values[0]
        print("vols",avg_volume)
        st.metric("Avg Volume", format_large_number(avg_volume))
    
    with col4:
        volatility = data['Close'].pct_change().std().values[0] * (252 ** 0.5) * 100
        
        
        st.metric("Volatility", f"{volatility:.2f}%")
    
    st.divider()
    
    # Tableau de performance
    st.markdown("#### 📈 Performance Summary")
    
    perf_data = {
        'Period': ['1 Day', '1 Week', '1 Month', '3 Months', '6 Months', 'YTD', '1 Year'],
        'Return (%)': [
            ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1).values[0] * 100 if len(data) >= 2 else 0,
            ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1).values[0] * 100 if len(data) >= 5 else 0,
            ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1).values[0] * 100 if len(data) >= 21 else 0,
            ((data['Close'].iloc[-1] / data['Close'].iloc[-63]) - 1).values[0] * 100 if len(data) >= 63 else 0,
            ((data['Close'].iloc[-1] / data['Close'].iloc[-126]) - 1).values[0] * 100 if len(data) >= 126 else 0,
            calculate_ytd_return(data),
            ((data['Close'].iloc[-1] / data['Close'].iloc[-252]) - 1).values[0] * 100 if len(data) >= 252 else 0
        ]
    }
    
    df_perf = pd.DataFrame(perf_data)
    
    st.dataframe(df_perf, use_container_width=True, hide_index=True)


def render_currency_volatility_tab(ticker, data, theme):
    """Analyse de volatilité"""
    
    st.markdown("### 📉 Volatility Analysis")
    
    # Calculer la volatilité roulante
    returns = data['Close'].pct_change().dropna()
    rolling_20=returns.rolling(window=20).std()* (252 ** 0.5) * 100
    rolling_vol = rolling_20.values 
    data_x=rolling_20.index
    data_y=[t[0] for t in rolling_vol]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Rolling Volatility (20-day)")
        
        fig_vol = go.Figure()
        
        fig_vol.add_trace(go.Scatter(
            x=data_x,
            y=data_y,
            mode='lines',
            name='Volatility',
            line=dict(color='#EF4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        fig_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
             plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Returns Distribution")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=[t[0]*100 for t in  returns.values],
            nbinsx=50,
            marker_color='#6366F1',
            name='Returns'
        ))
        
        fig_hist.update_layout(
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
             plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Statistiques de volatilité
    st.markdown("#### 📊 Volatility Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_vol = rolling_20.iloc[-1].values[0]
        st.metric("Current Vol", f"{current_vol:.2f}%")
    
    with col2:
        avg_vol = rolling_20.mean().values[0]
        st.metric("Average Vol", f"{avg_vol:.2f}%")
    
    with col3:
        min_vol = rolling_20.min().values[0]
        st.metric("Min Vol", f"{min_vol:.2f}%")
    
    with col4:
        max_vol = rolling_20.max().values[0]
        st.metric("Max Vol", f"{max_vol:.2f}%")


def render_currency_correlations_tab(ticker, data, theme):
    """Corrélations avec autres paires"""
    
    st.markdown("### 🔄 Correlation Analysis")
    
    st.info("ℹ️ Correlation analysis with other currency pairs")
    
    # Charger quelques paires majeures pour comparaison
    major_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X']
    
    if ticker in major_pairs:
        major_pairs.remove(ticker)
    
    correlation_data = {}
    
    for pair in major_pairs[:3]:  # Limiter à 3 paires
        try:
            pair_data = yahoo.get_ticker_data(pair, period='1y')
            if pair_data is not None and not pair_data.empty:
                correlation_data[pair] = [t[0] for t in pair_data['Close'].values]
        except:
            continue
    
    if correlation_data:
        # Créer dataframe de comparaison
        df_comparison = pd.DataFrame(correlation_data)
        df_comparison[ticker] = [t[0] for t in data['Close'].values]
        
        # Calculer corrélations
        correlations = df_comparison.corr()[ticker].drop(ticker)
        
        # Afficher
        st.markdown("#### Correlation with Major Pairs")
        
        fig_corr = go.Figure()
        
        fig_corr.add_trace(go.Bar(
            x=correlations.index,
            y=correlations.values,
            marker_color=['#10B981' if x > 0 else '#EF4444' for x in correlations.values],
            text=[f"{x:.2f}" for x in correlations.values],
            textposition='outside'
        ))
        
        fig_corr.update_layout(
           
            xaxis_title="Currency Pair",
            yaxis_title="Correlation",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
             plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            title=dict(text="Correlation Coefficients", font=dict(size=18, color=theme['text_primary'])),
           font=dict(color=theme['text_primary']),
            yaxis_range=[-1, 1]
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlation data not available")


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

