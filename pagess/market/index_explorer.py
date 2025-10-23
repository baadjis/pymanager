# pages/index_explorer.py
"""
Index Explorer - Analyse détaillée d'indices boursiers
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataprovider import yahoo
from uiconfig import get_theme_colors


# Définition des indices disponibles avec leurs composants
MAJOR_INDICES = {
    "🇺🇸 US Indices": {
        '^GSPC': {'name': 'S&P 500', 'desc': 'Top 500 US companies', 'components': True},
        '^DJI': {'name': 'Dow Jones', 'desc': '30 blue-chip companies', 'components': True},
        '^IXIC': {'name': 'NASDAQ Composite', 'desc': 'Tech-heavy index', 'components': False},
        '^RUT': {'name': 'Russell 2000', 'desc': 'Small-cap index', 'components': False},
    },
    "🌍 International": {
        '^FTSE': {'name': 'FTSE 100', 'desc': 'UK top 100', 'components': False},
        '^GDAXI': {'name': 'DAX', 'desc': 'German index', 'components': False},
        '^N225': {'name': 'Nikkei 225', 'desc': 'Japanese index', 'components': False},
        '^FCHI': {'name': 'CAC 40', 'desc': 'French index', 'components': False},
    },
    "📊 Sector Indices": {
        'XLK': {'name': 'Technology', 'desc': 'Tech sector ETF', 'components': False},
        'XLF': {'name': 'Financial', 'desc': 'Financial sector ETF', 'components': False},
        'XLE': {'name': 'Energy', 'desc': 'Energy sector ETF', 'components': False},
        'XLV': {'name': 'Healthcare', 'desc': 'Healthcare sector ETF', 'components': False},
    }
}


def render_index_explorer():
    """Explorer pour les indices"""
    theme = get_theme_colors()
    
    st.markdown("### 📊 Index Explorer")
    st.caption("Analyze major market indices and their components")
    
    # Sélection de l'indice
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Créer une liste plate de tous les indices
        all_indices = {}
        for region, indices in MAJOR_INDICES.items():
            for ticker, info in indices.items():
                name_i=info['name']
                key=f"""{name_i} ({ticker})"""
                print(key)
                all_indices[key] = ticker
        
        selected_display = st.selectbox(
            "Select Index",
            options=list(all_indices.keys()),
            key="index_selector"
        )
        
        ticker = all_indices[selected_display]
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("📊 Analyze", use_container_width=True, type="primary")
    
    if ticker and (analyze_button or ticker):
        try:
            with st.spinner(f"Loading {ticker}..."):
                # Charger les données
                data = yahoo.get_ticker_data(ticker, period='1y').dropna()
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return
                
                # Header avec métriques
                render_index_header(ticker, data, theme)
                
                st.divider()
                
                # Tabs
                tabs = st.tabs([
                    "📈 Chart",
                    "ℹ️ Overview",
                    "🏢 Components",
                    "📊 Sector Breakdown",
                    "📉 Performance"
                ])
                
                with tabs[0]:
                    render_index_chart_tab(ticker, theme)
                
                with tabs[1]:
                    render_index_overview_tab(ticker, data, theme)
                   
                
                with tabs[2]:
                    render_index_components_tab(ticker, theme)
                    
                
                with tabs[3]:
                    render_index_sectors_tab(ticker, theme)
                   
                
                with tabs[4]:
                    render_index_performance_tab(ticker, data, theme)
                    
                    
        
        except Exception as e:
            st.error(f"❌ Error loading {ticker}: {str(e)}")


def render_index_header(ticker, data, theme):
    """Header avec informations principales de l'indice"""
    
    # Récupérer les infos de l'indice
    index_info = None
    index_name = ticker
    
    for region, indices in MAJOR_INDICES.items():
        if ticker in indices:
            index_info = indices[ticker]
            index_name = index_info['name']
            break
    
    # Calculer les métriques
    current_price = float(data['Close'].iloc[-1])
    change = 0
    change_pct = 0
    if len(data) > 1:
        prev_price = float(data['Close'].iloc[-2])
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
    

    
    # Calculer les performances sur différentes périodes
    perf_1w = calculate_performance(data, 5) if len(data) >= 5 else 0
    perf_1m = calculate_performance(data, 21) if len(data) >= 21 else 0
    perf_ytd = calculate_ytd_performance(data)
    perf_1y = calculate_performance(data, 252) if len(data) >= 252 else 0
    
    # Layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"### {index_name}")
        st.caption(ticker)
        st.metric(
            label="Current Value",
            value=f"{current_price:,.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="1 Week",
            value=f"{perf_1w:+.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="1 Month",
            value=f"{perf_1m:+.2f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="YTD",
            value=f"{perf_ytd:+.2f}%",
            delta=None
        )
    
    with col5:
        st.metric(
            label="1 Year",
            value=f"{perf_1y:+.2f}%",
            delta=None
        )


def render_index_chart_tab(ticker, theme):
    """Onglet graphique de l'indice"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            key="index_chart_period"
        )
        
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_volume = st.checkbox("Show Volume", value=False)
    
    # Recharger pour la période sélectionnée
    data2 = yahoo.get_ticker_data(ticker, period=period)
    data2=data2.dropna()
    
    data_y=[t[0] for t in data2['Close'].values]
    data_volume=[t[0] for t in data2['Volume'].values]
    data_x=data2.index
    if data2 is not None and not data2.empty:
        # Créer le graphique
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Prix
            fig.add_trace(
                go.Scatter(
                    x=data_x,
                    y=data_y,
                    mode='lines',
                    name=ticker,
                    line=dict(color='#6366F1', width=2)
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=data_x,
                    y=data_volume,
                    name='Volume',
                    marker_color='rgba(99, 102, 241, 0.3)'
                ),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data_x,
                y=data_y,
                mode='lines',
                name=ticker,
                line=dict(color='#6366F1', width=2),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.1)'
            ))
        
        # Ajouter moyennes mobiles
        if show_ma:
            if len(data2) >= 50:
                ma50 =[t[0] for t in data2['Close'].rolling(window=50).mean().values]
                fig.add_trace(go.Scatter(
                    x=data_x,
                    y=ma50,
                    mode='lines',
                    name='MA 50',
                    line=dict(color='#F59E0B', width=1.5, dash='dash')
                ), row=1, col=1) if show_volume else fig.add_trace(go.Scatter(
                    x=data_x,
                    y=ma50,
                    mode='lines',
                    name='MA 50',
                    line=dict(color='#F59E0B', width=1.5, dash='dash')
                ))
            
            if len(data2) >= 200:
                ma200 = [t[0] for t in data2['Close'].rolling(window=200).mean().values]
                fig.add_trace(go.Scatter(
                    x=data_x,
                    y=ma200,
                    mode='lines',
                    name='MA 200',
                    line=dict(color='#EF4444', width=1.5, dash='dash')
                ), row=1, col=1) if show_volume else fig.add_trace(go.Scatter(
                    x=data_x,
                    y=ma200,
                    mode='lines',
                    name='MA 200',
                    line=dict(color='#EF4444', width=1.5, dash='dash')
                ))
        
        fig.update_layout(
            
            xaxis_title="Date",
            yaxis_title="Price",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=600,
            hovermode='x unified',
            paper_bgcolor=theme['bg_card'],
            plot_bgcolor=theme['bg_card'],
            font=dict(color=theme['text_primary']),
            title=dict(text=f"{ticker} Price Chart", font=dict(size=18, color=theme['text_primary'])),
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_index_overview_tab(ticker, data, theme):
    """Onglet overview de l'indice"""
    
    st.markdown("### 📋 Index Overview")
    
    # Trouver les infos de l'indice
    index_info = None
    for region, indices in MAJOR_INDICES.items():
        if ticker in indices:
            index_info = indices[ticker]
            
            break
    
    if index_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ℹ️ Basic Information")
            st.markdown(f"**Name:** {index_info['name']}")
            st.markdown(f"**Ticker:** {ticker}")
            st.markdown(f"**Description:** {index_info['desc']}")
            st.markdown(f"**Has Components:** {'Yes' if index_info['components'] else 'No'}")
        
        with col2:
            st.markdown("#### 📊 Statistics")
            
            # Calculer stats
            print("tail",data['High'].tail(252).max())
            high_52w = data['High'].tail(252).max() if len(data) >= 252 else data['High'].max().values[0]
            low_52w = data['Low'].tail(252).min() if len(data) >= 252 else data['Low'].min().values[0]
            avg_volume = data['Volume'].mean().values[0]
            
            st.markdown(f"**52W High:** {high_52w:,.2f}")
            st.markdown(f"**52W Low:** {low_52w:,.2f}")
            st.markdown(f"**Avg Volume:** {avg_volume:,.0f}")
            st.markdown(f"**Data Points:** {len(data)}")
            
            
    
    st.divider()
    
    # Tableau de performance
    st.markdown("### 📈 Historical Performance")
    
    perf_data = {
        'Period': ['1 Day', '1 Week', '1 Month', '3 Months', '6 Months', 'YTD', '1 Year'],
        'Return (%)': [
            calculate_performance(data, 1) if len(data) >= 1 else 0.0,
            calculate_performance(data, 5) if len(data) >= 5 else 0.0,
            calculate_performance(data, 21) if len(data) >= 21 else 0.0,
            calculate_performance(data, 63) if len(data) >= 63 else 0.0,
            calculate_performance(data, 126) if len(data) >= 126 else 0.0,
            calculate_ytd_performance(data),
            calculate_performance(data, 252) if len(data) >= 252 else 0.0
        ]
    }
    
    df_perf = pd.DataFrame(perf_data)
    
    # Colorer le tableau
    def color_performance(val):
        color = '#10B981' if val > 0 else '#EF4444' if val < 0 else '#94A3B8'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(
        df_perf.style.applymap(color_performance, subset=['Return (%)']),
        use_container_width=True,
        hide_index=True
    )


def render_index_components_tab(ticker, theme):
    """Onglet composants de l'indice"""
    
    st.markdown("### 🏢 Index Components")
    
    # Vérifier si l'indice a des composants
    has_components = False
    for region, indices in MAJOR_INDICES.items():
        if ticker in indices:
            has_components = indices[ticker]['components']
            break
    
    if not has_components:
        st.info(f"ℹ️ Component data not available for {ticker}")
        return
    
    # Pour les indices majeurs, afficher des composants simulés
    if ticker == '^GSPC':
        st.markdown("#### Top 10 S&P 500 Components by Weight")
        
        components = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ'],
            'Name': ['Apple Inc.', 'Microsoft Corp.', 'Amazon.com Inc.', 'NVIDIA Corp.', 
                     'Alphabet Inc.', 'Meta Platforms', 'Tesla Inc.', 'Berkshire Hathaway', 
                     'UnitedHealth Group', 'Johnson & Johnson'],
            'Weight (%)': [7.1, 6.8, 3.2, 2.9, 2.1, 1.8, 1.7, 1.6, 1.3, 1.2],
            'Sector': ['Technology', 'Technology', 'Consumer Discretionary', 'Technology',
                      'Communication', 'Communication', 'Consumer Discretionary', 'Financials',
                      'Healthcare', 'Healthcare']
        })
        
    elif ticker == '^DJI':
        st.markdown("#### Dow Jones 30 Components")
        
        components = pd.DataFrame({
            'Symbol': ['UNH', 'GS', 'MSFT', 'HD', 'CAT', 'AMGN', 'MCD', 'V', 'BA', 'HON'],
            'Name': ['UnitedHealth Group', 'Goldman Sachs', 'Microsoft', 'Home Depot',
                     'Caterpillar', 'Amgen', 'McDonalds', 'Visa', 'Boeing', 'Honeywell'],
            'Weight (%)': [8.5, 7.2, 6.8, 5.9, 5.1, 4.8, 4.5, 4.2, 4.0, 3.8],
            'Sector': ['Healthcare', 'Financials', 'Technology', 'Consumer Discretionary',
                      'Industrials', 'Healthcare', 'Consumer Discretionary', 'Financials',
                      'Industrials', 'Industrials']
        })
    else:
        st.info("Component data not available")
        return
    
    # Afficher le tableau
    st.dataframe(components, use_container_width=True, hide_index=True)
    
    # Graphique des poids
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=components['Symbol'],
        y=components['Weight (%)'],
        marker_color='#6366F1',
        text=components['Weight (%)'],
        texttemplate='%{text:.1f}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top Components by Weight",
        xaxis_title="Symbol",
        yaxis_title="Weight (%)",
        template=theme.get('plotly_template', 'plotly_dark'),
        height=400,
        plot_bgcolor= theme['bg_card'],
        paper_bgcolor= theme['bg_card'],
        font=dict(color=theme['text_primary'])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_index_sectors_tab(ticker, theme):
    """Onglet répartition sectorielle"""
    
    st.markdown("### 🏭 Sector Breakdown")
    
    # Données simulées de répartition sectorielle
    if ticker == '^GSPC':
        sectors = pd.DataFrame({
            'Sector': ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
                      'Communication', 'Industrials', 'Consumer Staples', 'Energy',
                      'Utilities', 'Real Estate', 'Materials'],
            'Weight (%)': [28.5, 13.2, 12.8, 10.5, 8.7, 8.2, 6.5, 4.2, 2.8, 2.4, 2.2]
        })
    elif ticker == '^DJI':
        sectors = pd.DataFrame({
            'Sector': ['Financials', 'Healthcare', 'Technology', 'Industrials',
                      'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Communication'],
            'Weight (%)': [18.5, 16.2, 15.8, 14.5, 12.3, 10.5, 7.2, 5.0]
        })
    else:
        st.info("Sector breakdown not available for this index")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=sectors['Sector'],
            values=sectors['Weight (%)'],
            hole=0.4,
            marker=dict(colors=[
                '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B',
                '#10B981', '#3B82F6', '#14B8A6', '#F97316',
                '#EF4444', '#8B5CF6', '#06B6D4'
            ])
        )])
        
        fig_pie.update_layout(
            title="Sector Allocation",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
           font=dict(color=theme['text_primary'])
        )
        
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=sectors['Weight (%)'],
            y=sectors['Sector'],
            orientation='h',
            marker_color='#6366F1',
            text=sectors['Weight (%)'],
            texttemplate='%{text:.1f}%',
            textposition='outside'
        ))
        
        fig_bar.update_layout(
            title="Sector Weights",
            xaxis_title="Weight (%)",
            yaxis_title="Sector",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
           
           font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tableau détaillé
    st.dataframe(sectors, use_container_width=True, hide_index=True)


def render_index_performance_tab(ticker, data, theme):
    """Onglet analyse de performance"""
    
    st.markdown("### 📉 Performance Analysis")
    
    # Calculer les rendements
    returns = data['Close'].pct_change().dropna()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_return = returns.mean().values[0] * 252  # Annualisé
        st.metric(
            label="Avg Annual Return",
            value=f"{avg_return * 100:.2f}%"
        )
    
    with col2:
        volatility = returns.std().values[0] * (252 ** 0.5)  # Annualisée
        st.metric(
            label="Volatility (Annual)",
            value=f"{volatility * 100:.2f}%"
        )
    
    with col3:
        sharpe = avg_return / volatility if volatility > 0 else 0
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}"
        )
    
    with col4:
        max_dd = calculate_max_drawdown(data['Close']).values[0]
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%"
        )
    
    st.divider()
    
    # Distribution des rendements
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Returns Distribution")
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=[t[0]* 100 for t in returns.values ],
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
            
           font=dict(color=theme['text_primary']),
            title=dict(text="Returns Distribution",font=dict(size=18, color=theme['text_primary'])),
            
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Cumulative Returns")
        
        cumulative_returns = (1 + returns).cumprod()
        
        fig_cum = go.Figure()
        
        fig_cum.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=[t[0] for t in cumulative_returns.values],
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#10B981', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig_cum.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template=theme.get('plotly_template', 'plotly_dark'),
            height=400,
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            
           font=dict(color=theme['text_primary'])
        )
        
        st.plotly_chart(fig_cum, use_container_width=True)


# Fonctions utilitaires
def calculate_performance(data, periods):
    """Calcule la performance sur N périodes"""
    
    if len(data) < periods:
        return 0
   
    start_price = float(data['Close'].iloc[-periods])
   
    end_price = float(data['Close'].iloc[-1])
    
    return ((end_price - start_price) / start_price) * 100


def calculate_ytd_performance(data):
    """Calcule la performance depuis le début de l'année"""
    import datetime
    data= data.dropna()
    current_year = datetime.datetime.now().year
    ytd_data = data[data.index.year == current_year]
    
    if len(ytd_data) < 2:
        return 0
    
    start_price = float(ytd_data['Close'].iloc[0].iloc[-1])
    end_price = float(ytd_data['Close'].iloc[-1].iloc[-1])
    
    return ((end_price - start_price) / start_price) * 100


def calculate_max_drawdown(prices):
    """Calcule le drawdown maximum"""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax * 100
    
    return drawdown.min()
