# pages/market_overview.py
"""
Market Overview - Modern UI style Perplexity Finance
Vue complète et élégante du marché avec indices principaux
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataprovider import yahoo
from uiconfig import get_theme_colors


def render_market_overview():
    """Market Overview avec design moderne"""
    theme = get_theme_colors()
    
    # Header moderne
    st.html("""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    ">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📊 Market Overview
        </h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.8;">
            Real-time market data and indices • Updated every 15 minutes
        </p>
    </div>
    """)
    
    # Section 1: Indices principaux
    render_main_indices(theme)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 2: Graphiques des indices
    render_indices_charts(theme)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 3: Market Stats
    col1, col2 = st.columns(2)
    
    with col1:
        render_sector_heatmap(theme)
    
    with col2:
        render_market_breadth(theme)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 4: Top Movers & Market News
    render_top_movers(theme)


def render_main_indices(theme):
    """Affiche les indices principaux avec design moderne"""
    
    st.markdown("""
    <div style="
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: #F8FAFC;
    ">
        🌍 Global Indices
    </div>
    """, unsafe_allow_html=True)
    
    # Définir les indices par région
    indices_data = {
        '🇺🇸 US Markets': {
            '^GSPC': {'name': 'S&P 500', 'desc': 'Large Cap'},
            '^DJI': {'name': 'Dow Jones', 'desc': 'Blue Chips'},
            '^IXIC': {'name': 'NASDAQ', 'desc': 'Tech Heavy'},
            '^RUT': {'name': 'Russell 2000', 'desc': 'Small Cap'}
        },
        
        'EUROPEAN MARKETS' :{
        
        '^FTSE': {'name': 'FTSE 100', 'desc': 'UK'},
        '^GDAXI': {'name': 'DAX', 'desc': 'Germany'},
        '^FCHI' : {'name': 'CAC 40','desc':'French'},
        '^STOXX50E':{'name':'EURO STOXX 50','desc':'Europe'}
        
        },
        '🌍 International': {
            
            '^N225': {'name': 'Nikkei 225', 'desc': 'Japan'},
            '000001.SS': {'name': 'Shanghai', 'desc': 'China'},
            '^BVSP' :{'name' :'IBOVESPA','desc':'Brazil'},
            '^990100-USD-STRD' :{'name' : 'MSCI WORLD', 'desc': 'World'}
        },
        '📊 Other Indices': {
            '^VIX': {'name': 'VIX', 'desc': 'Volatility Index'},
            'GC=F': {'name': 'Gold', 'desc': 'Commodities'},
            'CL=F': {'name': 'Crude Oil', 'desc': 'Energy'},
            '^TNX': {'name': 'US 10Y', 'desc': 'Treasury Yield'}
        }
    }
    
    for region, indices in indices_data.items():
        st.markdown(f"""
        <div style="
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1.5rem 0 0.8rem 0;
            opacity: 0.9;
        ">
            {region}
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(indices))
        
        for idx, (col, (ticker, info)) in enumerate(zip(cols, indices.items())):
            with col:
                render_index_card(ticker, info['name'], info['desc'], theme)


def render_index_card(ticker, name, description, theme):
    """Carte d'index individuelle avec design moderne"""
    try:
        # Récupérer les données via yahoo avec la bonne fonction
        data = yahoo.get_ticker_data(ticker, period='5d')
        
        # Vérifier si data est valide
        if data is not None and isinstance(data, pd.DataFrame) and not data.empty and len(data) > 1:
            # S'assurer que Close existe
            if 'Close' not in data.columns:
                render_index_card_fallback(ticker, name, description)
                return
            
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2])
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # Mini sparkline (7 derniers points)
            recent_data = data['Close'].tail(7).values
            sparkline = create_mini_sparkline(recent_data, change_pct >= 0)
            
            # Couleurs
            color = "#10B981" if change_pct >= 0 else "#EF4444"
            bg_color = "rgba(16, 185, 129, 0.1)" if change_pct >= 0 else "rgba(239, 68, 68, 0.1)"
            arrow = "↑" if change_pct >= 0 else "↓"
            
            # Format du prix selon le ticker
            if ticker == '^VIX':
                price_display = f"{current_price:.2f}"
            elif ticker == '^TNX':
                price_display = f"{current_price:.3f}%"
            elif ticker in ['GC=F', 'CL=F']:
                price_display = f"${current_price:.2f}"
            else:
                price_display = f"{current_price:,.2f}"
            
            st.html(f"""
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            "
            onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(99, 102, 241, 0.3)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(148, 163, 184, 0.1)';"
            >
                <!-- Top: Name & Description -->
                <div style="margin-bottom: 0.8rem;">
                    <div style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 0.2rem;">
                        {name}
                    </div>
                    <div style="font-size: 0.75rem; opacity: 0.6; color: #94A3B8;">
                        {description}
                    </div>
                </div>
                
                <!-- Price -->
                <div style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    {price_display}
                </div>
                
                <!-- Change Badge -->
                <div style="
                    display: inline-block;
                    background: {bg_color};
                    color: {color};
                    padding: 0.25rem 0.75rem;
                    border-radius: 6px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                ">
                    {arrow} {change_pct:+.2f}%
                </div>
                
                <!-- Mini Sparkline -->
                <div style="margin-top: 0.5rem; height: 30px;">
                    {sparkline}
                </div>
                
                <!-- Ticker at bottom -->
                <div style="
                    position: absolute;
                    bottom: 0.5rem;
                    right: 0.75rem;
                    font-size: 0.7rem;
                    opacity: 0.4;
                    font-weight: 600;
                ">
                    {ticker}
                </div>
            </div>
            """)
            
        else:
            # Fallback si pas de données
            render_index_card_fallback(ticker, name, description)
            
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        render_index_card_fallback(ticker, name, description)


def render_index_card_fallback(ticker, name, description):
    """Carte fallback si données indisponibles"""
    st.html(f"""
    <div style="
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    ">
        <div style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 0.5rem;">
            {name}
        </div>
        <div style="font-size: 0.75rem; opacity: 0.6; color: #94A3B8; margin-bottom: 0.5rem;">
            {description}
        </div>
        <div style="font-size: 1rem; color: #94A3B8; margin-top: 0.5rem;">
            Loading...
        </div>
    </div>
    """)


def create_mini_sparkline(data, is_positive):
    """Crée une mini sparkline SVG"""
    if len(data) < 2:
        return ""
    
    # Normaliser entre 0 et 30 (hauteur)
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        normalized = np.full_like(data, 15.0)
    else:
        normalized = 30 - ((data - min_val) / (max_val - min_val) * 30)
    
    # Créer les points du polyline
    width = 100
    step = width / (len(data) - 1)
    points = " ".join([f"{i*step},{val}" for i, val in enumerate(normalized)])
    
    color = "#10B981" if is_positive else "#EF4444"
    
    return f"""
    <svg width="100%" height="30" style="display: block;">
        <polyline
            points="{points}"
            fill="none"
            stroke="{color}"
            stroke-width="2"
            opacity="0.8"
        />
    </svg>
    """


def render_indices_charts(theme):
    """Graphiques comparatifs des indices"""
    st.markdown("""
    <div style="
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: #F8FAFC;
    ">
        📈 Indices Performance
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"],
            index=2,
            key="indices_chart_period"
        )
        
        indices_to_plot = st.multiselect(
            "Select Indices",
            ['^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^GDAXI'],
            default=['^GSPC', '^IXIC', '^DJI'],
            key="indices_selection"
        )
    
    with col1:
        if indices_to_plot:
            fig = create_indices_comparison_chart(indices_to_plot, period, theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one index to display")


def create_indices_comparison_chart(tickers, period, theme):
    """Crée un graphique de comparaison normalisé"""
    fig = go.Figure()
    
    colors = ['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']
    
    for idx, ticker in enumerate(tickers):
        try:
            # Utiliser yahoo avec la bonne fonction
            data = yahoo.get_ticker_data(ticker, period=period)
            print(data)
            if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                if 'Close' not in data.columns:
                    st.warning(f"No Close data for {ticker}")
                    continue
                
                # Normaliser à 100
                close_prices = data['Close'].dropna()
                if len(close_prices) > 0:
                    normalized = (close_prices / close_prices.iloc[0])*100  
                    print(normalized.values)
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=[t[0] for t in normalized.values],
                        mode='lines',
                        name=ticker.replace('^', ''),
                        line=dict(width=3, color=colors[idx % len(colors)]),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Date: %{x}<br>' +
                                      'Value: %{y:.2f}<br>' +
                                      '<extra></extra>'
                    ))
        except Exception as e:
            st.warning(f"Could not load data for {ticker}: {str(e)}")
            continue
    
    fig.update_layout(
        title=dict(
            text="Normalized Performance (Base 100)",
            font=dict(size=18, color='#F8FAFC')
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            color='#94A3B8'
        ),
        yaxis=dict(
            title="Index Value",
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            color='#94A3B8'
        ),
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Ligne de référence à 100
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.3)",
        annotation_text="Base",
        annotation_position="right"
    )
    
    return fig


def render_sector_heatmap(theme):
    """Heatmap des secteurs"""
    st.markdown("""
    <div style="
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        color: #F8FAFC;
    ">
        🏢 Sector Performance (Today)
    </div>
    """, unsafe_allow_html=True)
    
    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Disc.',
        'XLP': 'Consumer Staples',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLC': 'Communication'
    }
    
    sector_data = []
    
    for ticker, name in sector_etfs.items():
        try:
            # Utiliser yahoo avec la bonne fonction
            data = yahoo.get_ticker_data(ticker, period='5d')
            
            if data is not None and isinstance(data, pd.DataFrame) and not data.empty and len(data) > 1:
                if 'Close' in data.columns:
                    current = float(data['Close'].iloc[-1])
                    prev = float(data['Close'].iloc[-2])
                    change_pct = ((current - prev) / prev) * 100
                    
                    sector_data.append({
                        'Sector': name,
                        'Change': change_pct
                    })
        except Exception as e:
            continue
    
    if sector_data:
        df = pd.DataFrame(sector_data).sort_values('Change', ascending=False)
        
        # Heatmap style bars
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df['Sector'],
            x=df['Change'],
            orientation='h',
            marker=dict(
                color=df['Change'],
                colorscale=[
                    [0, '#EF4444'],
                    [0.5, '#94A3B8'],
                    [1, '#10B981']
                ],
                showscale=False
            ),
            text=[f"{x:+.2f}%" for x in df['Change']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis=dict(
                title="Change (%)",
                showgrid=True,
                gridcolor='rgba(148, 163, 184, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(148, 163, 184, 0.3)',
                color='#94A3B8'
            ),
            yaxis=dict(
                title="",
                color='#94A3B8'
            ),
            template=theme['plotly_template'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            showlegend=False,
            margin=dict(l=150, r=50, t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sector data not available")


def render_market_breadth(theme):
    """Market breadth et statistiques"""
    st.markdown("""
    <div style="
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.8rem 0;
        color: #F8FAFC;
    ">
        📊 Market Statistics
    </div>
    """, unsafe_allow_html=True)
    
    # Simuler des données (dans un vrai système, utiliser une API)
    advances = np.random.randint(1800, 2200)
    declines = np.random.randint(1200, 1600)
    unchanged = np.random.randint(100, 300)
    
    total = advances + declines + unchanged
    
    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=['Advances', 'Declines', 'Unchanged'],
        values=[advances, declines, unchanged],
        hole=0.6,
        marker=dict(colors=['#10B981', '#EF4444', '#94A3B8']),
        textinfo='label+percent',
        textfont=dict(size=12, color='#F8FAFC')
    )])
    
    fig.update_layout(
        title=dict(
            text="NYSE Advances/Declines",
            font=dict(size=14, color='#F8FAFC')
        ),
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        showlegend=False,
        annotations=[dict(
            text=f'{total}<br>stocks',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font=dict(color='#F8FAFC')
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics supplémentaires
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            margin: 0.5rem 0;
        ">
            <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.3rem;">New Highs</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #10B981;">{np.random.randint(200, 400)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(239, 68, 68, 0.2);
            margin: 0.5rem 0;
        ">
            <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.3rem;">New Lows</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #EF4444;">{np.random.randint(50, 150)}</div>
        </div>
        """, unsafe_allow_html=True)


def render_top_movers(theme):
    """Top gainers et losers"""
    st.markdown("""
    <div style="
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: #F8FAFC;
    ">
        🔥 Top Movers
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Top Gainers
    with col1:
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; color: #10B981;">
            📈 Top Gainers
        </div>
        """, unsafe_allow_html=True)
        
        # Simuler des données (remplacer par vraie API)
        gainers = [
            {'ticker': 'NVDA', 'name': 'NVIDIA Corp', 'change': 8.5, 'price': 485.20},
            {'ticker': 'AMD', 'name': 'AMD Inc', 'change': 6.2, 'price': 142.30},
            {'ticker': 'TSLA', 'name': 'Tesla Inc', 'change': 5.8, 'price': 245.60},
            {'ticker': 'META', 'name': 'Meta Platforms', 'change': 4.3, 'price': 512.80},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc', 'change': 3.9, 'price': 178.45}
        ]
        
        for stock in gainers:
            st.html(f"""
            <div style="
                background: rgba(16, 185, 129, 0.05);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <div style="font-weight: 700; color: #F8FAFC;">{stock['ticker']}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">{stock['name']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600; color: #F8FAFC;">${stock['price']:.2f}</div>
                    <div style="font-weight: 600; color: #10B981;">+{stock['change']:.2f}%</div>
                </div>
            </div>
            """)
    
    # Top Losers
    with col2:
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; color: #EF4444;">
            📉 Top Losers
        </div>
        """, unsafe_allow_html=True)
        
        losers = [
            {'ticker': 'NFLX', 'name': 'Netflix Inc', 'change': -5.2, 'price': 628.10},
            {'ticker': 'BA', 'name': 'Boeing Co', 'change': -4.8, 'price': 182.50},
            {'ticker': 'DIS', 'name': 'Walt Disney', 'change': -3.6, 'price': 98.75},
            {'ticker': 'PYPL', 'name': 'PayPal Holdings', 'change': -3.2, 'price': 65.40},
            {'ticker': 'INTC', 'name': 'Intel Corp', 'change': -2.9, 'price': 42.15}
        ]
        
        for stock in losers:
            st.html(f"""
            <div style="
                background: rgba(239, 68, 68, 0.05);
                border: 1px solid rgba(239, 68, 68, 0.2);
                border-radius: 8px;
                padding: 0.75rem;
                margin: 0.5rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <div style="font-weight: 700; color: #F8FAFC;">{stock['ticker']}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">{stock['name']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600; color: #F8FAFC;">${stock['price']:.2f}</div>
                    <div style="font-weight: 600; color: #EF4444;">{stock['change']:.2f}%</div>
                </div>
            </div>
            """)
