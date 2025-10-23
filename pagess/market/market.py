# pages/market.py
"""
Market Page - Hub complet pour explorer les marchés
4 onglets : Overview, Explorer, Screener, Watchlist
"""

import streamlit as st
from uiconfig import get_theme_colors
from .market_overview import render_market_overview
from .explorer import render_explorer
from .screener import render_screener


def render_market():
    """Page Market principale avec 4 onglets"""
    theme = get_theme_colors()
    
    st.markdown("<h1>📊 Market</h1>", unsafe_allow_html=True)
    
    # 4 onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🔍 Explorer", 
        "🎯 Screener", 
        "⭐ Watchlist"
    ])
    
    with tab1:
        render_market_overview()
    
    with tab2:
        render_explorer()
    
    with tab3:
        render_screener()
    
    with tab4:
        render_watchlist(theme)


def render_watchlist(theme):
    """Watchlist personnalisée de l'utilisateur"""
    st.markdown("### ⭐ My Watchlist")
    
    st.info("""
    **Watchlist** vous permet de suivre vos actions favorites.
    Ajoutez des tickers pour surveiller leurs performances en temps réel.
    """)
    
    # Session state pour la watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Ajouter un ticker
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_ticker = st.text_input(
            "Add Ticker",
            placeholder="AAPL",
            key="watchlist_add"
        )
    
    with col2:
        if st.button("➕ Add", use_container_width=True):
            if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker.upper())
                st.success(f"✅ {new_ticker.upper()} added!")
                st.rerun()
    
    st.markdown("---")
    
    # Afficher la watchlist
    if st.session_state.watchlist:
        render_watchlist_cards(st.session_state.watchlist, theme)
    else:
        st.info("Your watchlist is empty. Add some tickers to get started!")


def render_watchlist_cards(tickers, theme):
    """Affiche les cartes de la watchlist"""
    from stock import Stock
    import plotly.graph_objects as go
    
    # Options d'affichage
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        view_mode = st.radio(
            "View",
            ["Compact", "Detailed"],
            horizontal=True,
            key="watchlist_view"
        )
    
    with col2:
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo"],
            index=1,
            key="watchlist_period"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Ticker", "Change %", "Price"],
            key="watchlist_sort"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charger les données
    watchlist_data = []
    
    for ticker in tickers:
        try:
            stock = Stock(ticker)
            data = stock.retrieve_data(period=period)
            
            if not data.empty and len(data) > 1:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change = current - prev
                change_pct = (change / prev) * 100
                
                watchlist_data.append({
                    'ticker': ticker,
                    'current': current,
                    'change': change,
                    'change_pct': change_pct,
                    'data': data
                })
        except:
            continue
    
    # Trier
    if sort_by == "Change %":
        watchlist_data.sort(key=lambda x: x['change_pct'], reverse=True)
    elif sort_by == "Price":
        watchlist_data.sort(key=lambda x: x['current'], reverse=True)
    else:
        watchlist_data.sort(key=lambda x: x['ticker'])
    
    # Afficher selon le mode
    if view_mode == "Compact":
        # Vue compacte : 3 cartes par ligne
        cols_per_row = 3
        for i in range(0, len(watchlist_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(watchlist_data):
                    with col:
                        render_compact_watchlist_card(watchlist_data[i + j], theme)
    else:
        # Vue détaillée : 2 cartes par ligne avec mini-chart
        cols_per_row = 2
        for i in range(0, len(watchlist_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(watchlist_data):
                    with col:
                        render_detailed_watchlist_card(watchlist_data[i + j], theme)


def render_compact_watchlist_card(stock_data, theme):
    """Carte watchlist compacte"""
    ticker = stock_data['ticker']
    current = stock_data['current']
    change_pct = stock_data['change_pct']
    
    color = "#10B981" if change_pct >= 0 else "#EF4444"
    bg_color = "rgba(16, 185, 129, 0.1)" if change_pct >= 0 else "rgba(239, 68, 68, 0.1)"
    arrow = "↑" if change_pct >= 0 else "↓"
    
    st.html(f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        position: relative;
    "
    onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(99, 102, 241, 0.3)';"
    onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(148, 163, 184, 0.1)';"
    >
        <!-- Header: Ticker + Remove button -->
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #F8FAFC;">
                {ticker}
            </div>
        </div>
        
        <!-- Price -->
        <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
            ${current:.2f}
        </div>
        
        <!-- Change Badge -->
        <div style="
            display: inline-block;
            background: {bg_color};
            color: {color};
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
        ">
            {arrow} {change_pct:+.2f}%
        </div>
    </div>
    """)
    
    # Bouton remove
    if st.button("🗑️", key=f"remove_{ticker}", help="Remove from watchlist"):
        st.session_state.watchlist.remove(ticker)
        st.rerun()


def render_detailed_watchlist_card(stock_data, theme):
    """Carte watchlist détaillée avec mini-chart"""
    import plotly.graph_objects as go
    
    ticker = stock_data['ticker']
    current = stock_data['current']
    change_pct = stock_data['change_pct']
    data = stock_data['data']
    
    color = "#10B981" if change_pct >= 0 else "#EF4444"
    arrow = "↑" if change_pct >= 0 else "↓"
    
    # Mini chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        line=dict(color=color, width=2),
        fill='tonexty',
        fillcolor=f'rgba({16 if change_pct >= 0 else 239}, {185 if change_pct >= 0 else 68}, {129 if change_pct >= 0 else 68}, 0.1)',
        showlegend=False,
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        hovermode='x'
    )
    
    # Container
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.html(f"""
            <div style="padding: 0.5rem 0;">
                <div style="font-weight: 700; font-size: 1.2rem; color: #F8FAFC; margin-bottom: 0.5rem;">
                    {ticker}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    ${current:.2f}
                </div>
                <div style="
                    display: inline-block;
                    background: rgba({16 if change_pct >= 0 else 239}, {185 if change_pct >= 0 else 68}, {129 if change_pct >= 0 else 68}, 0.1);
                    color: {color};
                    padding: 0.3rem 0.6rem;
                    border-radius: 6px;
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    {arrow} {change_pct:+.2f}%
                </div>
            </div>
            """)
            
            if st.button("🗑️", key=f"remove_detailed_{ticker}", help="Remove"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()
        
        with col2:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<hr style='margin: 0.5rem 0; border-color: rgba(148, 163, 184, 0.1);'>", unsafe_allow_html=True)
