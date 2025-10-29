# pages/dashboard.py
"""
Dashboard - Vue d'ensemble complète et moderne
AUM, Cash, Portfolios, Watchlist, Market Overview, Performance
Optimisé sans espaces vides et alignement parfait
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from uiconfig import get_theme_colors, apply_plotly_theme, format_number
from dataprovider import yahoo
from database import get_watchlist
from .auth import render_auth
from utils import calculate_portfolio_current_value, format_pnl

user_id=''
try :
  user_id = st.session_state.user_id
except:
  
  render_auth()
  st.stop()

try:
    from database import get_portfolios
except:
    def get_portfolios(user_id):
        return []


def render_dashboard():
    """Dashboard principal avec toutes les métriques clés"""
    theme = get_theme_colors()
    
    # Header compact
    st.html(f"""
    <div style="
        background: {theme['gradient_primary']};
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
    ">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">
            Welcome back, {st.session_state.get('user_name', 'Portfolio Manager')} 👋
        </h1>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">
            {datetime.now().strftime('%A, %B %d, %Y')} • {datetime.now().strftime('%H:%M')}
        </p>
    </div>
    """)
    
    # Section 1: KPI Cards (4 colonnes)
    render_kpi_section(theme)
    
    # Section 2: Row 1 - Performance Chart + Allocation (ratio 2:1)
    col1, col2 = st.columns([2, 1], gap="medium")
    with col1:
        render_portfolio_performance_chart(theme)
    with col2:
        render_allocation_chart(theme)
    
    # Section 3: Row 2 - Portfolios + Watchlist (ratio 1:1)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        render_portfolios_overview(theme)
    with col2:
        render_watchlist_overview(theme)
    
    # Section 4: Row 3 - Market Overview + Recent Activity (ratio 1:1)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        render_market_overview(theme)
    with col2:
        render_recent_activity(theme)


def render_kpi_section(theme):
    """Section KPI avec 4 cartes principales"""
    
    # Calculer les métriques
    
    try:
        portfolios = list(get_portfolios(user_id=user_id))
        total_value=0.0
        total_pnl=0.0
        for portfolio in portfolios:
             current_value, pnl, _ = calculate_portfolio_current_value(portfolio)
             total_value += current_value
             total_pnl+=pnl
        num_portfolios = len(portfolios)
    except:
        total_value = 125430.50
        num_portfolios = 3
        total_pnl=9.234
    cash = 15230.75
    performance_pct = 12.34
    performance_value = 13658.92
    
    col1, col2, col3, col4 = st.columns(4, gap="small")
    
    with col1:
        render_kpi_card("Assets Under Management", format_number(total_value, 'currency', 0), 
                       8.5, f"+${total_pnl:.3f}", "💼", theme['primary_color'], theme)
    
    with col2:
        render_kpi_card("Cash Available", format_number(cash, 'currency', 2), 
                       2.1, "+$312", "💵", theme['success_color'], theme)
    
    with col3:
        render_kpi_card("Total Return", format_number(performance_value, 'currency', 2), 
                       performance_pct, f"+{performance_pct:.2f}%", "📈", theme['info_color'], theme)
    
    with col4:
        render_kpi_card("Active Portfolios", str(num_portfolios), 
                       0, "View All →", "🗂️", theme['warning_color'], theme, is_count=True)


def render_kpi_card(title, value, change_pct, change_value, icon, color, theme, is_count=False):
    """Render une carte KPI compacte"""
    
    change_color = theme['success_color'] if change_pct >= 0 else theme['danger_color']
    arrow = "↑" if change_pct >= 0 else "↓"
    
    st.html(f"""
    <div style="
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 10px;
        padding: 1rem;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    "
    onmouseover="this.style.transform='translateY(-2px)'; this.style.borderColor='{color}';"
    onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='{theme['border']}';"
    >
        <div style="position: absolute; top: 0.75rem; right: 0.75rem; font-size: 1.5rem; opacity: 0.2;">
            {icon}
        </div>
        
        <div style="font-size: 0.7rem; font-weight: 600; color: {theme['text_secondary']}; 
                    text-transform: uppercase; letter-spacing: 0.5px;">
            {title}
        </div>
        
        <div style="font-size: {'1.8rem' if is_count else '1.6rem'}; font-weight: 700; 
                    color: {theme['text_primary']}; line-height: 1;">
            {value}
        </div>
        
        <div style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.5rem;
                    background: {'rgba(16, 185, 129, 0.1)' if change_pct >= 0 else 'rgba(239, 68, 68, 0.1)'};
                    color: {change_color}; border-radius: 4px; font-size: 0.7rem; font-weight: 600;
                    width: fit-content;">
            {arrow if not is_count else ''} {change_value}
        </div>
    </div>
    """)


def render_portfolio_performance_chart(theme):
    """Graphique de performance compact"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        📊 Portfolio Performance (30 Days)
    </div>
    """)
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    base = 100000
    values = base + np.cumsum(np.random.randn(30) * 500)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode='lines', name='Portfolio Value',
        line=dict(color=theme['primary_color'], width=2.5),
        fill='tozeroy', fillcolor=f'rgba(99, 102, 241, 0.1)',
        hovertemplate='<b>%{x|%b %d}</b><br>$%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=base, line_dash="dash", line_color=theme['text_secondary'], 
                  opacity=0.4, annotation_text="Initial", annotation_position="right")
    
    apply_plotly_theme(fig, title="", xaxis_title="", yaxis_title="Value ($)", height=280)
    
    st.plotly_chart(fig, use_container_width=True, key="perf_chart")


def render_allocation_chart(theme):
    """Graphique d'allocation compact"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        🎯 Asset Allocation
    </div>
    """)
    
    allocations = {'Stocks': 55, 'Bonds': 25, 'Cash': 12, 'Crypto': 5, 'Commodities': 3}
    colors = ['#6366F1', '#8B5CF6', '#10B981', '#F59E0B', '#EC4899']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(allocations.keys()), values=list(allocations.values()),
        hole=0.5, marker=dict(colors=colors),
        textposition='inside', textinfo='percent',
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
    )])
    
    apply_plotly_theme(fig, title="", height=280)
    fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=10, b=10),
                     legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05))
    
    st.plotly_chart(fig, use_container_width=True, key="alloc_chart")


def render_portfolios_overview(theme):
    """Portfolios overview compact"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        💼 Your Portfolios
    </div>
    """)
    
    try:
        portfolios = list(get_portfolios(user_id=user_id))
    except:
        portfolios = [
            {'name': 'Growth Portfolio', 'amount': 65000, 'change': 8.2, 'holdings': 12},
            {'name': 'Income Portfolio', 'amount': 45000, 'change': 3.5, 'holdings': 8},
            {'name': 'Tech Portfolio', 'amount': 15430, 'change': 15.7, 'holdings': 5}
        ]
    
    if not portfolios:
        st.info("No portfolios yet. Create one!")
        if st.button("➕ Create Portfolio", use_container_width=True):
            st.session_state.current_page = 'Portfolio'
            st.rerun()
        return
    
    # Container avec scroll
    container_html = f"""
    <div style="height: 320px; overflow-y: auto; padding-right: 0.5rem;">
    """
    
    for idx, portfolio in enumerate(portfolios[:5]):
        name = portfolio.get('name', f'Portfolio {idx+1}')
        value, _, change = calculate_portfolio_current_value(portfolio)
        #value = portfolio.get('total_amount', 0)
        #change = portfolio.get('total_pnl_pct', 0)
        holdings = len(portfolio.get('assets', []))
        
        change_color = theme['success_color'] if change >= 0 else theme['danger_color']
        arrow = "↑" if change >= 0 else "↓"
        
        container_html += f"""
        <div style="
            background: {theme['bg_card']}; border: 1px solid {theme['border']};
            border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem;
            transition: all 0.2s ease;
        "
        onmouseover="this.style.borderColor='{theme['border_hover']}';"
        onmouseout="this.style.borderColor='{theme['border']}';"
        >
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: {theme['text_primary']}; 
                                font-size: 0.95rem; margin-bottom: 0.2rem;">
                        {name}
                    </div>
                    <div style="font-size: 0.7rem; color: {theme['text_secondary']};">
                        {holdings} holdings
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 700; color: {theme['text_primary']}; 
                                font-size: 1rem; margin-bottom: 0.2rem;">
                        ${value:,.0f}
                    </div>
                    <div style="display: inline-block; padding: 0.15rem 0.4rem;
                                background: {'rgba(16, 185, 129, 0.1)' if change >= 0 else 'rgba(239, 68, 68, 0.1)'};
                                color: {change_color}; border-radius: 3px; 
                                font-size: 0.7rem; font-weight: 600;">
                        {arrow} {abs(change):.2f}%
                    </div>
                </div>
            </div>
        </div>
        """
    
    container_html += "</div>"
    st.html(container_html)
    
    if len(portfolios) > 5:
        if st.button("📂 View All Portfolios", use_container_width=True, key="view_all_pf"):
            st.session_state.current_page = 'Portfolio'
            st.rerun()


def render_watchlist_overview(theme):
    """Watchlist overview compact avec sparklines"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        ⭐ Watchlist
    </div>
    """)
    
    #watchlist = st.session_state.get('watchlist', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    #portfolios = get_portfolios(user_id=user_id)
    watchlist = get_watchlist(user_id=user_id)
    
    if not watchlist:
        st.info("Your watchlist is empty!")
        if st.button("➕ Add to Watchlist", use_container_width=True, key="add_watch"):
            st.session_state.current_page = 'Market'
            st.rerun()
        return
    
    # Container avec scroll
    container_html = f"""
    <div style="height: 320px; overflow-y: auto; padding-right: 0.5rem;">
    """
    
    for ticker in watchlist[:5]:
        try:
            data = yahoo.get_ticker_data(ticker, period='5d')
            
            if data is not None and not data.empty and len(data) > 1:
                current = float(data['Close'].iloc[-1])
                prev = float(data['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100
                
                change_color = theme['success_color'] if change_pct >= 0 else theme['danger_color']
                arrow = "↑" if change_pct >= 0 else "↓"
                
                prices = data['Close'].tail(7).values
                sparkline = create_mini_sparkline_svg(prices, change_pct >= 0, theme)
                
                container_html += f"""
                <div style="
                    background: {theme['bg_card']}; border: 1px solid {theme['border']};
                    border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem;
                ">
                    <div style="display: flex; justify-content: space-between; 
                                align-items: center; margin-bottom: 0.4rem;">
                        <div style="font-weight: 700; color: {theme['text_primary']}; font-size: 0.95rem;">
                            {ticker}
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 700; color: {theme['text_primary']}; font-size: 0.95rem;">
                                ${current:.2f}
                            </div>
                            <div style="color: {change_color}; font-size: 0.7rem; font-weight: 600;">
                                {arrow} {abs(change_pct):.2f}%
                            </div>
                        </div>
                    </div>
                    <div style="height: 35px;">
                        {sparkline}
                    </div>
                </div>
                """
        except:
            continue
    
    container_html += "</div>"
    st.html(container_html)
    
    if len(watchlist) > 5:
        if st.button("📋 View Full Watchlist", use_container_width=True, key="view_watch"):
            st.session_state.current_page = 'Market'
            st.rerun()


def render_market_overview(theme):
    """Market overview compact"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        🌍 Market Overview
    </div>
    """)
    
    indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ', '^RUT': 'Russell 2000'}
    
    container_html = f"""
    <div style="height: 320px; overflow-y: auto; padding-right: 0.5rem;">
    """
    
    for ticker, name in indices.items():
        try:
            data = yahoo.get_ticker_data(ticker, period='5d')
            
            if data is not None and not data.empty and len(data) > 1:
                current = float(data['Close'].iloc[-1])
                prev = float(data['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100
                
                change_color = theme['success_color'] if change_pct >= 0 else theme['danger_color']
                arrow = "↑" if change_pct >= 0 else "↓"
                
                container_html += f"""
                <div style="display: flex; justify-content: space-between; align-items: center;
                            padding: 0.75rem; margin-bottom: 0.5rem;
                            background: {theme['bg_card']}; border: 1px solid {theme['border']};
                            border-radius: 8px;">
                    <div>
                        <div style="font-weight: 600; color: {theme['text_primary']}; font-size: 0.9rem;">
                            {name}
                        </div>
                        <div style="font-size: 0.7rem; color: {theme['text_secondary']};">{ticker}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: 600; color: {theme['text_primary']}; font-size: 0.9rem;">
                            {current:,.2f}
                        </div>
                        <div style="color: {change_color}; font-size: 0.75rem; font-weight: 600;">
                            {arrow} {abs(change_pct):.2f}%
                        </div>
                    </div>
                </div>
                """
        except:
            continue
    
    container_html += "</div>"
    st.html(container_html)
    
    if st.button("📊 View Full Market", use_container_width=True, key="view_market"):
        st.session_state.current_page = 'Market'
        st.rerun()


def render_recent_activity(theme):
    """Recent activity compact"""
    
    st.html(f"""
    <div style="font-size: 1.1rem; font-weight: 600; color: {theme['text_primary']}; 
                margin-bottom: 0.75rem;">
        🔔 Recent Activity
    </div>
    """)
    
    activities = [
        {'type': 'buy', 'ticker': 'AAPL', 'action': 'Bought', 'shares': 10, 'price': 178.50, 'time': '2h ago'},
        {'type': 'sell', 'ticker': 'TSLA', 'action': 'Sold', 'shares': 5, 'price': 245.30, 'time': '5h ago'},
        {'type': 'dividend', 'ticker': 'MSFT', 'action': 'Dividend', 'amount': 25.50, 'time': '1d ago'},
        {'type': 'buy', 'ticker': 'GOOGL', 'action': 'Bought', 'shares': 8, 'price': 142.15, 'time': '2d ago'},
        {'type': 'alert', 'ticker': 'NVDA', 'action': 'Price Alert', 'message': 'Target reached', 'time': '3d ago'}
    ]
    
    icon_map = {'buy': '🟢', 'sell': '🔴', 'dividend': '💵', 'alert': '🔔'}
    
    container_html = f"""
    <div style="height: 320px; overflow-y: auto; padding-right: 0.5rem;">
    """
    
    for activity in activities:
        icon = icon_map.get(activity['type'], '📌')
        
        if activity['type'] in ['buy', 'sell']:
            detail = f"{activity['shares']} shares @ ${activity['price']:.2f}"
        elif activity['type'] == 'dividend':
            detail = f"${activity['amount']:.2f}"
        else:
            detail = activity.get('message', '')
        
        container_html += f"""
        <div style="display: flex; gap: 0.75rem; padding: 0.75rem; margin-bottom: 0.5rem;
                    background: {theme['bg_card']}; border: 1px solid {theme['border']};
                    border-radius: 8px;">
            <div style="font-size: 1.2rem; flex-shrink: 0;">{icon}</div>
            <div style="flex: 1; min-width: 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.15rem;">
                    <span style="font-weight: 600; color: {theme['text_primary']}; font-size: 0.85rem;">
                        {activity['action']} {activity['ticker']}
                    </span>
                    <span style="font-size: 0.7rem; color: {theme['text_secondary']}; white-space: nowrap;">
                        {activity['time']}
                    </span>
                </div>
                <div style="font-size: 0.75rem; color: {theme['text_secondary']};">
                    {detail}
                </div>
            </div>
        </div>
        """
    
    container_html += "</div>"
    st.html(container_html)


def create_mini_sparkline_svg(data, is_positive, theme):
    """Crée une mini sparkline SVG optimisée"""
    if len(data) < 2:
        return ""
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        normalized = np.full_like(data, 17.5)
    else:
        normalized = 35 - ((data - min_val) / (max_val - min_val) * 35)
    
    width = 100
    step = width / (len(data) - 1)
    points = " ".join([f"{i*step},{val}" for i, val in enumerate(normalized)])
    
    color = theme['success_color'] if is_positive else theme['danger_color']
    
    return f"""
    <svg width="100%" height="35" style="display: block;">
        <polyline points="{points}" fill="none" stroke="{color}" 
                  stroke-width="2" opacity="0.8"/>
    </svg>
    """
