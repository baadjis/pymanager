# pages/portfolio_manager.py
"""
Page Portfolio Manager V2 avec structure holdings[]
✅ Utilise get_portfolio_summary() unifié
✅ PnL identique dashboard et portfolio_details
✅ Support ancienne ET nouvelle structure
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from portfolio import Portfolio
from factory import create_portfolio_by_name, create_benchmark
from dataprovider import yahoo
from database import get_portfolios, get_single_portfolio, save_portfolio
from utils import get_portfolio_summary
from uiconfig import get_theme_colors
from .portfolio_helpers import (
    color_column, color_background_column, cached_get_sectors_weights,
    METRIC_TOOLTIPS, create_metric_card, render_advanced_metrics_section
)
from .portfolio_builder import render_portfolio_build_tab, display_portfolio_results
from .experiments_tab import render_experiments_tab

user_id = st.session_state.user_id


def load_enhanced_styles():
    """Charge les styles CSS personnalisés"""
    css = """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .positive { color: #10B981; }
    .negative { color: #EF4444; }
    .warning { color: #F59E0B; }
    
    @media (max-width: 768px) {
        .metric-card { padding: 16px; }
        .metric-value { font-size: 24px; }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_portfolio_manager():
    """Page Portfolio Manager principale avec 4 onglets"""
    load_enhanced_styles()
    
    st.markdown("<h1>💼 Portfolio Manager</h1>", unsafe_allow_html=True)
    
    # 4 onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔨 Build Portfolio", 
        "💼 My Portfolios", 
        "📋 Portfolio Details",
        "🧪 Experiments"
    ])
    
    with tab1:
        render_portfolio_build_tab()
    
    with tab2:
        render_portfolio_list_tab()
    
    with tab3:
        render_portfolio_details_tab()
    
    with tab4:
        render_experiments_tab()


def render_portfolio_list_tab():
    """Onglet liste des portfolios - ✅ Utilise get_portfolio_summary()"""
    theme = get_theme_colors()
    st.markdown("### My Portfolios")
    
    try:
        portfolios = list(get_portfolios(user_id=user_id))
        
        if not portfolios:
            st.info("No portfolios saved yet. Create one in 'Build Portfolio' tab!")
        else:
            cols = st.columns(2)
            
            for idx, pf in enumerate(portfolios):
                with cols[idx % 2]:
                    # ✅ CALCUL UNIFIÉ
                    summary = get_portfolio_summary(pf)
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3 style="color: {theme['text_primary']}; margin: 0 0 12px 0;">{summary['name']}</h3>
                        <div style="color: {theme['text_secondary']}; margin: 6px 0;">
                            <strong>Model:</strong> {summary['model'].title()}
                        </div>
                        <div style="color: {theme['text_secondary']}; margin: 6px 0;">
                            <strong>Method:</strong> {summary['method'].title()}
                        </div>
                        <div style="color: {theme['text_secondary']}; margin: 6px 0;">
                            <strong>Holdings:</strong> {summary['num_holdings']} assets
                        </div>
                        <hr style="border-color: {theme['border']}; margin: 12px 0;">
                        <div class="metric-label">Initial Investment</div>
                        <div style="font-size: 18px; color: {theme['text_secondary']}; margin-bottom: 8px;">
                            ${summary['initial_amount']:,.2f}
                        </div>
                        <div class="metric-label">Current Value</div>
                        <div class="metric-value">${summary['current_value']:,.2f}</div>
                        <div style="margin-top: 8px;">
                            <span class="{'positive' if summary['pnl'] >= 0 else 'negative'}" 
                                  style="font-size: 18px; font-weight: 700;">
                                ${summary['pnl']:+,.2f} ({summary['pnl_pct']:+.2f}%)
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())


def render_portfolio_details_tab():
    """
    Onglet détails portfolio V2
    ✅ Utilise get_portfolio_summary() pour PnL unifié
    ✅ Holdings details depuis summary['holdings']
    """
    theme = get_theme_colors()
    st.markdown("### Portfolio Details")
    
    try:
        portfolios = list(get_portfolios(user_id=user_id))
        portfolio_names = [p['name'] for p in portfolios]
        
        if not portfolio_names:
            st.info("No portfolios available. Create one first!")
            return
        
        selected = st.selectbox("Select Portfolio", portfolio_names, key="portfolio_selector")
        
        if not selected:
            return
        
        portfolio = get_single_portfolio(user_id, selected)
        
        # ✅ CALCUL UNIFIÉ - même fonction que dashboard!
        summary = get_portfolio_summary(portfolio)
        
        # Extraire les valeurs
        initial_amount = summary['initial_amount']
        current_value = summary['current_value']
        pnl = summary['pnl']
        pnl_pct = summary['pnl_pct']
        holdings_details = summary['holdings']
        
        # Section métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Initial Investment</div>
                <div class="metric-value">${initial_amount:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">${current_value:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            pnl_color = "positive" if pnl >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="{pnl_color}" style="font-size: 28px; font-weight: 700; margin-top: 8px;">
                    ${pnl:+,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Return</div>
                <div class="{pnl_color}" style="font-size: 28px; font-weight: 700; margin-top: 8px;">
                    {pnl_pct:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs détails
        detail_tabs = st.tabs(["Holdings", "Composition", "Analytics", "Info"])
        
        with detail_tabs[0]:
            render_holdings_tab(holdings_details, theme)
        
        with detail_tabs[1]:
            render_composition_tab(portfolio, holdings_details, theme)
        
        with detail_tabs[2]:
            render_analytics_tab(portfolio, holdings_details)
        
        with detail_tabs[3]:
            render_info_tab(portfolio, summary, theme)
    
    except Exception as e:
        st.error(f"Error loading portfolio details: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())


def render_holdings_tab(holdings_details, theme):
    """Affiche les holdings avec DataFrame stylé"""
    st.markdown("### Holdings")
    
    if not holdings_details:
        st.info("No holdings data available")
        return
    
    # Créer DataFrame depuis holdings_details
    df_data = []
    for h in holdings_details:
        df_data.append({
            "Asset": h['symbol'],
            "Type": h['type'].title(),
            "Weight": h['weight'],
            "Quantity": h['quantity'],
            "Initial Price": h['initial_price'],
            "Current Price": h['current_price'],
            "Initial Value": h['initial_value'],
            "Market Value": h['market_value'],
            "P&L": h['pnl'],
            "P&L %": h['pnl_pct']
        })
    
    df = pd.DataFrame(df_data)
    
    # Styling
    def color_pnl(val):
        color = 'green' if val >= 0 else 'red'
        return f'background-color: {color}; color: white; font-weight: bold;'
    
    styler = df.style
    
    # Format colonnes
    styler.format({
        "Weight": "{:.2%}",
        "Quantity": "{:.4f}",
        "Initial Price": "${:.2f}",
        "Current Price": "${:.2f}",
        "Initial Value": "${:,.2f}",
        "Market Value": "${:,.2f}",
        "P&L": "${:+,.2f}",
        "P&L %": "{:+.2f}%"
    })
    
    # Colorer P&L
    styler.applymap(color_pnl, subset=["P&L", "P&L %"])
    
    # Cacher index
    styler.hide(axis='index')
    
    st.write(styler.to_html(), unsafe_allow_html=True)
    
    # Résumé
    total_initial = sum(h['initial_value'] for h in holdings_details)
    total_current = sum(h['market_value'] for h in holdings_details)
    total_pnl = total_current - total_initial
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Initial", f"${total_initial:,.2f}")
    with col2:
        st.metric("Total Current", f"${total_current:,.2f}")
    with col3:
        st.metric("Total P&L", f"${total_pnl:+,.2f}", f"{(total_pnl/total_initial*100):+.2f}%")


def render_composition_tab(portfolio, holdings_details, theme):
    """Affiche les graphiques de composition"""
    st.markdown("### Asset & Sector Allocation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset Allocation (par poids)
        if holdings_details:
            labels = [h['symbol'] for h in holdings_details]
            values = [h['weight'] for h in holdings_details]
            
            fig_assets = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']),
                textfont=dict(size=14, color='#F8FAFC')
            )])
            fig_assets.update_layout(
                template='plotly_dark',
                paper_bgcolor=theme['bg_card'],
                font=dict(color=theme['text_primary']),
                title=dict(text="Asset Allocation (by Weight)", font=dict(size=18,color=theme['text_primary'])),
                height=400,
                showlegend=True,
                legend=dict(font=dict(size=12,color=theme['text_primary']
                    ),
      
                    )
            )
            st.plotly_chart(fig_assets, use_container_width=True)
    
    with col2:
        # Sector Allocation
        try:
            symbols = [h['symbol'] for h in holdings_details]
            weights = [h['weight'] for h in holdings_details]
            sector_weights = yahoo.get_sectors_weights(symbols, weights)
            
            if sector_weights:
                fig_sectors = go.Figure(data=[go.Pie(
                    labels=list(sector_weights.keys()),
                    values=list(sector_weights.values()),
                    hole=0.4,
                    marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']),
                    textfont=dict(size=14, color='#F8FAFC')
                )])
                fig_sectors.update_layout(
                    template='plotly_dark',
                    paper_bgcolor=theme['bg_card'],
                    font=dict(color=theme['text_primary']),
                    title=dict(text="Sector Allocation", font=dict(size=18,color=theme['text_primary'])),
                    height=400,
                    showlegend=True,
                    legend=dict(font=dict(size=12,color=theme['text_primary']
                    ),
      
                    )
                )
                st.plotly_chart(fig_sectors, use_container_width=True)
            else:
                st.info("Sector data not available")
        except Exception as e:
            st.info("Sector data not available")
    
    # Value Allocation (par market value actuelle)
    st.markdown("### Current Value Distribution")
    
    if holdings_details:
        labels = [h['symbol'] for h in holdings_details]
        values = [h['market_value'] for h in holdings_details]
        
        fig_value = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']),
            textfont=dict(size=14, color='#F8FAFC')
        )])
        fig_value.update_layout(
            template='plotly_dark',
            paper_bgcolor=theme['bg_card'],
            font=dict(color=theme['text_primary']),
            title=dict(text="Current Value Distribution", font=dict(size=18,color=theme['text_primary'])),
            height=400,
            showlegend=True,
            legend=dict(font=dict(size=12,color=theme['text_primary']
                    ),
      
                    )
        )
        st.plotly_chart(fig_value, use_container_width=True)


def render_analytics_tab(portfolio, holdings_details):
    """Affiche les analytics avancés"""
    st.markdown("### Portfolio Analytics")
    
    try:
        # Extraire les symbols et weights depuis holdings
        if 'holdings' in portfolio:
            symbols = [h['symbol'] for h in portfolio['holdings']]
            weights = [h['weight'] for h in portfolio['holdings']]
        else:
            # Legacy structure
            symbols = portfolio.get('assets', [])
            weights = portfolio.get('weights', [])
        
        # Charger les données complètes
        full_data = yahoo.retrieve_data(tuple(symbols))
        port = Portfolio(symbols, full_data)
        port.set_weights(weights)
        
        # Charger le benchmark
        benchmark = None
        try:
            benchmark = create_benchmark("^GSPC", period="5y")
        except Exception as e:
            st.warning("Could not load benchmark for comparison")
        
        # Afficher les métriques avancées
        render_advanced_metrics_section(port, benchmark)
        
    except Exception as e:
        st.error(f"Could not calculate advanced metrics: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())


def render_info_tab(portfolio, summary, theme):
    """Affiche les informations du portfolio"""
    st.markdown("### Portfolio Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                {summary['model'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Method</div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                {summary['method'].title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Number of Holdings</div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                {summary['num_holdings']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Created</div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                {portfolio.get('created_at', 'Unknown')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculer total weight
        if summary['holdings']:
            total_weight = sum(h['weight'] for h in summary['holdings'])
        else:
            total_weight = sum(portfolio.get('weights', []))
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Weight</div>
            <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                {total_weight:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assets list
        if summary['holdings']:
            assets_list = ', '.join([h['symbol'] for h in summary['holdings']])
        else:
            assets_list = ', '.join(portfolio.get('assets', []))
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Assets</div>
            <div style="font-size: 14px; margin-top: 8px;">
                {assets_list}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metadata
    if 'metadata' in portfolio:
        st.markdown("### Metadata")
        metadata = portfolio['metadata']
        
        meta_col1, meta_col2 = st.columns(2)
        
        with meta_col1:
            if metadata.get('risk_profile'):
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Profile</div>
                    <div style="font-size: 16px; margin-top: 8px;">
                        {metadata['risk_profile'].title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if metadata.get('investment_goal'):
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Investment Goal</div>
                    <div style="font-size: 16px; margin-top: 8px;">
                        {metadata['investment_goal'].title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with meta_col2:
            if metadata.get('time_horizon'):
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Time Horizon</div>
                    <div style="font-size: 16px; margin-top: 8px;">
                        {metadata['time_horizon'].title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if metadata.get('description'):
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Description</div>
                    <div style="font-size: 14px; margin-top: 8px;">
                        {metadata['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
