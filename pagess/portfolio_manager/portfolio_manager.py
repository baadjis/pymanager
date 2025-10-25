# pages/portfolio_manager.py
"""
Page Portfolio Manager avec design amélioré et nouveaux ratios
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from portfolio import Portfolio
from factory import create_portfolio_by_name, create_benchmark
from dataprovider import yahoo
from database import get_portfolios, get_single_portfolio, save_portfolio
from utils import calculate_portfolio_current_value, format_pnl
from uiconfig import get_theme_colors
from .portfolio_helpers import color_column , color_background_column,cached_get_sectors_weights,METRIC_TOOLTIPS, create_metric_card,render_advanced_metrics_section
from .portfolio_builder import  render_portfolio_build_tab,display_portfolio_results
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
        "🧪 Experiments"  # NOUVEAU!
    ])
    
    with tab1:
        render_portfolio_build_tab()
    
    with tab2:
        render_portfolio_list_tab()
    
    with tab3:
        render_portfolio_details_tab()
    
    with tab4:
        render_experiments_tab()  # NOUVEAU!


def render_portfolio_list_tab():
    """Onglet liste des portfolios"""
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
                    current_value, pnl, pnl_pct = calculate_portfolio_current_value(pf)
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3 style="color: {theme['text_primary']}; margin: 0 0 12px 0;">{pf['name']}</h3>
                        <div style="color: {theme['text_secondary']}; margin: 6px 0;">
                            <strong>Model:</strong> {pf.get('model', 'N/A').title()}
                        </div>
                        <div style="color: {theme['text_secondary']}; margin: 6px 0;">
                            <strong>Method:</strong> {pf.get('method', 'N/A').title()}
                        </div>
                        <hr style="border-color: {theme['border']}; margin: 12px 0;">
                        <div class="metric-label">Current Value</div>
                        <div class="metric-value">${current_value:,.2f}</div>
                        <div style="margin-top: 8px;">
                            {format_pnl(pnl, pnl_pct)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")

def render_portfolio_details_tab():
    """Onglet détails portfolio avec métriques avancées"""
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
        
        portfolio = get_single_portfolio(user_id,selected)
        assets = portfolio["assets"]
        weights = portfolio["weights"]
        amount = portfolio["amount"]
        print(portfolio)
        data = yahoo.retrieve_data(tuple(assets), "1d")
        
        if isinstance(data.columns, pd.MultiIndex):
            
              
            latest_prices = data["Adj Close"].iloc[-1]
                
        else:
            
           latest_prices = data["Adj Close"].iloc[-1] 
        
        quantities = portfolio.get('quantities',[])
        portfolio_comp = []
        if quantities:
           for i, q in enumerate(quantities):
        
               #qty = quantities[i] 
            
               current_price = latest_prices[i]
               initial_amt = amount * weights[i]
            
               market_val = float(current_price) * float(q)
               print("details",q,current_price,market_val)
            
               portfolio_comp.append({
                "Asset": assets[i],
                "Weight": weights[i],
                "Initial Amount": initial_amt,
                "Quantity": q,
                "Current Price": current_price,
                "Market Value": market_val
            })
        
        df = pd.DataFrame(portfolio_comp)
        mtm = df["Market Value"].sum()
        
        pnl = mtm - amount
        pnl_pct = (pnl / amount) * 100 if amount > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Initial Investment</div>
                <div class="metric-value">${amount:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">${mtm:,.0f}</div>
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
        
        detail_tabs = st.tabs(["Holdings", "Composition", "Analytics", "Info"])
        
        with detail_tabs[0]:
            st.markdown("### Holdings")
            
            display_df = df.copy()
            pnl_values = [(mv - ia) for mv, ia in zip(df["Market Value"], df["Initial Amount"])]
            display_df["P&L"] = [f"${x:+,.2f}" for x in pnl_values]
            styler= display_df.style
            
            styler.format(lambda x: f"{x:,.2f}", subset=["Quantity","Initial Amount","Current Price","Market Value"])
            styler.format(lambda x: f"{x*100:,.2f}%", subset=["Weight"])
            styler.map(lambda x: f"background-color:{'green' if '+' in x else 'red'} !important;",subset=["P&L"])
            
            """display_df["Quantity"] = display_df["Quantity"].apply(lambda x: f"{x:,.2f}")
            display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x*100:.2f}%")
            display_df["Initial Amount"] = display_df["Initial Amount"].apply(lambda x: f"${x:,.2f}")
            display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"${x:.2f}")
            display_df["Market Value"] = display_df["Market Value"].apply(lambda x: f"${x:,.2f}")
            """
            styler.hide()
            #display_df.style.applymap(lambda x: f"background-color:'green' !important;",subset=["P&L"])
         
            #st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.write(styler.to_html(index=False), unsafe_allow_html=True)
            #style_data_frame(df, background=theme['bg_card'], color=theme['text_primary'])
        
        with detail_tabs[1]:
            st.markdown("### Asset & Sector Allocation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_assets = go.Figure(data=[go.Pie(
                    labels=assets,
                    values=weights,
                    hole=0.4,
                    marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']),
                    textfont=dict(size=14, color='#F8FAFC')
                )])
                fig_assets.update_layout(
                    template='plotly_dark',
                    paper_bgcolor=theme['bg_card'],
                     font=dict(color=theme['text_primary']),
                    title=dict(text="Asset Allocation", font=dict(size=18, color='#F8FAFC')),
                   
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_assets, use_container_width=True)
            
            with col2:
                try:
                    sector_weights = yahoo.get_sectors_weights(assets, weights)
                    
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
                            plot_bgcolor=theme['bg_card'],
                            font=dict(color=theme['text_primary']),
                            title=dict(text="Sector Allocation", font=dict(size=18, color=theme['text_primary'])),
                            
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_sectors, use_container_width=True)
                    else:
                        st.info("Sector data not available")
                except Exception as e:
                    st.info("Sector data not available")
        
        with detail_tabs[2]:
            st.markdown("### Portfolio Analytics")
            
            try:
                full_data = yahoo.retrieve_data(tuple(assets))
                port = Portfolio(assets, full_data)
                port.set_weights(weights)
                
                # Charger le benchmark
                benchmark = None
                try:
                    benchmark = create_benchmark("^GSPC", period="5y")
                except Exception as e:
                    st.warning("Could not load benchmark for comparison")
                
                # Afficher les métriques avancées avec le nouveau design
                render_advanced_metrics_section(port, benchmark)
                
            except Exception as e:
                st.error(f"Could not calculate advanced metrics: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
        
        with detail_tabs[3]:
            st.markdown("### Portfolio Information")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                        {portfolio.get('model', 'N/A').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Number of Assets</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                        {len(assets)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Created</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                        {portfolio.get('created_at', 'Unknown')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with info_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Method</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                        {portfolio.get('method', 'N/A').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Weight</div>
                    <div style="font-size: 18px; font-weight: 600; margin-top: 8px;">
                        {sum(weights):.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Assets</div>
                    <div style="font-size: 14px; margin-top: 8px;">
                        {', '.join(assets)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading portfolio details: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
            
            
            
            
            
            
            
            
            



