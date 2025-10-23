"""
ΦManager - Modern Portfolio & Market Intelligence Platform
Collapsible icon sidebar, dark/light mode, enhanced UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import your modules
from portfolio import Portfolio
from factory import create_portfolio_by_name
from dataprovider import yahoo
from stock import Stock
from database import get_portfolios, get_single_portfolio, save_portfolio
from describe import summary
from ta import lrc, moving_average, rsi
from index import Index
import dask.dataframe as dd


st.set_page_config(
    page_title="ΦManager",
    page_icon="Φ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def init_session_state():
    """Initialise tous les états de session"""
    defaults = {
        'chat_history': [],
        'tab_data': None,
        'theme': 'dark',
        'current_page': 'Dashboard'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
            


init_session_state()

def get_theme_colors():
    """Retourne les couleurs selon le thème actif"""
    if st.session_state.theme == "dark":
        return {
            'bg_primary': '#0B0F19',
            'bg_secondary': '#131720',
            'bg_card': 'rgba(26, 31, 46, 0.7)',
            'text_primary': '#F8FAFC',
            'text_secondary': '#94A3B8',
            'border': 'rgba(148, 163, 184, 0.12)',
            'plotly_template': 'plotly_dark',
            'plot_bg': 'rgba(26, 31, 46, 0.5)'
        }
    else:
        return {
            'bg_primary': '#F8FAFC',
            'bg_secondary': '#F1F5F9',
            'bg_card': 'rgba(255, 255, 255, 0.8)',
            'text_primary': '#1E293B',
            'text_secondary': '#64748B',
            'border': 'rgba(100, 116, 139, 0.2)',
            'plotly_template': 'plotly',
            'plot_bg': 'rgba(255, 255, 255, 0.5)'
        }

theme = get_theme_colors()


def apply_custom_css(theme):
    """Applique les styles CSS personnalisés"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{ font-family: 'Inter', sans-serif; }}
        
        .stApp {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%);
        }}
        
        [data-testid="stSidebar"] {{
            background: {theme['bg_card']};
            backdrop-filter: blur(20px);
            border-right: 1px solid {theme['border']};
        }}
        
        .phi-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 20px 0;
            margin-bottom: 24px;
            border-bottom: 1px solid {theme['border']};
        }}
        
        .phi-logo {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .phi-title {{
            font-size: 20px;
            font-weight: 700;
            color: {theme['text_primary']};
        }}
        
        .glass-card {{
            background: {theme['bg_card']};
            backdrop-filter: blur(20px);
            border: 1px solid {theme['border']};
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            transition: all 0.3s ease;
        }}
        
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
        }}
        
        .metric-card {{
            background: {theme['bg_card']};
            backdrop-filter: blur(20px);
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        
        .metric-label {{
            font-size: 12px;
            font-weight: 500;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .positive {{ color: #10B981 !important; font-weight: 600; }}
        .negative {{ color: #EF4444 !important; font-weight: 600; }}
        
        .stButton > button {{
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }}
        
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            color: {theme['text_primary']};
            padding: 10px 14px;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {{
            border-color: #6366F1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {theme['border']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            color: {theme['text_secondary']};
            padding: 10px 20px;
            font-weight: 600;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {theme['text_primary']};
            background: rgba(99, 102, 241, 0.1);
            border-bottom: 2px solid #6366F1;
        }}
        
        h1 {{
            font-size: 42px !important;
            font-weight: 700 !important;
            color: {theme['text_primary']} !important;
            margin-bottom: 20px !important;
        }}
        
        h3 {{
            color: {theme['text_primary']} !important;
            font-weight: 600 !important;
        }}
        
        .stDataFrame {{
            background: {theme['bg_card']};
            border-radius: 8px;
        }}
        
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {theme['bg_secondary']}; }}
        ::-webkit-scrollbar-thumb {{ 
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            border-radius: 4px;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_css(theme)





# ==================== FONCTIONS UTILITAIRES ====================
def format_pnl(value, pct):
    """Formate P&L avec couleur"""
    color_class = "positive" if value >= 0 else "negative"
    return f'<span class="{color_class}">${value:+,.2f} ({pct:+.2f}%)</span>'

def get_indicator(stock: Stock, indicator):
    """Récupère les indicateurs techniques"""
    data = stock.data
    indicators_map = {
        "MA 50": lambda: moving_average(data, 50),
        "MA 200": lambda: moving_average(data, 200),
        "RSI": lambda: rsi(data),
        "Volume": lambda: data["Volume"],
        "LRC": lambda: lrc(data)
    }
    return indicators_map.get(indicator, lambda: None)()

def dask_read_json(file):
    """Lit un fichier JSON avec Dask"""
    return dd.read_json(file, blocksize=None, orient="records", lines=False).compute()

# ==================== CREATION GRAPHIQUES ====================
def create_candlestick_chart(data, indicators=[]):
    """Crée un graphique en chandelier avec indicateurs"""
    # Flatten MultiIndex si nécessaire
    if isinstance(data.columns, pd.MultiIndex):
        ticker = data.columns[0][1]
        data = data.xs(ticker, level=1, axis=1)
    
    # Vérifier colonnes requises
    required = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required):
        return None
    
    data_clean = data.dropna(subset=required)
    if data_clean.empty:
        return None
    
    # Configuration subplots
    has_volume = "Volume" in indicators and 'Volume' in data_clean.columns
    has_rsi = "RSI" in indicators
    
    rows = 1
    row_heights = [0.7]
    if has_volume:
        rows += 1
        row_heights.append(0.15)
    if has_rsi:
        rows += 1
        row_heights.append(0.15)
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # Chandelier principal
    fig.add_trace(
        go.Candlestick(
            x=data_clean.index,
            open=data_clean['Open'],
            high=data_clean['High'],
            low=data_clean['Low'],
            close=data_clean['Close'],
            increasing_line_color='#10B981',
            decreasing_line_color='#EF4444'
        ),
        row=1, col=1
    )
    
    current_row = 1
    
    # Ajout MA 50
    if "MA 50" in indicators:
        try:
            ma50 = moving_average(data_clean, 50)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=ma50, name='MA 50', 
                          line=dict(color='#8B5CF6', width=1.5)),
                row=1, col=1
            )
        except:
            pass
    
    # Ajout MA 200
    if "MA 200" in indicators:
        try:
            ma200 = moving_average(data_clean, 200)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=ma200, name='MA 200',
                          line=dict(color='#EC4899', width=1.5)),
                row=1, col=1
            )
        except:
            pass
    
    # Ajout LRC
    if "LRC" in indicators:
        try:
            lrc_data = lrc(data_clean)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=lrc_data["high_trend"], name='High Trend',
                          line=dict(color='#10B981', width=1.5, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=lrc_data["low_trend"], name='Low Trend',
                          line=dict(color='#EF4444', width=1.5, dash='dash')),
                row=1, col=1
            )
        except:
            pass
    
    # Ajout Volume
    if has_volume:
        current_row += 1
        try:
            colors = ['#10B981' if c >= o else '#EF4444' 
                     for c, o in zip(data_clean['Close'], data_clean['Open'])]
            fig.add_trace(
                go.Bar(x=data_clean.index, y=data_clean['Volume'], name='Volume',
                      marker_color=colors, opacity=0.5),
                row=current_row, col=1
            )
        except:
            pass
    
    # Ajout RSI
    if has_rsi:
        current_row += 1
        try:
            rsi_data = rsi(data_clean)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=rsi_data, name='RSI',
                          line=dict(color='#6366F1', width=2)),
                row=current_row, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", 
                         opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#10B981",
                         opacity=0.5, row=current_row, col=1)
        except:
            pass
    
    # Mise en forme
    fig.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        height=600,
        showlegend=True,
        font=dict(family='Inter', color=theme['text_primary']),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
    
    return fig

def calculate_portfolio_current_value(portfolio_data):
    """Calcule la valeur actuelle d'un portfolio"""
    try:
        assets = portfolio_data['assets']
        data = yahoo.retrieve_data(tuple(assets), "1d")
        
        if isinstance(data.columns, pd.MultiIndex):
            latest_prices = data["Adj Close"].iloc[-1]
        else:
            latest_prices = data["Adj Close"].iloc[-1]
        
        quantities = portfolio_data.get('quantities', [])
        if quantities:
            current_value = sum(q * latest_prices[i] for i, q in enumerate(quantities))
            pnl = current_value - portfolio_data['amount']
            pnl_pct = (pnl / portfolio_data['amount']) * 100
            return current_value, pnl, pnl_pct
    except:
        pass
    
    return portfolio_data['amount'], 0, 0

# ==================== SIDEBAR ====================
def render_sidebar():
    """Affiche la sidebar avec navigation"""
    with st.sidebar:
        st.markdown(f"""
        <div class="phi-header">
            <div class="phi-logo">Φ</div>
            <div class="phi-title">Manager</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Toggle theme
        theme_emoji = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(f"{theme_emoji} Theme", key="theme_toggle", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        nav_items = [
            ("Dashboard", "🏠"),
            ("Portfolio Manager", "📊"),
            ("Stock Explorer", "🔍"),
            ("Stock Screener", "🎯"),
            ("AI Assistant", "🤖")
        ]
        
        for page_name, icon in nav_items:
            if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()
        
        st.markdown("---")
        
        # Stats rapides
        try:
            portfolios = list(get_portfolios())
            total = sum([p.get('amount', 0) for p in portfolios])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Value</div>
                <div class="metric-value">${total:,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Portfolios</div>
                <div class="metric-value">{len(portfolios)}</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

render_sidebar()
page = st.session_state.current_page


# ==================== PAGES ====================
def render_dashboard():
    """Page Dashboard"""
    st.markdown("<h1>Welcome to Φ Manager</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        portfolios = list(get_portfolios())
        total = sum([p.get('amount', 0) for p in portfolios])
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Total Portfolio</div>
                <div class="metric-value">${total:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Portfolios</div>
                <div class="metric-value">{len(portfolios)}</div>
            </div>
            """, unsafe_allow_html=True)
    except:
        pass
    
    st.markdown("---")
    st.markdown("### Recent Portfolios")
    
    try:
        for pf in portfolios[:5]:
            with st.expander(f"📁 {pf['name']}"):
                col1, col2 = st.columns(2)
                col1.write(f"**Model:** {pf.get('model', 'N/A').title()}")
                col2.write(f"**Value:** ${pf.get('amount', 0):,.2f}")
    except:
        st.info("No portfolios yet. Create one in Portfolio Manager!")

def render_portfolio_build_tab():
    """Onglet construction de portfolio"""
    st.markdown("### Build New Portfolio")
    
    with st.form("build_portfolio"):
        col1, col2 = st.columns(2)
        tickers = col1.text_input("Stock Tickers", "AAPL, MSFT, GOOGL", help="Comma-separated")
        model = col2.selectbox("Model", ["markowitz", "naive", "betaweighted"])
        
        method, risk_tol, exp_ret = None, None, None
        
        if model == "markowitz":
            method = st.selectbox("Method", ["sharp", "risk", "return"])
            if method == "risk":
                risk_tol = st.slider("Risk Tolerance", 0.0, 100.0, 20.0)
            elif method == "return":
                exp_ret = st.number_input("Expected Return", 0.0, 1.0, 0.25)
        
        col3, col4 = st.columns(2)
        name = col3.text_input("Portfolio Name")
        amount = col4.number_input("Initial Amount ($)", 1000.0, value=10000.0)
        
        submit = st.form_submit_button("Build Portfolio", use_container_width=True)
    
    if submit and tickers and name:
        with st.spinner("Building portfolio..."):
            try:
                assets = [t.strip().upper() for t in tickers.split(",")]
                data = yahoo.retrieve_data(tuple(assets))
                
                if model == "markowitz":
                    if method == "risk":
                        portfolio = create_portfolio_by_name(assets, "risk", data, risk_tolerance=risk_tol)
                    elif method == "return":
                        portfolio = create_portfolio_by_name(assets, "return", data, expected_return=exp_ret)
                    else:
                        portfolio = create_portfolio_by_name(assets, method, data)
                else:
                    portfolio = Portfolio(assets, data)
                    n = len(assets)
                    portfolio.set_weights([1/n] * n)
                
                st.success("Portfolio built successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Return", f"{portfolio.expected_return:.2%}")
                col2.metric("Volatility", f"{portfolio.stdev:.2%}")
                col3.metric("Sharpe Ratio", f"{portfolio.sharp_ratio:.2f}")
                
                weights_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [f"{w:.2%}" for w in portfolio.weights]
                })
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
                
                if st.button("Save Portfolio", use_container_width=True):
                    save_portfolio(portfolio, name, model=model, amount=amount)
                    st.success(f"Portfolio '{name}' saved!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error: {e}")

def render_portfolio_list_tab():
    """Onglet liste des portfolios"""
    st.markdown("### My Portfolios")
    
    try:
        portfolios = list(get_portfolios())
        
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
    """Onglet détails portfolio"""
    st.markdown("### Portfolio Details")
    
    try:
        portfolios = list(get_portfolios())
        portfolio_names = [p['name'] for p in portfolios]
        
        if not portfolio_names:
            st.info("No portfolios available. Create one first!")
            return
        
        selected = st.selectbox("Select Portfolio", portfolio_names, key="portfolio_selector")
        
        if not selected:
            return
        
        portfolio = get_single_portfolio(selected)
        assets = portfolio["assets"]
        weights = portfolio["weights"]
        amount = portfolio["amount"]
        
        # Récupérer données actuelles
        data = yahoo.retrieve_data(tuple(assets), "1d")
        
        if isinstance(data.columns, pd.MultiIndex):
            latest_prices = {}
            for asset in assets:
                try:
                    latest_prices[asset] = data[("Adj Close", asset)].iloc[-1]
                except:
                    latest_prices[asset] = 0
        else:
            if len(assets) == 1:
                latest_prices = {assets[0]: data["Adj Close"].iloc[-1]}
            else:
                latest_prices = {asset: data["Adj Close"][asset].iloc[-1] for asset in assets}
        
        quantities = portfolio.get('quantities', [0] * len(assets))
        portfolio_comp = []
        
        for i, (asset, weight) in enumerate(zip(assets, weights)):
            qty = quantities[i] if i < len(quantities) else 0
            current_price = latest_prices.get(asset, 0)
            initial_amt = amount * weight
            market_val = current_price * qty
            
            portfolio_comp.append({
                "Asset": asset,
                "Weight": weight,
                "Initial Amount": initial_amt,
                "Quantity": qty,
                "Current Price": current_price,
                "Market Value": market_val
            })
        
        df = pd.DataFrame(portfolio_comp)
        mtm = df["Market Value"].sum()
        pnl = mtm - amount
        pnl_pct = (pnl / amount) * 100 if amount > 0 else 0
        
        # Métriques
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
        
        # Sous-onglets
        detail_tabs = st.tabs(["Holdings", "Composition", "Performance", "Metrics"])
        
        with detail_tabs[0]:
            st.markdown("### Holdings")
            
            display_df = df.copy()
            display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x*100:.2f}%")
            display_df["Initial Amount"] = display_df["Initial Amount"].apply(lambda x: f"${x:,.2f}")
            display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"${x:.2f}")
            display_df["Market Value"] = display_df["Market Value"].apply(lambda x: f"${x:,.2f}")
            
            # Ajouter P&L avec couleur
            pnl_values = [(mv - ia) for mv, ia in zip(df["Market Value"], df["Initial Amount"])]
            display_df["P&L"] = [f"${x:+,.2f}" for x in pnl_values]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
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
                        paper_bgcolor='rgba(0,0,0,0)',
                        title=dict(text="Asset Allocation", font=dict(size=18, color='#F8FAFC')),
                        font=dict(family='Inter', color='#F8FAFC'),
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
                                paper_bgcolor='rgba(0,0,0,0)',
                                title=dict(text="Sector Allocation", font=dict(size=18, color='#F8FAFC')),
                                font=dict(family='Inter', color='#F8FAFC'),
                                height=400,
                                showlegend=True
                            )
                            st.plotly_chart(fig_sectors, use_container_width=True)
                        else:
                            st.info("Sector data not available")
                    except Exception as e:
                        st.info("Sector data not available")
        
        with detail_tabs[2]:
            st.markdown("### Performance")
            st.info("Performance tracking requires historical quantities data")
        
        with detail_tabs[3]:
            st.markdown("### Metrics")
            
            try:
                full_data = yahoo.retrieve_data(tuple(assets))
                port = Portfolio(assets, full_data)
                port.set_weights(weights)
                
                metrics_data = {
                        "Expected Return": port.expected_return,
                        "Volatility (Std Dev)": port.stdev,
                        "Sharpe Ratio": port.sharp_ratio,
                        "Variance": port.variance,
                        "Skewness": port.skewness,
                        "Kurtosis": port.kurtosis,
                        "VaR (95%)": port.VAR(conf_level=0.05),
                    }
                
                try:
                        from factory import create_benchmark
                        benchmark = create_benchmark("SPY", period="1y")
                        metrics_data["Alpha"] = port.alpha(benchmark)
                        metrics_data["Beta"] = port.beta(benchmark)
                        metrics_data["Treynor Ratio"] = port.treynor_ratio(benchmark)
                except:
                        pass
                    
                    # Display in grid
                cols = st.columns(2)
                    
                for i, (key, value) in enumerate(metrics_data.items()):
                        with cols[i % 2]:
                            # Color coding for certain metrics
                            if key == "Sharpe Ratio":
                                color = "#10B981" if value > 1 else "#F59E0B" if value > 0 else "#EF4444"
                            elif key == "Alpha":
                                color = "#10B981" if value > 0 else "#EF4444"
                            else:
                                color = "#F8FAFC" if  st.session_state.theme == "dark"   else "black"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{key}</div>
                                <div style="font-size: 20px; font-weight: 600; color: {color}; margin-top: 8px;">
                                    {value:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Additional info
                st.markdown("---")
                st.markdown("### Portfolio Information")
                    
                info_col1, info_col2 = st.columns(2)
                    
                with info_col1:
                        st.write(f"**Model:** {portfolio.get('model', 'N/A').title()}")
                        st.write(f"**Method:** {portfolio.get('method', 'N/A').title()}")
                        st.write(f"**Number of Assets:** {len(assets)}")
                    
                with info_col2:
                        st.write(f"**Total Weight:** {sum(weights):.4f}")
                        st.write(f"**Created:** {portfolio.get('created_at', 'Unknown')}")
                
            except Exception as e:
                    st.error(f"Could not calculate advanced metrics: {e}")
                    
                    # Fallback: show basic info
                    st.markdown("### Basic Information")
                    st.write(f"**Model:** {portfolio.get('model', 'N/A')}")
                    st.write(f"**Method:** {portfolio.get('method', 'N/A')}")
                    st.write(f"**Assets:** {', '.join(assets)}")
                    st.write(f"**Weights:** {', '.join([f'{w:.2%}' for w in weights])}")
        
    except Exception as e:
            st.error(f"Error loading portfolio details: {e}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())
    



# DASHBOARD
if page == "Dashboard":
       render_dashboard()
# PORTFOLIO MANAGER
elif page == "Portfolio Manager":
    st.markdown("<h1>Portfolio Manager</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Build Portfolio", "My Portfolios", "Portfolio Details"])
    
    with tab1:
         render_portfolio_build_tab()
        
    
    with tab2:
         render_portfolio_list_tab()
         
    with tab3:
         render_portfolio_details_tab()
         
#stock explorer
elif page == "Stock Explorer":
    st.markdown("<h1>Stock Explorer</h1>", unsafe_allow_html=True)
    
    ticker = st.text_input("Enter Ticker", placeholder="AAPL")
    
    if ticker:
        try:
            stock = Stock(ticker.upper())
            stock.get_data("1y")
            
            if not stock.data.empty:
                close =[p[0] for p in stock.data["Close"].values]
                #print(close)
                value = close[-1]
                if len(close) > 1:
                    pch = (close[-1] - close[-2]) / close[-2] * 100
                    delta = close[-1] - close[-2]
                else:
                    pch, delta = 0, 0
                
                # Header
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">{ticker.upper()}</div>
                        <div class="metric-value">${value:.2f}</div>
                        <div style="color: {'#10B981' if pch > 0 else '#EF4444'}; font-weight: 600; margin-top: 8px;">
                            {pch:+.2f}% (${delta:+.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if stock.infos:
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sector</div>
                            <div style="font-size: 16px; color: #F8FAFC; margin-top: 8px;">
                                {stock.get_sector()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tabs
            tabs = st.tabs(["Chart", "Info", "Financials", "Dividends", "News"])
            
            # CHART TAB
            with tabs[0]:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
                    indicators = st.multiselect("Indicators", ["MA 50", "MA 200", "RSI", "Volume", "LRC"])
                
                stock.get_data(period)
                data = stock.data

                st.write("Debug - Data structure:")
                st.write(f"Columns: {data.columns.tolist()}")
                st.write(f"Index: {data.index[:5]}")
                st.dataframe(data.head())  # Show first few rows

                # Create chart
                fig = create_candlestick_chart(data, indicators)
                if fig:
                   st.plotly_chart(fig, use_container_width=True)
                
               
            
            # INFO TAB
            with tabs[1]:
                if stock.infos:
                    st.write(stock.infos.get('longBusinessSummary', 'N/A'))
                else:
                    st.info("Info not available")
            
            # FINANCIALS TAB
            with tabs[2]:
                try:
                    financials = stock.get_financials()
                    if financials:
                        for key, df in financials.items():
                            st.markdown(f"**{key}**")
                            st.dataframe(df, use_container_width=True)
                except:
                    st.info("Financials not available")
            
            # DIVIDENDS TAB
            with tabs[3]:
                try:
                    divs = stock.get_dividends()
                    if not divs.empty:
                        st.dataframe(divs, use_container_width=True)
                    else:
                        st.info("No dividends")
                except:
                    st.info("Dividends not available")
            
            # NEWS TAB
            with tabs[4]:
                try:
                    news = stock.get_news()
                    for article in news[:10]:
                        st.subheader(article["content"]["title"])
                        st.write(article["content"].get("summary", ""))
                        st.link_button("Read", article["content"]["clickThroughUrl"]["url"])
                        st.divider()
                except:
                    st.info("News not available")
        
        except Exception as e:
            st.error(f"Error: {e}")
elif page == "Stock Screener":
    st.markdown("<h1>Stock Screener</h1>", unsafe_allow_html=True)
    st.info("Screener integration coming soon...")

# AI ASSISTANT
else:
    st.markdown("<h1>AI Assistant</h1>", unsafe_allow_html=True)
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        response = "Add OpenAI API key for full AI features."
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.rerun()
