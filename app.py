"""
FinanceAI - Stunning Modern Interface with Full Stock Explorer
Greek Phi (Φ) branding, collapsible sidebar, glassmorphism
Using Plotly for all charts (better performance & interactivity)
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

# CSS (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    :root {
        --bg-primary: #0B0F19;
        --bg-card: rgba(26, 31, 46, 0.7);
        --accent-primary: #6366F1;
        --accent-secondary: #8B5CF6;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --border: rgba(148, 163, 184, 0.12);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #131720 50%, #1A1F2E 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(19, 23, 32, 0.95) 0%, rgba(26, 31, 46, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }
    
    .phi-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 24px 0;
        margin-bottom: 32px;
        border-bottom: 1px solid var(--border);
    }
    
    .phi-logo {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3));
    }
    
    .phi-title {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 28px;
        margin: 16px 0;
        transition: all 0.4s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-6px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 16px 48px rgba(99, 102, 241, 0.2);
    }
    
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.6);
    }
    
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background: rgba(26, 31, 46, 0.6);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text-primary);
        padding: 12px 16px;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text-primary);
        background: rgba(99, 102, 241, 0.1);
        border-bottom: 2px solid var(--accent-primary);
    }
    
    h1 {
        font-size: 48px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #131720; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'tab_data' not in st.session_state:
    st.session_state.tab_data = None

# Helper functions
def get_indicator(stock: Stock, indicator):
    data = stock.data
    if indicator == "MA 50":
        return moving_average(data, 50)
    if indicator == "MA 200":
        return moving_average(data, 200)
    if indicator == "RSI":
        return rsi(data)
    if indicator == "Volume":
        return data["Volume"]
    if indicator == "LRC":
        return lrc(data)

def create_candlestick_chart(data, indicators=[]):
    """Create beautiful Plotly candlestick chart with indicators"""
    
    # Handle MultiIndex columns - flatten them
    if isinstance(data.columns, pd.MultiIndex):
        # Get the ticker symbol (second level)
        ticker = data.columns[0][1]
        # Select just that ticker's data and flatten column names
        data = data.xs(ticker, level=1, axis=1)
        print(f"Flattened MultiIndex for {ticker}")
    
    print(f"Cleaned columns: {data.columns.tolist()}")
    print(f"Data shape: {data.shape}")
    
    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required):
        st.error(f"Missing columns. Have: {data.columns.tolist()}")
        return None
    
    # Remove NaN rows
    data_clean = data.dropna(subset=required)
    
    if data_clean.empty:
        st.error("No valid data")
        return None
    
    # Determine subplots
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
        row_heights=row_heights,
        subplot_titles=(['Price'] + (['Volume'] if has_volume else []) + (['RSI'] if has_rsi else []))
    )
    
    # Main candlestick
    fig.add_trace(
        go.Candlestick(
            x=data_clean.index,
            open=data_clean['Open'],
            high=data_clean['High'],
            low=data_clean['Low'],
            close=data_clean['Close'],
            name='Price',
            increasing_line_color='#10B981',
            decreasing_line_color='#EF4444'
        ),
        row=1, col=1
    )
    
    current_row = 1
    
    # Add indicators
    if "MA 50" in indicators:
        try:
            ma50 = moving_average(data_clean, 50)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=ma50, name='MA 50', 
                          line=dict(color='#8B5CF6', width=1.5)),
                row=1, col=1
            )
        except Exception as e:
            print(f"MA 50 error: {e}")
    
    if "MA 200" in indicators:
        try:
            ma200 = moving_average(data_clean, 200)
            fig.add_trace(
                go.Scatter(x=data_clean.index, y=ma200, name='MA 200',
                          line=dict(color='#EC4899', width=1.5)),
                row=1, col=1
            )
        except Exception as e:
            print(f"MA 200 error: {e}")
    
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
        except Exception as e:
            print(f"LRC error: {e}")
    
    # Volume
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
        except Exception as e:
            print(f"Volume error: {e}")
    
    # RSI
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
        except Exception as e:
            print(f"RSI error: {e}")
    
    # Styling
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 31, 46, 0.5)',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family='Inter', color='#F8FAFC'),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=False, linecolor='rgba(148, 163, 184, 0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
    
    return fig
def dask_read_json(file):
    return dd.read_json(file, blocksize=None, orient="records", lines=False).compute()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="phi-header">
        <div class="phi-logo">Φ</div>
        <div class="phi-title">Manager</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("", ["Dashboard", "Portfolio Manager", "Stock Explorer", "Stock Screener", "AI Assistant"])
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
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

# DASHBOARD
if page == "Dashboard":
    st.markdown("<h1>Welcome to Φ FinanceAI</h1>", unsafe_allow_html=True)
    
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
    
    st.markdown("### Recent Portfolios")
    try:
        for pf in portfolios[:3]:
            with st.expander(f"📁 {pf['name']}"):
                col1, col2 = st.columns(2)
                col1.write(f"**Model:** {pf.get('model')}")
                col2.write(f"**Value:** ${pf.get('amount'):,.2f}")
    except:
        st.info("No portfolios yet!")

# Add this to your app.py - replace the "My Portfolios" tab content in Portfolio Manager

elif page == "Portfolio Manager":
   st.markdown("<h1>Portfolio Manager</h1>", unsafe_allow_html=True)
    
   tab1, tab2, tab3 = st.tabs(["Build Portfolio", "My Portfolios", "Portfolio Details"])
    
    # ... (keep existing Build tab code) ...
   with tab1:
        with st.form("build"):
            col1, col2 = st.columns(2)
            tickers = col1.text_input("Tickers", "AAPL, MSFT, GOOGL")
            model = col2.selectbox("Model", ["markowitz", "naive", "betaweighted"])
            
            if model == "markowitz":
                method = st.selectbox("Method", ["sharp", "risk", "return"])
                if method == "risk":
                    risk_tol = st.slider("Risk Tolerance", 0.0, 100.0, 20.0)
                elif method == "return":
                    exp_ret = st.number_input("Expected Return", 0.0, 1.0, 0.25)
            
            col3, col4 = st.columns(2)
            name = col3.text_input("Name")
            amount = col4.number_input("Amount ($)", 1000.0, value=10000.0)
            
            submit = st.form_submit_button("Build", use_container_width=True)
        
        if submit and tickers and name:
            with st.spinner("Building..."):
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
                    
                    st.success("Built!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Return", f"{portfolio.expected_return:.2%}")
                    col2.metric("Volatility", f"{portfolio.stdev:.2%}")
                    col3.metric("Sharpe", f"{portfolio.sharp_ratio:.2f}")
                    
                    df = pd.DataFrame({
                        'Asset': assets,
                        'Weight': [f"{w:.2%}" for w in portfolio.weights]
                    })
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("Save"):
                        save_portfolio(portfolio, name, model=model, amount=amount)
                        st.success("Saved!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
   with tab2:
        st.markdown("### My Portfolios")
        
        try:
            portfolios = list(get_portfolios())
            
            if not portfolios:
                st.info("No portfolios saved yet. Create one in the 'Build Portfolio' tab!")
            else:
                # Portfolio cards grid
                cols = st.columns(2)
                
                for idx, pf in enumerate(portfolios):
                    with cols[idx % 2]:
                        # Calculate current value
                        try:
                            assets = pf['assets']
                            data = yahoo.retrieve_data(tuple(assets), "1d")
                            
                            if not data.empty and len(data) > 0:
                                # Handle MultiIndex
                                if isinstance(data.columns, pd.MultiIndex):
                                    latest_prices = data["Adj Close"].iloc[-1]
                                else:
                                    latest_prices = data["Adj Close"].iloc[-1]
                                
                                quantities = pf.get('quantities', [])
                                if quantities:
                                    current_value = sum(
                                        q * latest_prices[i] 
                                        for i, q in enumerate(quantities)
                                    )
                                    initial = pf['amount']
                                    pnl = current_value - initial
                                    pnl_pct = (pnl / initial) * 100
                                else:
                                    current_value = pf['amount']
                                    pnl = 0
                                    pnl_pct = 0
                            else:
                                current_value = pf['amount']
                                pnl = 0
                                pnl_pct = 0
                        except:
                            current_value = pf['amount']
                            pnl = 0
                            pnl_pct = 0
                        
                        # Portfolio card
                        st.markdown(f"""
                        <div class="glass-card">
                            <h3 style="margin: 0 0 16px 0; color: #F8FAFC;">{pf['name']}</h3>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <span style="color: #94A3B8;">Model</span>
                                <span style="color: #F8FAFC; font-weight: 600;">{pf.get('model', 'N/A').title()}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <span style="color: #94A3B8;">Method</span>
                                <span style="color: #F8FAFC; font-weight: 600;">{pf.get('method', 'N/A').title()}</span>
                            </div>
                            <div style="border-top: 1px solid rgba(148, 163, 184, 0.12); padding-top: 12px; margin-top: 12px;">
                                <div style="font-size: 13px; color: #94A3B8; margin-bottom: 4px;">CURRENT VALUE</div>
                                <div style="font-size: 28px; font-weight: 700; color: #6366F1;">${current_value:,.2f}</div>
                                <div style="margin-top: 8px; font-size: 14px; font-weight: 600; color: {'#10B981' if pnl >= 0 else '#EF4444'};">
                                    {pnl:+.2f} ({pnl_pct:+.2f}%)
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"View Details →", key=f"view_{pf['name']}", use_container_width=True):
                            st.session_state.selected_portfolio = pf['name']
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error loading portfolios: {e}")
    
   with tab3:
    if 'selected_portfolio' not in st.session_state:
        st.info("Select a portfolio from 'My Portfolios' to view details")
    else:
        portfolio_name = st.session_state.selected_portfolio
        
        try:
            portfolio = get_single_portfolio(portfolio_name)
            assets = portfolio["assets"]
            weights = portfolio["weights"]
            amount = portfolio["amount"]
            
            # Header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<h1>{portfolio_name}</h1>", unsafe_allow_html=True)
            with col2:
                if st.button("← Back", use_container_width=True):
                    del st.session_state.selected_portfolio
                    st.rerun()
            
            # Fetch current data
            data = yahoo.retrieve_data(tuple(assets), "1d")
            
            # Handle MultiIndex properly
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
            
            # Calculate metrics
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
            
            # Metrics row
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
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value" style="color: {'#10B981' if pnl >= 0 else '#EF4444'};">${pnl:+,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Return</div>
                    <div class="metric-value" style="color: {'#10B981' if pnl_pct >= 0 else '#EF4444'};">{pnl_pct:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Sub-tabs for different views
            detail_tabs = st.tabs(["Holdings", "Composition", "Performance", "Metrics"])
            
            # TAB 1: HOLDINGS
            with detail_tabs[0]:
                st.markdown("### Holdings")
                
                display_df = df.copy()
                display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x*100:.2f}%")
                display_df["Initial Amount"] = display_df["Initial Amount"].apply(lambda x: f"${x:,.2f}")
                display_df["Current Price"] = display_df["Current Price"].apply(lambda x: f"${x:.2f}")
                display_df["Market Value"] = display_df["Market Value"].apply(lambda x: f"${x:,.2f}")
                display_df["P&L"] = [(mv - ia) for mv, ia in zip(df["Market Value"], df["Initial Amount"])]
                display_df["P&L"] = display_df["P&L"].apply(lambda x: f"${x:+,.2f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Export button
                if st.button("📥 Export Holdings to CSV", use_container_width=True):
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{portfolio_name}_holdings.csv",
                        mime="text/csv"
                    )
            
            # TAB 2: COMPOSITION
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
                
                # Weights breakdown table
                st.markdown("### Weight Distribution")
                weight_df = pd.DataFrame({
                    'Asset': assets,
                    'Weight': [f"{w*100:.2f}%" for w in weights],
                    'Amount': [f"${amount * w:,.2f}" for w in weights]
                })
                st.dataframe(weight_df, use_container_width=True, hide_index=True)
            
            # TAB 3: PERFORMANCE
            with detail_tabs[2]:
                st.markdown("### Performance Analysis")
                
                # Time period selector
                perf_period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
                
                try:
                    # Get historical data
                    hist_data = yahoo.retrieve_data(tuple(assets), perf_period)
                    
                    # Calculate portfolio value over time
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        # Get close prices for each asset
                        portfolio_values = []
                        
                        for date in hist_data.index:
                            daily_value = 0
                            for i, asset in enumerate(assets):
                                price = hist_data[("Adj Close", asset)].loc[date]
                                qty = quantities[i] if i < len(quantities) else 0
                                daily_value += price * qty
                            portfolio_values.append(daily_value)
                        
                        portfolio_series = pd.Series(portfolio_values, index=hist_data.index)
                    else:
                        # Single asset
                        portfolio_series = hist_data["Adj Close"] * quantities[0]
                    
                    # Create performance chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=portfolio_series.index,
                        y=portfolio_series.values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#6366F1', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(99, 102, 241, 0.1)'
                    ))
                    
                    # Add initial investment line
                    fig.add_hline(
                        y=amount,
                        line_dash="dash",
                        line_color="#94A3B8",
                        annotation_text="Initial Investment",
                        annotation_position="right"
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(26, 31, 46, 0.5)',
                        title=dict(text="Portfolio Value Over Time", font=dict(size=20, color='#F8FAFC')),
                        font=dict(family='Inter', color='#F8FAFC'),
                        height=500,
                        xaxis=dict(showgrid=False, title="Date"),
                        yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', title="Value ($)"),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    returns = portfolio_series.pct_change().dropna()
                    
                    with col1:
                        total_return = ((portfolio_series.iloc[-1] - amount) / amount) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Return</div>
                            <div style="font-size: 24px; font-weight: 700; color: {'#10B981' if total_return >= 0 else '#EF4444'}; margin-top: 8px;">
                                {total_return:+.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        best_day = returns.max() * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Best Day</div>
                            <div style="font-size: 24px; font-weight: 700; color: #10B981; margin-top: 8px;">
                                +{best_day:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        worst_day = returns.min() * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Worst Day</div>
                            <div style="font-size: 24px; font-weight: 700; color: #EF4444; margin-top: 8px;">
                                {worst_day:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        volatility = returns.std() * np.sqrt(252) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Volatility (Annual)</div>
                            <div style="font-size: 24px; font-weight: 700; color: #F8FAFC; margin-top: 8px;">
                                {volatility:.2f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Could not calculate performance: {e}")
                    st.info("Performance tracking requires historical data. Ensure your portfolio has quantities saved.")
            
            # TAB 4: METRICS
            with detail_tabs[3]:
                st.markdown("### Portfolio Metrics")
                
                try:
                    # Get full data for portfolio object
                    full_data = yahoo.retrieve_data(tuple(assets))
                    
                    # Create portfolio
                    port = Portfolio(assets, full_data)
                    port.set_weights(weights)
                    
                    # Calculate metrics manually to avoid summary() issues
                    metrics_data = {
                        "Expected Return": port.expected_return,
                        "Volatility (Std Dev)": port.stdev,
                        "Sharpe Ratio": port.sharp_ratio,
                        "Variance": port.variance,
                        "Skewness": port.skewness,
                        "Kurtosis": port.kurtosis,
                        "VaR (95%)": port.VAR(conf_level=0.05),
                    }
                    
                    # Try benchmark metrics
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
                                color = "#F8FAFC"
                            
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
    
    
    
# STOCK EXPLORER (Using Plotly)
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

# STOCK SCREENER
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
