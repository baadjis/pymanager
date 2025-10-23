# pages/bond_explorer.py
"""
Bond Explorer - Analyse des obligations et fixed income
Taux, ETFs obligataires, yield curve, duration & convexity
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataprovider import yahoo
from uiconfig import get_theme_colors, apply_plotly_theme


BONDS_AND_RATES = {
    "🇺🇸 US Treasury": {
        '^IRX': 'US 3-Month Treasury',
        '^FVX': 'US 5-Year Treasury',
        '^TNX': 'US 10-Year Treasury',
        '^TYX': 'US 30-Year Treasury',
    },
    "📊 Bond ETFs": {
        'AGG': 'iShares Core US Aggregate Bond',
        'BND': 'Vanguard Total Bond Market',
        'TLT': 'iShares 20+ Year Treasury',
        'IEF': 'iShares 7-10 Year Treasury',
        'SHY': 'iShares 1-3 Year Treasury',
        'LQD': 'iShares Investment Grade Corporate',
        'HYG': 'iShares High Yield Corporate',
        'MUB': 'iShares National Muni Bond',
    },
    "🌍 International": {
        'BNDX': 'Vanguard Total International Bond',
        'EMB': 'iShares J.P. Morgan USD Emerging Markets',
        'IGOV': 'iShares International Treasury Bond',
    },
    "💼 Corporate": {
        'VCIT': 'Vanguard Intermediate-Term Corporate',
        'VCSH': 'Vanguard Short-Term Corporate',
        'VCLT': 'Vanguard Long-Term Corporate',
        'USIG': 'iShares Broad USD Investment Grade',
    }
}


def render_bond_explorer():
    """Explorer pour les obligations"""
    theme = get_theme_colors()
    
    st.markdown("### 📈 Bond & Fixed Income Explorer")
    st.caption("Analyze treasury yields, bond ETFs, and fixed income securities")
    
    # Recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "🔍 Search Bond/Rate",
            placeholder="Enter symbol (e.g., ^TNX for 10Y Treasury, AGG for Bond ETF)",
            key="bond_search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Analyze", use_container_width=True, type="primary")
    
    # Suggestions populaires
    if not ticker:
        st.markdown("#### 💡 Popular Bonds & Rates")
        
        for category, bonds in BONDS_AND_RATES.items():
            with st.expander(category):
                cols = st.columns(2)
                for idx, (symbol, name) in enumerate(bonds.items()):
                    with cols[idx % 2]:
                        if st.button(f"{symbol}\n{name}", key=f"bond_{symbol}", use_container_width=True):
                            st.session_state.bond_search = symbol
                            st.rerun()
        return
    
    if ticker and (search_button or ticker):
        ticker = ticker.upper().strip()
        
        try:
            with st.spinner(f"Loading {ticker}..."):
                # Charger données
                data = load_bond_data(ticker, period='1y')
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return
                
                # Header
                render_bond_header(ticker, data, theme)
                
                st.divider()
                
                # Tabs
                tabs = st.tabs([
                    "📈 Chart",
                    "📊 Analysis",
                    "📉 Yield Curve",
                    "🔄 Duration & Metrics"
                ])
                
                with tabs[0]:
                    render_bond_chart_tab(ticker, data, theme)
                
                with tabs[1]:
                    render_bond_analysis_tab(ticker, data, theme)
                
                with tabs[2]:
                    render_yield_curve_tab(theme)
                
                with tabs[3]:
                    render_bond_metrics_tab(ticker, data, theme)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)


def load_bond_data(ticker, period='1y'):
    """
    Charge les données d'obligations avec gestion des erreurs
    """
    try:
        data = yahoo.get_ticker_data(ticker, period=period)
        
        if data is None:
            return None
        
        # Vérifier si les données sont transposées
        if isinstance(data, pd.DataFrame):
            if data.columns.dtype == 'datetime64[ns]' or isinstance(data.columns[0], pd.Timestamp):
                st.info("🔄 Detected transposed data, fixing...")
                data = data.T
            
            # Vérifier les colonnes essentielles
            if 'Close' not in data.columns and len(data.columns) > 0:
                data['Close'] = data.iloc[:, 0]
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def render_bond_header(ticker, data, theme):
    """Header du bond"""
    
    # Trouver le nom
    bond_name = ticker
    for category, bonds in BONDS_AND_RATES.items():
        if ticker in bonds:
            bond_name = bonds[ticker]
            break
    
    current_value = float(data['Close'].iloc[-1])
    
    if len(data) > 1:
        prev_value = float(data['Close'].iloc[-2])
        change = current_value - prev_value
        change_pct = (change / prev_value) * 100 if prev_value != 0 else 0
    else:
        change = 0
        change_pct = 0
    
    # Calculer performances
    perf_1w = calculate_performance(data, 5)
    perf_1m = calculate_performance(data, 21)
    perf_ytd = calculate_ytd_return(data)
    perf_1y = calculate_performance(data, 252)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    is_yield = ticker.startswith('^')
    
    with col1:
        st.markdown(f"### {bond_name}")
        st.caption(ticker)
        
        if is_yield:
            st.metric(
                label="Current Yield",
                value=f"{current_value:.3f}%",
                delta=f"{change:+.3f} bps"
            )
        else:
            st.metric(
                label="Current Price",
                value=f"${current_value:.2f}",
                delta=f"{change_pct:+.2f}%"
            )
    
    with col2:
        st.metric("1 Week", f"{perf_1w:+.2f}%")
    
    with col3:
        st.metric("1 Month", f"{perf_1m:+.2f}%")
    
    with col4:
        st.metric("YTD", f"{perf_ytd:+.2f}%")
    
    with col5:
        st.metric("1 Year", f"{perf_1y:+.2f}%")


def render_bond_chart_tab(ticker, data, theme):
    """Graphique du bond"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            key="bond_period"
        )
        
        show_ma = st.checkbox("Moving Averages", value=True)
        show_volume = st.checkbox("Show Volume", value=False)
    
    # Recharger pour la période
    data = load_bond_data(ticker, period=period)
    
    if data is not None and not data.empty:
        is_yield = ticker.startswith('^')
        line_color = '#EF4444' if is_yield else '#10B981'
        
        # Créer le graphique avec ou sans volume
        if show_volume and 'Volume' in data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Prix/Yield
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=ticker,
                    line=dict(color=line_color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba(239, 68, 68, 0.1)' if is_yield else 'rgba(16, 185, 129, 0.1)'
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(99, 102, 241, 0.3)'
                ),
                row=2, col=1
            )
            
            # Ajouter moyennes mobiles
            if show_ma:
                add_moving_averages_to_subplot(fig, data, 1, 1, theme)
            
            fig.update_yaxes(title_text="Yield (%)" if is_yield else "Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
        else:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=ticker,
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba(239, 68, 68, 0.1)' if is_yield else 'rgba(16, 185, 129, 0.1)'
            ))
            
            # Ajouter moyennes mobiles
            if show_ma:
                add_moving_averages(fig, data, theme)
        
        ylabel = "Yield (%)" if is_yield else "Price ($)"
        
        apply_plotly_theme(
            fig,
            title=f"{ticker} {ylabel} Chart",
            xaxis_title="Date",
            yaxis_title=ylabel,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_bond_analysis_tab(ticker, data, theme):
    """Analyse du bond"""
    
    st.markdown("### 📊 Bond Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_52w = data['High'].tail(252).max() if len(data) >= 252 else data['High'].max()
        st.metric("52W High", f"{high_52w:.3f}")
    
    with col2:
        low_52w = data['Low'].tail(252).min() if len(data) >= 252 else data['Low'].min()
        st.metric("52W Low", f"{low_52w:.3f}")
    
    with col3:
        volatility = data['Close'].pct_change().std() * (252 ** 0.5) * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    with col4:
        current = float(data['Close'].iloc[-1])
        st.metric("Current", f"{current:.3f}")
    
    st.divider()
    
    # Tableau de performance
    st.markdown("#### 📈 Performance Summary")
    
    perf_data = create_performance_table(data)
    df_perf = pd.DataFrame(perf_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_perf, use_container_width=True, hide_index=True)
    
    with col2:
        # Distribution des changements
        st.markdown("#### 📊 Daily Changes Distribution")
        
        returns = data['Close'].pct_change().dropna() * 100
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color='#6366F1',
            name='Returns'
        ))
        
        apply_plotly_theme(
            fig_hist,
            title="",
            xaxis_title="Daily Change (%)",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)


def render_yield_curve_tab(theme):
    """Courbe des taux"""
    
    st.markdown("### 📉 US Treasury Yield Curve")
    
    try:
        # Charger les taux pour différentes maturités
        yields_data = {}
        maturities = {
            '^IRX': ('3M', 0.25),
            '^FVX': ('5Y', 5),
            '^TNX': ('10Y', 10),
            '^TYX': ('30Y', 30)
        }
        
        with st.spinner("Loading yield curve data..."):
            for ticker, (label, maturity) in maturities.items():
                try:
                    data = load_bond_data(ticker, period='5d')
                    if data is not None and not data.empty:
                        current_yield = float(data['Close'].iloc[-1])
                        yields_data[maturity] = {
                            'label': label,
                            'yield': current_yield
                        }
                except:
                    continue
        
        if yields_data:
            # Créer la courbe
            sorted_data = sorted(yields_data.items())
            maturities_list = [x[0] for x in sorted_data]
            labels_list = [x[1]['label'] for x in sorted_data]
            yields_list = [x[1]['yield'] for x in sorted_data]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_curve = go.Figure()
                
                fig_curve.add_trace(go.Scatter(
                    x=maturities_list,
                    y=yields_list,
                    mode='lines+markers',
                    name='Yield Curve',
                    line=dict(color='#6366F1', width=3),
                    marker=dict(size=12, color='#6366F1'),
                    hovertemplate='<b>%{text}</b><br>Yield: %{y:.3f}%<extra></extra>',
                    text=labels_list
                ))
                
                apply_plotly_theme(
                    fig_curve,
                    title="Current Yield Curve",
                    xaxis_title="Maturity (Years)",
                    yaxis_title="Yield (%)",
                    height=400
                )
                
                fig_curve.update_xaxes(
                    tickmode='array',
                    tickvals=maturities_list,
                    ticktext=labels_list
                )
                
                st.plotly_chart(fig_curve, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Current Yields")
                
                for maturity, data_dict in sorted_data:
                    label = data_dict['label']
                    yield_val = data_dict['yield']
                    st.metric(label, f"{yield_val:.3f}%")
                
                # Calculer le spread 10Y-3M (indicateur de récession)
                if 10 in yields_data and 0.25 in yields_data:
                    spread = yields_data[10]['yield'] - yields_data[0.25]['yield']
                    st.divider()
                    
                    is_inverted = spread < 0
                    
                    st.metric(
                        "10Y-3M Spread",
                        f"{spread:.3f}%",
                        delta="⚠️ Inverted" if is_inverted else "✅ Normal",
                        delta_color="inverse" if is_inverted else "normal"
                    )
                    
                    if is_inverted:
                        st.warning("⚠️ Inverted yield curve may indicate recession risk")
            
            st.divider()
            
            # Analyse historique de la courbe
            st.markdown("#### 📈 Historical Yield Curve Analysis")
            
            st.info("""
            **Yield Curve Interpretation:**
            - **Normal** (positive slope): Economy in expansion
            - **Flat**: Transition period
            - **Inverted** (negative slope): Possible recession ahead
            - **Steep**: Strong growth expectations
            """)
            
        else:
            st.info("Yield curve data not available")
    
    except Exception as e:
        st.error(f"Error loading yield curve: {str(e)}")


def render_bond_metrics_tab(ticker, data, theme):
    """Métriques obligataires (duration, convexity, etc.)"""
    
    st.markdown("### 🔄 Bond Metrics")
    
    is_yield = ticker.startswith('^')
    
    if is_yield:
        st.info("ℹ️ Duration and convexity metrics apply to bond securities, not yield indices.")
        
        # Afficher plutôt des stats sur les taux
        st.markdown("#### 📊 Yield Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_yield = data['Close'].mean()
            st.metric("Average Yield", f"{mean_yield:.3f}%")
        
        with col2:
            std_yield = data['Close'].std()
            st.metric("Std Deviation", f"{std_yield:.3f}%")
        
        with col3:
            current = float(data['Close'].iloc[-1])
            percentile = (data['Close'] < current).sum() / len(data) * 100
            st.metric("Current Percentile", f"{percentile:.1f}%")
        
        with col4:
            range_yield = data['Close'].max() - data['Close'].min()
            st.metric("Range", f"{range_yield:.3f}%")
        
        st.divider()
        
        # Graphique historique des percentiles
        st.markdown("#### 📈 Historical Percentile")
        
        rolling_percentile = data['Close'].rolling(window=252).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else 50
        )
        
        fig_perc = go.Figure()
        
        fig_perc.add_trace(go.Scatter(
            x=rolling_percentile.index,
            y=rolling_percentile.values,
            mode='lines',
            name='Percentile',
            line=dict(color='#6366F1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        
        # Lignes de référence
        fig_perc.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Median")
        fig_perc.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="25th")
        fig_perc.add_hline(y=75, line_dash="dot", line_color="gray", annotation_text="75th")
        
        apply_plotly_theme(
            fig_perc,
            title="Yield Percentile Over Time",
            xaxis_title="Date",
            yaxis_title="Percentile (%)",
            height=400
        )
        
        fig_perc.update_yaxes(range=[0, 100])
        
        st.plotly_chart(fig_perc, use_container_width=True)
        
    else:
        # Pour les ETFs obligataires, afficher des métriques estimées
        st.markdown("#### 📊 Estimated Bond Metrics")
        
        st.info("ℹ️ These are estimated metrics based on price movements. For precise metrics, consult the fund's fact sheet.")
        
        # Estimer la duration effective à partir de la volatilité
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100
        
        # Duration estimée (simplifiée)
        estimated_duration = volatility / 1.5  # Approximation
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Est. Duration", f"{estimated_duration:.2f} years")
            st.caption("Based on price volatility")
        
        with col2:
            sharpe = calculate_sharpe_ratio(data)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.caption("Risk-adjusted return")
        
        with col3:
            max_dd = calculate_max_drawdown(data['Close'])
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
            st.caption("Worst peak-to-trough")
        
        with col4:
            avg_return = returns.mean() * 252 * 100
            st.metric("Avg Annual Return", f"{avg_return:.2f}%")
            st.caption("Annualized")
        
        st.divider()
        
        # Graphique de drawdown
        st.markdown("#### 📉 Drawdown Chart")
        
        cummax = data['Close'].cummax()
        drawdown = (data['Close'] - cummax) / cummax * 100
        
        fig_dd = go.Figure()
        
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#EF4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.2)'
        ))
        
        apply_plotly_theme(
            fig_dd,
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.divider()
        
        # Rolling metrics
        st.markdown("#### 📊 Rolling Metrics (1-Year Window)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rolling return
            rolling_return = data['Close'].pct_change(252) * 100
            
            fig_roll_ret = go.Figure()
            
            fig_roll_ret.add_trace(go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values,
                mode='lines',
                name='1Y Return',
                line=dict(color='#10B981', width=2)
            ))
            
            fig_roll_ret.add_hline(y=0, line_dash="dash", line_color="gray")
            
            apply_plotly_theme(
                fig_roll_ret,
                title="Rolling 1-Year Return",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=300
            )
            
            st.plotly_chart(fig_roll_ret, use_container_width=True)
        
        with col2:
            # Rolling volatility
            rolling_vol = returns.rolling(window=252).std() * (252 ** 0.5) * 100
            
            fig_roll_vol = go.Figure()
            
            fig_roll_vol.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#F59E0B', width=2),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ))
            
            apply_plotly_theme(
                fig_roll_vol,
                title="Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=300
            )
            
            st.plotly_chart(fig_roll_vol, use_container_width=True)


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def add_moving_averages(fig, data, theme):
    """Ajoute les moyennes mobiles au graphique"""
    
    if len(data) >= 20:
        ma20 = data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma20,
            mode='lines',
            name='MA 20',
            line=dict(color='#F59E0B', width=1.5, dash='dash')
        ))
    
    if len(data) >= 50:
        ma50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma50,
            mode='lines',
            name='MA 50',
            line=dict(color='#8B5CF6', width=1.5, dash='dash')
        ))
    
    if len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma200,
            mode='lines',
            name='MA 200',
            line=dict(color='#EF4444', width=1.5, dash='dash')
        ))


def add_moving_averages_to_subplot(fig, data, row, col, theme):
    """Ajoute les moyennes mobiles à un subplot"""
    
    if len(data) >= 20:
        ma20 = data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma20,
            mode='lines',
            name='MA 20',
            line=dict(color='#F59E0B', width=1.5, dash='dash')
        ), row=row, col=col)
    
    if len(data) >= 50:
        ma50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma50,
            mode='lines',
            name='MA 50',
            line=dict(color='#8B5CF6', width=1.5, dash='dash')
        ), row=row, col=col)


def calculate_performance(data, periods):
    """Calcule la performance sur N périodes"""
    if len(data) < periods:
        return 0
    
    try:
        start_price = float(data['Close'].iloc[-periods])
        end_price = float(data['Close'].iloc[-1])
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0


def calculate_ytd_return(data):
    """Calcule le retour YTD"""
    import datetime
    
    try:
        current_year = datetime.datetime.now().year
        ytd_data = data[data.index.year == current_year]
        
        if len(ytd_data) < 2:
            return 0
        
        start_price = float(ytd_data['Close'].iloc[0])
        end_price = float(ytd_data['Close'].iloc[-1])
        
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0


def create_performance_table(data):
    """Crée un tableau de performance"""
    return {
        'Period': ['1 Day', '1 Week', '1 Month', '3 Months', '6 Months', 'YTD', '1 Year'],
        'Return (%)': [
            calculate_performance(data, 1),
            calculate_performance(data, 5),
            calculate_performance(data, 21),
            calculate_performance(data, 63),
            calculate_performance(data, 126),
            calculate_ytd_return(data),
            calculate_performance(data, 252)
        ]
    }


def calculate_max_drawdown(prices):
    """Calcule le drawdown maximum"""
    try:
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax * 100
        return drawdown.min()
    except:
        return 0


def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    """Calcule le ratio de Sharpe"""
    try:
        returns = data['Close'].pct_change().dropna()
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * (252 ** 0.5)
        
        if volatility == 0:
            return 0
        
        return excess_returns / volatility
    except:
        return 0
