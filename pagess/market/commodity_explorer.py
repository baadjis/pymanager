# pages/commodity_explorer.py
"""
Commodity Explorer - Analyse des matières premières
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataprovider import yahoo
from uiconfig import get_theme_colors


COMMODITIES = {
    "⚡ Energy": {
        'CL=F': 'Crude Oil WTI',
        'BZ=F': 'Brent Crude Oil',
        'NG=F': 'Natural Gas',
        'RB=F': 'Gasoline',
        'HO=F': 'Heating Oil',
    },
    "🥇 Precious Metals": {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'PL=F': 'Platinum',
        'PA=F': 'Palladium',
    },
    "🔩 Industrial Metals": {
        'HG=F': 'Copper',
        'ALI=F': 'Aluminum',
    },
    "🌾 Agriculture": {
        'ZC=F': 'Corn',
        'ZW=F': 'Wheat',
        'ZS=F': 'Soybeans',
        'KC=F': 'Coffee',
        'SB=F': 'Sugar',
        'CC=F': 'Cocoa',
        'CT=F': 'Cotton',
    },
    "🥩 Livestock": {
        'LE=F': 'Live Cattle',
        'GF=F': 'Feeder Cattle',
        'HE=F': 'Lean Hogs',
    }
}

def get_values(data):
    return [t[0] for t in data.values]
def render_commodity_explorer():
    """Explorer pour les commodities"""
    theme = get_theme_colors()
    
    st.markdown("### 🏭 Commodity Explorer")
    st.caption("Analyze commodities, precious metals, energy, and agricultural products")
    
    # Recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "🔍 Search Commodity",
            placeholder="Enter symbol (e.g., GC=F for Gold, CL=F for Crude Oil)",
            key="commodity_search"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Analyze", use_container_width=True, type="primary")
    
    # Suggestions populaires
    if not ticker:
        st.markdown("#### 💡 Popular Commodities")
        
        for category, commodities in COMMODITIES.items():
            with st.expander(category):
                cols = st.columns(3)
                for idx, (symbol, name) in enumerate(commodities.items()):
                    with cols[idx % 3]:
                        if st.button(f"{symbol}\n{name}", key=f"comm_{symbol}", use_container_width=True):
                            st.session_state.commodity_search = symbol
                            st.rerun()
        return
    
    if ticker and (search_button or ticker):
        ticker = ticker.upper().strip()
        
        try:
            with st.spinner(f"Loading {ticker}..."):
                # Charger données avec gestion des transpositions
                data = load_commodity_data(ticker, period='1y')
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return
                
                # Header
                render_commodity_header(ticker, data, theme)
                
                st.divider()
                
                # Tabs
                tabs = st.tabs([
                    "📈 Chart",
                    "📊 Analysis",
                    "📉 Seasonality",
                    "🔄 Correlations"
                ])
                
                with tabs[0]:
                    render_commodity_chart_tab(ticker, data, theme)
                
                with tabs[1]:
                    render_commodity_analysis_tab(ticker, data, theme)
                
                with tabs[2]:
                    render_commodity_seasonality_tab(ticker, data, theme)
                
                with tabs[3]:
                    render_commodity_correlations_tab(ticker, data, theme)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)


def load_commodity_data(ticker, period='1y'):
    """
    Charge les données de commodity avec gestion des transpositions
    """
    try:
        data = yahoo.get_ticker_data(ticker, period=period)
        
        if data is None:
            return None
        
        # Vérifier si les données sont transposées
        if isinstance(data, pd.DataFrame):
            # Si les colonnes sont des dates (données transposées)
            if data.columns.dtype == 'datetime64[ns]' or isinstance(data.columns[0], pd.Timestamp):
                st.info("🔄 Detected transposed data, fixing...")
                data = data.T
            
            # Vérifier que les colonnes essentielles existent
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {missing_cols}")
                # Essayer de récupérer au moins Close
                if 'Close' not in data.columns and len(data.columns) > 0:
                    # Prendre la première colonne comme Close
                    data['Close'] = data.iloc[:, 0]
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def render_commodity_header(ticker, data, theme):
    """Header du commodity"""
    
    # Trouver le nom du commodity
    commodity_name = ticker
    for category, commodities in COMMODITIES.items():
        if ticker in commodities:
            commodity_name = commodities[ticker]
            break
    
    current_price = float(data['Close'].iloc[-1])
    
    if len(data) > 1:
        prev_price = float(data['Close'].iloc[-2])
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
    else:
        change = 0
        change_pct = 0
    
    # Calculer performances
    perf_1w = calculate_performance(data, 5)
    perf_1m = calculate_performance(data, 21)
    perf_ytd = calculate_ytd_return(data)
    perf_1y = calculate_performance(data, 252)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"### {commodity_name}")
        st.caption(ticker)
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
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


def render_commodity_chart_tab(ticker, data, theme):
    """Graphique du commodity"""
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3,
            key="commodity_period"
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Candlestick", "Area"],
            key="commodity_chart_type"
        )
    
    with col3:
        indicators = st.multiselect(
            "Indicators",
            ["MA 20", "MA 50", "MA 200", "Bollinger Bands", "Volume"],
            default=["MA 50"],
            key="commodity_indicators"
        )
    
    # Recharger pour la période
    data = load_commodity_data(ticker, period=period)
    
    if data is not None and not data.empty:
        # Créer le graphique avec le thème approprié
        if "Volume" in indicators and 'Volume' in data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Price", "Volume")
            )
            has_volume = True
        else:
            fig = go.Figure()
            has_volume = False
        
        # Ajouter le graphique principal
        if chart_type == "Candlestick" and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            trace = go.Candlestick(
                x=data.index,
                open=get_values(data['Open']),
                high=get_values(data['High']),
                low=get_values(data['Low']),
                close=get_values(data['Close']),
                name=ticker
            )
            if has_volume:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)
        else:
            line_color = theme.get('primary_color', '#6366F1')
            
            if chart_type == "Area":
                trace = go.Scatter(
                    x=data.index,
                    y=get_values(data['Close']),
                    mode='lines',
                    name=ticker,
                    line=dict(color=line_color, width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba(99, 102, 241, 0.1)'
                )
            else:
                trace = go.Scatter(
                    x=data.index,
                    y=get_values(data['Close']),
                    mode='lines',
                    name=ticker,
                    line=dict(color=line_color, width=2)
                )
            
            if has_volume:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)
        
        # Ajouter indicateurs
        add_indicators_to_chart(fig, data, indicators, has_volume, theme)
        
        # Ajouter volume si demandé
        if has_volume and 'Volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(99, 102, 241, 0.3)'
                ),
                row=2, col=1
            )
        
        # Appliquer le thème
        fig.update_layout(
            
            xaxis_title="Date",
            template=theme.get('plotly_template', 'plotly_dark'),
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            title=dict(text=f"{ticker} Price Chart", font=dict(size=18, color=theme['text_primary'])),
           font=dict(color=theme['text_primary']),
            height=600,
            hovermode='x unified',
            showlegend=True
        )
        
        if has_volume:
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Price")
        
        st.plotly_chart(fig, use_container_width=True)


def render_commodity_analysis_tab(ticker, data, theme):
    """Analyse du commodity"""
    
    st.markdown("### 📊 Statistical Analysis")
    
    # Statistiques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_52w = data['High'].tail(252).max().values[0] if len(data) >= 252 else data['High'].max().values[0]
        st.metric("52W High", f"${high_52w:.2f}")
    
    with col2:
        low_52w = data['Low'].tail(252).min().values[0] if len(data) >= 252 else data['Low'].min().values[0]
        st.metric("52W Low", f"${low_52w:.2f}")
    
    with col3:
        volatility = data['Close'].pct_change().std().values[0] * (252 ** 0.5) * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    with col4:
        current = float(data['Close'].iloc[-1])
        st.metric("Current", f"{current:.3f}")
    
    st.divider()
    
   
    
    
    perf_data = create_performance_table(data)
    df_perf = pd.DataFrame(perf_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tableau de performance
        st.markdown("#### 📈 Performance Summary")
        st.dataframe(df_perf, use_container_width=True, hide_index=True)
    
    with col2:
        # Distribution des changements
        st.markdown("#### 📊 Daily Changes Distribution")
        
        returns = data['Close'].pct_change().dropna() * 100
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=get_values(returns),
            nbinsx=50,
            marker_color='#6366F1',
            name='Returns'
        ))
        
        fig_hist.update_layout(
            xaxis_title="Daily Change (%)",
            yaxis_title="Frequency",
            template=theme.get('plotly_template', 'plotly_dark'),
           plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],

           font=dict(color=theme['text_primary']),
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
        
        for ticker, (label, maturity) in maturities.items():
            try:
                data = load_commodity_data(ticker, period='5d')
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
                    marker=dict(size=10, color='#6366F1')
                ))
                
                fig_curve.update_layout(
                    
                    xaxis_title="Maturity (Years)",
                    yaxis_title="Yield (%)",
                    template=theme.get('plotly_template', 'plotly_dark'),
                    plot_bgcolor= theme['bg_card'],
                    paper_bgcolor= theme['bg_card'],
                    title=dict(text="Current Yield Curve", font=dict(size=18, color=theme['text_primary'])),
                    font=dict(color=theme['text_primary']),
                    height=400,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=maturities_list,
                        ticktext=labels_list
                    )
                )
                
                st.plotly_chart(fig_curve, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Current Yields")
                
                for maturity, data_dict in sorted_data:
                    label = data_dict['label']
                    yield_val = data_dict['yield']
                    st.metric(label, f"{yield_val:.3f}%")
                
                # Calculer le spread 10Y-2Y (indicateur de récession)
                if 10 in yields_data and 0.25 in yields_data:
                    spread = yields_data[10]['yield'] - yields_data[0.25]['yield']
                    st.divider()
                    st.metric(
                        "10Y-3M Spread",
                        f"{spread:.3f}%",
                        delta="Inversion" if spread < 0 else "Normal",
                        delta_color="inverse" if spread < 0 else "normal"
                    )
        else:
            st.info("Yield curve data not available")
    
    except Exception as e:
        st.error(f"Error loading yield curve: {str(e)}")


def render_bond_metrics_tab(ticker, data, theme):
    """Métriques obligataires (duration, convexity)"""
    
    st.markdown("### 🔄 Bond Metrics")
    
    is_yield = ticker.startswith('^')
    
    if is_yield:
        st.info("ℹ️ Duration and convexity metrics apply to bond securities, not yield indices.")
        
        # Afficher plutôt des stats sur les taux
        st.markdown("#### 📊 Yield Statistics")
        
        col1, col2, col3 = st.columns(3)
        
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
        
        fig_perc.update_layout(
            xaxis_title="Date",
            yaxis_title="Percentile (%)",
            template=theme.get('plotly_template', 'plotly_dark'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            height=400,
            yaxis_range=[0, 100]
        )
        
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
        
        col1, col2, col3 = st.columns(3)
        
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
        
        fig_dd.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=theme.get('plotly_template', 'plotly_dark'),
           plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],

           font=dict(color=theme['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)


# Fonctions utilitaires communes
def add_indicators_to_chart(fig, data, indicators, has_volume, theme):
    """Ajoute des indicateurs au graphique"""
    
    row = 1 if has_volume else None
    col = 1 if has_volume else None
    
    if "MA 20" in indicators and len(data) >= 20:
        ma20 = data['Close'].rolling(window=20).mean()
        trace = go.Scatter(
            x=data.index,
            y=get_values(ma20),
            mode='lines',
            name='MA 20',
            line=dict(color='#F59E0B', width=1.5, dash='dash')
        )
        if has_volume:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    
    if "MA 50" in indicators and len(data) >= 50:
        ma50 = data['Close'].rolling(window=50).mean()
        trace = go.Scatter(
            x=data.index,
            y=get_values(ma50),
            mode='lines',
            name='MA 50',
            line=dict(color='#8B5CF6', width=1.5, dash='dash')
        )
        if has_volume:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    
    if "MA 200" in indicators and len(data) >= 200:
        ma200 = data['Close'].rolling(window=200).mean()
        trace = go.Scatter(
            x=data.index,
            y=get_values(ma200),
            mode='lines',
            name='MA 200',
            line=dict(color='#EF4444', width=1.5, dash='dash')
        )
        if has_volume:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
    
    if "Bollinger Bands" in indicators and len(data) >= 20:
        ma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        trace_upper = go.Scatter(
            x=data.index,
            y=upper_band,
            mode='lines',
            name='BB Upper',
            line=dict(color='rgba(99, 102, 241, 0.5)', width=1, dash='dot')
        )
        trace_lower = go.Scatter(
            x=data.index,
            y=get_values(lower_band),
            mode='lines',
            name='BB Lower',
            line=dict(color='rgba(99, 102, 241, 0.5)', width=1, dash='dot'),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.1)'
        )
        
        if has_volume:
            fig.add_trace(trace_upper, row=row, col=col)
            fig.add_trace(trace_lower, row=row, col=col)
        else:
            fig.add_trace(trace_upper)
            fig.add_trace(trace_lower)


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


def format_large_number(num):
    """Formate les grands nombres"""
    if not num or num == 0:
        return "N/A"
    
    num = float(num)
    
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


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
        
"""pct_change().std() * (252 ** 0.5) * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    with col4:
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].mean()
            st.metric("Avg Volume", format_large_number(avg_volume))
        else:
            st.metric("Data Points", f"{len(data)}")
    
    st.divider()
    
    # Tableau de performance
    st.markdown("#### 📈 Performance Summary")
    
    perf_data = create_performance_table(data)
    
    # Graphique de performance
    col1, col2 = st.columns(2)
    
    with col1:
        fig_perf = go.Figure()
        
        colors = ['#10B981' if x > 0 else '#EF4444' for x in perf_data['Return (%)']]
        
        fig_perf.add_trace(go.Bar(
            x=perf_data['Period'],
            y=perf_data['Return (%)'],
            marker_color=colors,
            text=[f"{x:+.1f}%" for x in perf_data['Return (%)']],
            textposition='outside'
        ))
        
        fig_perf.update_layout(
            title="Returns by Period",
            xaxis_title="Period",
            yaxis_title="Return (%)",
            template=theme.get('plotly_template', 'plotly_dark'),
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],
            title=dict(text="Returns by Period", font=dict(size=18, color=theme['text_primary'])),
           font=dict(color=theme['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        st.dataframe(
            pd.DataFrame(perf_data),
            use_container_width=True,
            hide_index=True
        )

"""
def render_commodity_seasonality_tab(ticker, data, theme):
    """Analyse de saisonnalité"""
    
    st.markdown("### 📉 Seasonality Analysis")
    
    if len(data) < 252:
        st.info("ℹ️ Need at least 1 year of data for seasonality analysis")
        return
    
    # Calculer rendements mensuels moyens
    data_copy = data.copy()
    data_copy['Month'] = data_copy.index.month
    data_copy['Returns'] = data_copy['Close'].pct_change() * 100
    
    monthly_returns = data_copy.groupby('Month')['Returns'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Average Monthly Returns")
        
        fig_season = go.Figure()
        
        colors = ['#10B981' if x > 0 else '#EF4444' for x in monthly_returns.values]
        
        fig_season.add_trace(go.Bar(
            x=months,
            y=monthly_returns.values,
            marker_color=colors,
            text=[f"{x:+.2f}%" for x in monthly_returns.values],
            textposition='outside'
        ))
        
        fig_season.update_layout(
            xaxis_title="Month",
            yaxis_title="Avg Return (%)",
            template=theme.get('plotly_template', 'plotly_dark'),
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],

           font=dict(color=theme['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_season, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Monthly Performance Heatmap")
        
        # Créer une heatmap par année
        data_copy['Year'] = data_copy.index.year
        pivot_data = data_copy.pivot_table(
            values='Returns',
            index='Year',
            columns='Month',
            aggfunc='sum'
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=months,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=pivot_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)")
        ))
        
        fig_heatmap.update_layout(
            xaxis_title="Month",
            yaxis_title="Year",
            template=theme.get('plotly_template', 'plotly_dark'),
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],

           font=dict(color=theme['text_primary']),
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)


def render_commodity_correlations_tab(ticker, data, theme):
    """Corrélations avec autres commodities"""
    
    st.markdown("### 🔄 Correlation Analysis")
    
    # Définir les commodities de référence selon la catégorie
    reference_commodities = {
        'GC=F': ['SI=F', 'PL=F', 'PA=F'],  # Precious metals
        'CL=F': ['BZ=F', 'NG=F', 'RB=F'],  # Energy
        'ZC=F': ['ZW=F', 'ZS=F', 'KC=F'],  # Agriculture
    }
    
    # Déterminer les commodities à comparer
    comps = reference_commodities.get(ticker, ['GC=F', 'CL=F', 'SI=F'])
    
    if ticker in comps:
        comps.remove(ticker)
    
    correlation_data = {}
    
    for comp in comps:
        try:
            comp_data = load_commodity_data(comp, period='1y')
            if comp_data is not None and not comp_data.empty:
                correlation_data[comp] = get_values(comp_data['Close'])
        except:
            continue
    
    if correlation_data:
        df_comparison = pd.DataFrame(correlation_data)
        df_comparison[ticker] = get_values(data['Close'])
        
        # Calculer corrélations
        correlations = df_comparison.corr()[ticker].drop(ticker)
        
        st.markdown("#### Correlation with Related Commodities")
        
        fig_corr = go.Figure()
        
        fig_corr.add_trace(go.Bar(
            x=[c.replace('=F', '') for c in correlations.index],
            y=correlations.values,
            marker_color=['#10B981' if x > 0.5 else '#F59E0B' if x > 0 else '#EF4444' 
                          for x in correlations.values],
            text=[f"{x:.2f}" for x in correlations.values],
            textposition='outside'
        ))
        
        fig_corr.update_layout(
            title="Correlation Coefficients",
            xaxis_title="Commodity",
            yaxis_title="Correlation",
            template=theme.get('plotly_template', 'plotly_dark'),
            plot_bgcolor= theme['bg_card'],
            paper_bgcolor= theme['bg_card'],

           font=dict(color=theme['text_primary']),
            height=400,
            yaxis_range=[-1, 1]
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlation data not available")


