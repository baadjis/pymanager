# charts.py
"""
Génération de graphiques Plotly
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta import moving_average, rsi, lrc
from uiconfig import get_theme_colors


def create_candlestick_chart(data, indicators=[]):
    """Crée un graphique en chandelier avec indicateurs"""
    theme = get_theme_colors()
    
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
