# uiconfig.py
"""
Configuration UI et thème pour PyManager
Gestion centralisée des couleurs, styles et templates Plotly
Support Dark/Light mode
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio


def init_session_state():
    """Initialise tous les états de session"""
    defaults = {
        'chat_history': [],
        'tab_data': None,
        'theme': 'dark',
        'current_page': 'Dashboard',
        'sidebar_expanded': True,
        'user_name': 'Portfolio Manager',
        'user_email': 'manager@phi.com',
        'user_initial': 'P',
        'sidebar_collapsed': False,
        'theme_initialized': False,
        'explorer_asset': 'stock',
        'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'user_settings': {
            'default_currency': 'USD',
            'date_format': '%Y-%m-%d',
            'number_format': 'compact'
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_theme_colors():
    """Retourne les couleurs selon le thème actif (dark/light)"""
    
    theme = st.session_state.get('theme', 'dark')
    
    if theme == "dark":
        return {
            # Couleurs de fond
            'bg_primary': '#0B0F19',
            'bg_secondary': '#131720',
            'bg_card': 'rgba(26, 31, 46, 0.7)',
            'bg_transparent': 'rgba(0, 0, 0, 0)',
            'plot_bg': 'rgba(26, 31, 46, 0.5)',
            
            # Couleurs de texte
            'text_primary': '#F8FAFC',
            'text_secondary': '#94A3B8',
            'text_muted': '#64748B',
            
            # Couleurs de bordure
            'border': 'rgba(148, 163, 184, 0.12)',
            'border_hover': 'rgba(99, 102, 241, 0.3)',
            
            # Couleurs principales
            'primary_color': '#6366F1',
            'secondary_color': '#8B5CF6',
            'accent': '#6366F1',
            'accent_hover': '#7C3AED',
            
            # Couleurs de statut
            'success_color': '#10B981',
            'danger_color': '#EF4444',
            'warning_color': '#F59E0B',
            'info_color': '#3B82F6',
            
            # Template Plotly
            'plotly_template': 'plotly_dark',
            
            # Couleurs pour graphiques
            'chart_colors': [
                '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B',
                '#10B981', '#3B82F6', '#14B8A6', '#F97316'
            ],
            
            # Gradients
            'gradient_primary': 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
            'gradient_success': 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
            'gradient_danger': 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
        }
    else:  # Light mode
        return {
            # Couleurs de fond
            'bg_primary': '#F8FAFC',
            'bg_secondary': '#F1F5F9',
            'bg_card': 'rgba(255, 255, 255, 0.8)',
            'bg_transparent': 'rgba(0, 0, 0, 0)',
            'plot_bg': 'rgba(255, 255, 255, 0.5)',
            
            # Couleurs de texte
            'text_primary': '#1E293B',
            'text_secondary': '#64748B',
            'text_muted': '#94A3B8',
            
            # Couleurs de bordure
            'border': 'rgba(100, 116, 139, 0.2)',
            'border_hover': 'rgba(99, 102, 241, 0.4)',
            
            # Couleurs principales
            'primary_color': '#6366F1',
            'secondary_color': '#8B5CF6',
            'accent': '#6366F1',
            'accent_hover': '#7C3AED',
            
            # Couleurs de statut
            'success_color': '#10B981',
            'danger_color': '#EF4444',
            'warning_color': '#F59E0B',
            'info_color': '#3B82F6',
            
            # Template Plotly
            'plotly_template': 'plotly',
            
            # Couleurs pour graphiques
            'chart_colors': [
                '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B',
                '#10B981', '#3B82F6', '#14B8A6', '#F97316'
            ],
            
            # Gradients
            'gradient_primary': 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
            'gradient_success': 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
            'gradient_danger': 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
        }


def get_plotly_layout_template(title="", xaxis_title="", yaxis_title="", height=500):
    """
    Retourne un template de layout Plotly adapté au thème actuel
    
    Args:
        title: Titre du graphique
        xaxis_title: Titre de l'axe X
        yaxis_title: Titre de l'axe Y
        height: Hauteur du graphique
    
    Returns:
        dict: Configuration de layout Plotly
    """
    theme = get_theme_colors()
    
    return {
        'title': {
            'text': title,
            'font': {
                'size': 18,
                'color': theme['text_primary'],
                'family': 'Inter, sans-serif'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'title': {
                'text': xaxis_title,
                'font': {'color': theme['text_secondary']}
            },
            'showgrid': True,
            'gridcolor': theme['border'],
            'gridwidth': 1,
            'color': theme['text_secondary'],
            'linecolor': theme['border'],
            'zeroline': False
        },
        'yaxis': {
            'title': {
                'text': yaxis_title,
                'font': {'color': theme['text_secondary']}
            },
            'showgrid': True,
            'gridcolor': theme['border'],
            'gridwidth': 1,
            'color': theme['text_secondary'],
            'linecolor': theme['border'],
            'zeroline': True,
            'zerolinecolor': theme['border'],
            'zerolinewidth': 1
        },
        'template': theme['plotly_template'],
        'paper_bgcolor': theme['bg_transparent'],
        'plot_bgcolor': theme['plot_bg'],
        'height': height,
        'hovermode': 'x unified',
        'showlegend': True,
        'legend': {
            'bgcolor': theme['bg_card'],
            'bordercolor': theme['border'],
            'borderwidth': 1,
            'font': {'color': theme['text_primary']}
        },
        'font': {
            'family': 'Inter, sans-serif',
            'color': theme['text_primary']
        },
        'margin': dict(l=60, r=40, t=60, b=60)
    }


def apply_plotly_theme(fig, title="", xaxis_title="", yaxis_title="", height=500):
    """
    Applique le thème PyManager à une figure Plotly existante
    
    Args:
        fig: Figure Plotly
        title: Titre du graphique
        xaxis_title: Titre de l'axe X
        yaxis_title: Titre de l'axe Y
        height: Hauteur du graphique
    
    Returns:
        fig: Figure avec le thème appliqué
    """
    layout = get_plotly_layout_template(title, xaxis_title, yaxis_title, height)
    fig.update_layout(**layout)
    return fig


def create_custom_plotly_template():
    """
    Crée un template Plotly personnalisé pour PyManager
    S'adapte au thème dark/light
    """
    theme = get_theme_colors()
    
    custom_template = go.layout.Template()
    
    # Configuration du layout
    custom_template.layout = go.Layout(
        font=dict(
            family='Inter, sans-serif',
            color=theme['text_primary']
        ),
        paper_bgcolor=theme['bg_transparent'],
        plot_bgcolor=theme['plot_bg'],
        xaxis=dict(
            showgrid=True,
            gridcolor=theme['border'],
            gridwidth=1,
            color=theme['text_secondary'],
            linecolor=theme['border']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=theme['border'],
            gridwidth=1,
            color=theme['text_secondary'],
            linecolor=theme['border']
        ),
        title=dict(
            font=dict(size=18, color=theme['text_primary']),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            bgcolor=theme['bg_card'],
            bordercolor=theme['border'],
            borderwidth=1,
            font=dict(color=theme['text_primary'])
        ),
        colorway=theme['chart_colors']
    )
    
    # Configuration des traces
    custom_template.data.scatter = [go.Scatter(
        line=dict(width=2),
        marker=dict(size=8)
    )]
    
    custom_template.data.bar = [go.Bar(
        marker=dict(line=dict(width=0))
    )]
    
    # Enregistrer le template
    pio.templates['pymanager'] = custom_template
    pio.templates.default = 'pymanager'


def load_custom_css():
    """
    Charge les styles CSS personnalisés pour l'application
    S'adapte au thème dark/light
    """
    theme = get_theme_colors()
    
    custom_css = f"""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Card styles */
    .glass-card {{
        background: {theme['bg_card']};
        backdrop-filter: blur(10px);
        border: 1px solid {theme['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-2px);
        border-color: {theme['border_hover']};
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }}
    
    /* Metric card */
    .metric-card {{
        background: {theme['bg_secondary']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }}
    
    .metric-card:hover {{
        border-color: {theme['border_hover']};
    }}
    
    .metric-label {{
        font-size: 0.875rem;
        color: {theme['text_secondary']};
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {theme['text_primary']};
        line-height: 1.2;
    }}
    
    /* Buttons */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid {theme['border']};
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {theme['bg_secondary']};
        padding: 0.5rem;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: {theme['text_secondary']};
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {theme['text_primary']};
    }}
    
    /* Input fields */
    .stTextInput > div > div > input {{
        border-radius: 8px;
        border: 1px solid {theme['border']};
        background-color: {theme['bg_secondary']};
        color: {theme['text_primary']};
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {theme['primary_color']};
        box-shadow: 0 0 0 1px {theme['primary_color']};
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        border-radius: 8px;
        border: 1px solid {theme['border']};
        background-color: {theme['bg_secondary']};
        color: {theme['text_primary']};
    }}
    
    /* Dataframes */
    .dataframe {{
        border: 1px solid {theme['border']} !important;
        border-radius: 8px;
        color: {theme['text_primary']};
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['bg_secondary']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme['primary_color']};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme['accent_hover']};
    }}
    
    /* Divider */
    hr {{
        margin: 2rem 0;
        border: none;
        border-top: 1px solid {theme['border']};
    }}
    
    /* Alerts */
    .stAlert {{
        border-radius: 8px;
        border-left: 4px solid {theme['info_color']};
    }}
    
    /* Success message */
    .stSuccess {{
        background-color: rgba(16, 185, 129, 0.1);
        border-left-color: {theme['success_color']};
    }}
    
    /* Error message */
    .stError {{
        background-color: rgba(239, 68, 68, 0.1);
        border-left-color: {theme['danger_color']};
    }}
    
    /* Warning message */
    .stWarning {{
        background-color: rgba(245, 158, 11, 0.1);
        border-left-color: {theme['warning_color']};
    }}
    
    /* Info message */
    .stInfo {{
        background-color: rgba(59, 130, 246, 0.1);
        border-left-color: {theme['info_color']};
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {theme['primary_color']} !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {theme['primary_color']};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        border-radius: 8px;
        background-color: {theme['bg_secondary']};
        border: 1px solid {theme['border']};
        color: {theme['text_primary']};
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {theme['border_hover']};
    }}
    
    /* Metric container */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: {theme['text_primary']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme['text_secondary']};
    }}
    
    /* Link buttons */
    .stLinkButton > a {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    
    .stLinkButton > a:hover {{
        transform: translateY(-1px);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {theme['bg_secondary']};
    }}
    
    /* Main content background */
    .main {{
        background-color: {theme['bg_primary']};
    }}
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)


def get_color_scale(type='diverging'):
    """
    Retourne une échelle de couleurs pour les graphiques
    
    Args:
        type: Type d'échelle ('diverging', 'sequential', 'categorical', 'heatmap')
    
    Returns:
        list: Liste de couleurs
    """
    theme = get_theme_colors()
    
    scales = {
        'diverging': [
            [0, theme['danger_color']],
            [0.5, theme['text_secondary']],
            [1, theme['success_color']]
        ],
        'sequential': [
            [0, 'rgba(99, 102, 241, 0.2)'],
            [0.5, 'rgba(99, 102, 241, 0.6)'],
            [1, theme['primary_color']]
        ],
        'categorical': theme['chart_colors'],
        'heatmap': [
            [0, theme['danger_color']],
            [0.25, '#F97316'],
            [0.5, theme['warning_color']],
            [0.75, '#84CC16'],
            [1, theme['success_color']]
        ]
    }
    
    return scales.get(type, scales['categorical'])


def format_number(value, format_type='auto', decimals=2):
    """
    Formate un nombre selon le type demandé
    
    Args:
        value: Valeur à formater
        format_type: Type de format ('auto', 'currency', 'percent', 'compact', 'integer')
        decimals: Nombre de décimales
    
    Returns:
        str: Valeur formatée
    """
    import pandas as pd
    
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    
    try:
        value = float(value)
        
        if format_type == 'percent':
            return f"{value:.{decimals}f}%"
        
        elif format_type == 'currency':
            if abs(value) >= 1e12:
                return f"${value/1e12:.{decimals}f}T"
            elif abs(value) >= 1e9:
                return f"${value/1e9:.{decimals}f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.{decimals}f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.{decimals}f}K"
            else:
                return f"${value:.{decimals}f}"
        
        elif format_type == 'compact':
            if abs(value) >= 1e12:
                return f"{value/1e12:.{decimals}f}T"
            elif abs(value) >= 1e9:
                return f"{value/1e9:.{decimals}f}B"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.{decimals}f}M"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.{decimals}f}K"
            else:
                return f"{value:.{decimals}f}"
        
        elif format_type == 'integer':
            return f"{int(value):,}"
        
        else:  # auto
            if abs(value) >= 1e6:
                return format_number(value, 'compact', decimals)
            else:
                return f"{value:,.{decimals}f}"
    
    except (ValueError, TypeError):
        return str(value)


def toggle_theme():
    """Bascule entre dark et light mode"""
    if st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'
    
    # Recréer le template Plotly
    create_custom_plotly_template()


# Initialisation au chargement du module
if 'theme_initialized' not in st.session_state or not st.session_state.theme_initialized:
    init_session_state()
    create_custom_plotly_template()
    st.session_state.theme_initialized = True
