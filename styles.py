# styles.py
"""
Gestion des styles CSS personnalisés avec support light/dark
"""

import streamlit as st


def apply_custom_css(theme):
    """Applique les styles CSS personnalisés"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {{ 
            font-family: 'Inter', sans-serif; 
        }}
        
        /* ==== BODY BACKGROUND - Correction ==== */
        html {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            color: {theme['text_primary']} !important;
        }}
        
        body {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            color: {theme['text_primary']} !important;
        }}
        
        [data-testid="stApp"]{{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
        }}
        
        .stApp {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
        }}
        /* ==== OPTIMISATION BODY - Réduction scroll ==== */
        .main {{
            padding-top: 0 !important;
            padding-bottom: 1rem !important;
        }}
        
        .block-container {{
            background: transparent !important;
            color: {theme['text_primary']} !important;
            padding-top: 0.5rem !important;
            padding-bottom: 1rem !important;
            max-width: 100% !important;
        }}
        
        /* Réduire espacement éléments */
        .element-container {{
            margin-bottom: 0.5rem !important;
        }}
        
        p {{
            margin-bottom: 0.5rem !important;
        }}
        
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
        }}
        
        [data-testid="stAppScrollToBottomContainer"] {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
        }}
        
         [data-testid="stBottomBlockContainer"] {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            width: '100%';
        }}
        
        [data-testid="stMainBlockContainer"] {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            
        }}
        
        [data-testid="stBottom"] {{
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            
        }}
        
        [data-testid="stBaseLinkButton-secondary"] {{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
         [data-testid="stDataFrame"] {{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
               
        
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        
        section[data-testid="stSidebar"] ~ * {{
            background: transparent !important;
        }}
        
        .st-emotion-cache-128upt6  {{
        
         background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
        
        }}
        
         .section-title {{
            padding: 8px 16px 4px 16px !important;
            margin-top: 4px !important;
            font-size: 10px !important;
        }}
        
         
        [data-testid="stSidebar"] {{
            background: {theme['bg_card']} !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid {theme['border']} !important;
            transition: width 0.3s ease;
            padding-top: 0 !important;
        }}
        
        Réduire padding container 
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 0.5rem !important;
        }}
        
        Logo compact 
        .sb-logo {{
            padding: 10px 16px !important;
            margin-bottom: 8px !important;
            border-bottom: 1px solid {theme['border']};
        }}
        
        .logo-phi {{
            font-size: 28px !important;
            line-height: 1 !important;
            margin-bottom: 2px !important;
        }}
        
        .logo-text {{
            font-size: 16px !important;
            line-height: 1 !important;
        }}
        
       
        [data-testid="stSidebar"] .stButton {{
            margin: 1px 0 !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent !important;
            color: {theme['text_secondary']} !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            text-align: left !important;
            width: 100% !important;
            margin: 0 !important;
            font-size: 13px !important;
            line-height: 1.3 !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(99, 102, 241, 0.1) !important;
            color: {theme['text_primary']} !important;
            transform: none !important;
            box-shadow: none !important;
        }}
        
        
       
        
        
        .stat-card {{
            padding: 8px 12px !important;
            margin: 6px 16px !important;
        }}
        
        .stat-label {{
            font-size: 10px !important;
            margin-bottom: 2px !important;
        }}
        
        .stat-value {{
            font-size: 16px !important;
        }}
        
        
        .user-box {{
            padding: 8px 12px !important;
            margin: 10px 16px !important;
        }}
        
        .user-avatar {{
            width: 30px !important;
            height: 30px !important;
            font-size: 13px !important;
        }}
        
        .user-name {{
            font-size: 12px !important;
            line-height: 1.2 !important;
        }}
        
        .user-email {{
            font-size: 10px !important;
            line-height: 1.2 !important;
        }}
        
        hr {{
            margin: 8px 0 !important;
        }}
        
        
        
        [data-testid="stNumberInputStepDown"] {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
         [data-testid="stNumberInputStepUp"] {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stBaseButton-secondaryFormSubmit"] {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stTextInputRootElement"] {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
         [data-testid="stChatInput"] {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        [data-testid="stChatInput"] div {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        
        
         [data-testid="stNumberInputField"] {{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stTooltipHoverTarget"]{{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stSelectboxVirtualDropdown"]{{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stSelectboxVirtualDropdown"]  div {{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        [data-testid="stSelectboxVirtualDropdown"]  div li {{
            background: rgba(99, 102, 241, 0.1) !important;
            color: {theme['text_primary']} !important;
            
        }}
        
         [data-testid="stSelectboxVirtualDropdown"]  div  li:hover {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        
        [data-testid="data-grid-canvas"] {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-testid="glide-cell-0-0"] {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
         canvas {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-testid="data-grid-canvas"] table {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
         [data-testid="data-grid-canvas"] table thead {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-testid="data-grid-canvas"] table thead tr {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        [data-testid="data-grid-canvas"] table thead tr th {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-testid="stFullScreenFrame"]  {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-testid="stDataFrameResizable"]  {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        [data-baseweb="base-input"] {{
            background: {theme['bg_card']} !important;
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        
        Input{{
          background: {theme['bg_card']} !important;
          background-color:{theme['bg_card']} !important;
          color: {theme['text_primary']} !important;
        
        }}
        
        
        
        
        /* ==== TEXTE - Adapté au thème ==== */
        * {{
            color: inherit;
        }}
        
        body, .main, .block-container {{
            color: {theme['text_primary']} !important;
        }}
        
        p, span, div, label, li, td, th {{
            color: {theme['text_primary']} !important;
        }}
        
        .stMarkdown, .stMarkdown p, .stMarkdown span {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Labels de formulaires */
        label, [data-baseweb="label"], .stLabel {{
            color: {theme['text_primary']} !important;
        }}
        
        /* ==== HEADERS OPTIMISÉS ==== */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['text_primary']} !important;
            line-height: 1.2 !important;
        }}
        
        /* H1 sticky et compact */
        h1 {{
            font-size: 30px !important;
            font-weight: 700 !important;
            margin: 0 !important;
            padding: 0.5rem 0 !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
            background: linear-gradient(135deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%) !important;
            backdrop-filter: blur(10px) !important;
            border-bottom: 1px solid {theme['border']} !important;
        }}
        
        h2 {{
            font-size: 22px !important;
            margin: 0.75rem 0 0.5rem 0 !important;
        }}
        
        h3 {{
            font-size: 17px !important;
            font-weight: 600 !important;
            margin: 0.5rem 0 0.4rem 0 !important;
        }}
        
        h4, h5, h6 {{
            font-size: 15px !important;
            margin: 0.4rem 0 0.3rem 0 !important;
        }}
        
        /* ==== SELECT / DROPDOWN ==== */
        [data-baseweb="select"] {{
            background-color: {theme['bg_card']} !important;
        }}
        
        [data-baseweb="select"] > div {{
            background-color: {theme['bg_card']} !important;
            border-color: {theme['border']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        [data-baseweb="select"] input {{
            color: {theme['text_primary']} !important;
            
        }}
        
        /* Menu déroulant */
        [data-baseweb="popover"] {{
            background-color: {theme['bg_card']} !important;
        }}
        
        [data-baseweb="menu"] {{
            background-color: {theme['bg_card']} !important;
        }}
        
        [data-baseweb="menu"] li {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        [data-baseweb="menu"] li:hover {{
            background-color: rgba(99, 102, 241, 0.1) !important;
        }}
        
        /* ==== INPUTS - TOUS LES TYPES ==== */
        input, textarea {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        .stTextInput input,
        .stTextInput textarea,
        .stNumberInput input,
        input[type="text"],
        input[type="number"],
        input[type="email"] {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            color: {theme['text_primary']} !important;
            padding: 10px 14px !important;
        }}
        
        .stTextInput > div > div > input,
        {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            color: {theme['text_primary']} !important;
            padding: 10px 14px !important;
        }}
        
        input:focus, textarea:focus {{
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
            background-color: {theme['bg_card']} !important;
        }}
        
        input::placeholder, textarea::placeholder {{
            color: {theme['text_secondary']} !important;
            opacity: 0.7 !important;
        }}
        
        /* ==== SELECTBOX ==== */
        .stSelectbox > div > div {{
            background: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            color: {theme['text_primary']} !important;
        }}
        
        .stSelectbox > div > div > select {{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        .stSelectbox > div > div:focus {{
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }}
        
        /* ==== FORMS / TABS / EXPANDERS COMPACTS ==== */
        .stForm {{
            padding: 0.75rem !important;
        }}
        
        .stTextInput, .stNumberInput, .stSelectbox {{
            margin-bottom: 0.5rem !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 14px !important;
            font-size: 13px !important;
        }}
        
        .streamlit-expanderHeader {{
            padding: 8px 14px !important;
            font-size: 13px !important;
        }}
        
        .streamlit-expanderContent {{
            padding: 10px 14px !important;
        }}
        
        /* ==== DATAFRAME / TABLES COMPACTES ==== */
         .stDataFrame *,
        [data-testid="stDataFrame"] *,
        .dataframe *,
        table *,
        .stDataFrame table *,
        .stDataFrame tbody *,
        .stDataFrame thead *,
        .stDataFrame tr *,
        .stDataFrame td *,
        .stDataFrame th * {{
            background-color: inherit !important;
        }}
        
        .stDataFrame,
        .stDataFrame > div,
        [data-testid="stDataFrame"],
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        .row-widget.stDataFrame {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            margin: 0.5rem 0 !important;
        }}
        
        .stDataFrame [data-testid="stDataFrameResizable"],
        .dataframe-container {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
        }}
        
        .stDataFrame table,
        table.dataframe,
        .element-container table,
        div[data-testid="stDataFrame"] table {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
        }}
        
        
        .stDataFrame thead,
        table.dataframe thead,
        .element-container thead,
        div[data-testid="stDataFrame"] thead,
        thead {{
            background-color: {theme['bg_secondary']} !important;
            background: {theme['bg_secondary']} !important;
            background-image: none !important;
        }}
        
        .stDataFrame thead tr,
        table.dataframe thead tr,
        .element-container thead tr,
        div[data-testid="stDataFrame"] thead tr,
        thead tr {{
            background-color: {theme['bg_secondary']} !important;
            background: {theme['bg_secondary']} !important;
            background-image: none !important;
        }}
        
        .stDataFrame thead tr th,
        .stDataFrame thead th,
        table.dataframe thead th,
        .element-container thead th,
        div[data-testid="stDataFrame"] thead th,
        thead th,
        th.col_heading,
        th[class*="col"] {{
            background-color: {theme['bg_secondary']} !important;
            background: {theme['bg_secondary']} !important;
            background-image: none !important;
            color: {theme['text_primary']} !important;
            border-bottom: 2px solid {theme['border']} !important;
            border-right: 1px solid {theme['border']} !important;
            padding: 6px 10px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
        }}
        
      
        .stDataFrame tbody,
        table.dataframe tbody,
        .element-container tbody,
        div[data-testid="stDataFrame"] tbody,
        tbody {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            background-image: none !important;
        }}
        
        .stDataFrame tbody tr,
        table.dataframe tbody tr,
        .element-container tbody tr,
        div[data-testid="stDataFrame"] tbody tr,
        tbody tr {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            background-image: none !important;
        }}
        
        /*.stDataFrame tbody tr td,
        .stDataFrame tbody td,
        table.dataframe tbody td,
        .element-container tbody td,
        div[data-testid="stDataFrame"] tbody td,
        tbody td,
        td[class*="col"],
        td.data {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            background-image: none !important;
            color: {theme['text_primary']} !important;
            border-bottom: 1px solid {theme['border']} !important;
            border-right: 1px solid {theme['border']} !important;
            padding: 6px 10px !important;
            font-size: 13px !important;
        }}*/
        
        
        .stDataFrame tbody tr:hover,
        .stDataFrame tbody tr:hover *,
        table.dataframe tbody tr:hover,
        table.dataframe tbody tr:hover *,
        tbody tr:hover,
        tbody tr:hover * {{
            background-color: rgba(99, 102, 241, 0.08) !important;
            background: rgba(99, 102, 241, 0.08) !important;
            background-image: none !important;
        }}
        
        
        .stDataFrame tbody th,
        table.dataframe tbody th,
        tbody th,
        th.row_heading {{
            background-color: {theme['bg_secondary']} !important;
            background: {theme['bg_secondary']} !important;
            background-image: none !important;
            color: {theme['text_primary']} !important;
            border-right: 2px solid {theme['border']} !important;
            font-weight: 600 !important;
        }}
        
        .stDataFrame table[style*="background"],
        .stDataFrame tr[style*="background"],
        .stDataFrame td[style*="background"],
        .stDataFrame th[style*="background"],
        table[style*="background"],
        tr[style*="background"],
        td[style*="background"],
        th[style*="background"] {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            background-image: none !important;
        }}
        
        .stDataFrame thead[style*="background"],
        .stDataFrame thead tr[style*="background"],
        .stDataFrame thead th[style*="background"],
        thead[style*="background"],
        thead tr[style*="background"],
        thead th[style*="background"] {{
            background-color: {theme['bg_secondary']} !important;
            background: {theme['bg_secondary']} !important;
            background-image: none !important;
        }}
        
        /* ==== TABS ==== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 1px solid {theme['border']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            color: {theme['text_secondary']} !important;
            padding: 8px 14px !important;
            font-weight: 600;
            background: transparent !important;
            font-size: 13px !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {theme['text_primary']} !important;
            background: rgba(99, 102, 241, 0.1) !important;
            border-bottom: 2px solid #6366F1 !important;
        }}
        
        /* ==== EXPANDER ==== */
        .streamlit-expanderHeader {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 8px !important;
            padding: 8px 14px !important;
            font-size: 13px !important;
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: rgba(99, 102, 241, 0.05) !important;
        }}
        
        .streamlit-expanderContent {{
            background-color: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            border-top: none !important;
            padding: 10px 14px !important;
        }}
        
        /* ==== METRIC COMPACT ==== */
        [data-testid="stMetric"] {{
            padding: 6px 10px !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {theme['text_primary']} !important;
            font-size: 18px !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {theme['text_secondary']} !important;
            font-size: 11px !important;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {theme['text_secondary']} !important;
        }}
        
        /* ==== BOUTONS PRINCIPAUX ==== */
        button[kind="primary"],
        button[data-testid="baseButton-primary"],
        .stButton button[kind="primary"],
        .stButton > button {{
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 20px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }}
        
        button[kind="primary"]:hover,
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        }}
        
        [data-testid="stSidebar"] button {{
            background: transparent !important;
            color: {theme['text_secondary']} !important;
        }}
        
        [data-testid="stSidebar"] button:hover {{
            background: rgba(99, 102, 241, 0.1) !important;
            color: {theme['text_primary']} !important;
            transform: none !important;
        }}
        
        .stForm button[type="submit"],
        button[kind="primaryFormSubmit"] {{
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            color: white !important;
        }}
        
        /* ==== CARDS COMPACTES ==== */
        .glass-card {{
            background: {theme['bg_card']};
            backdrop-filter: blur(20px);
            border: 1px solid {theme['border']};
            border-radius: 16px;
            padding: 14px !important;
            margin: 6px 0 !important;
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
            padding: 10px 14px !important;
            transition: all 0.3s ease;
        }}
        
        .metric-label {{
            font-size: 11px !important;
            font-weight: 500;
            color: {theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px !important;
        }}
        
        .metric-value {{
            font-size: 22px !important;
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* ==== LOGO ==== */
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
            transition: opacity 0.3s ease;
        }}
        
        /* ==== COLORS ==== */
        .positive {{ color: #10B981 !important; font-weight: 600; }}
        .negative {{ color: #EF4444 !important; font-weight: 600; }}
        
        /* ==== TABLES GÉNÉRIQUES ==== */
       /* table {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        tr {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        td {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        th {{
            background-color: {theme['bg_secondary']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        thead {{
            background-color: {theme['bg_secondary']} !important;
        }}
        
        tbody {{
            background-color: {theme['bg_card']} !important;
        }}
        
        table[style],
        tr[style],
        td[style],
        th[style] {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
        }}
        
        th[style] {{
            background-color: {theme['bg_secondary']} !important;
        }}*/
        
        /* ==== AI ASSISTANT ==== */
        .ai-welcome {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
        }}
        
        .ai-logo {{
            font-size: 72px;
            margin-bottom: 24px;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .ai-title {{
            font-size: 32px;
            font-weight: 700;
            color: {theme['text_primary']};
            margin-bottom: 12px;
        }}
        
        .ai-subtitle {{
            font-size: 18px;
            color: {theme['text_secondary']};
            margin-bottom: 40px;
        }}
        
        .suggestion-card {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 16px 20px;
            margin: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-block;
        }}
        
        .suggestion-card:hover {{
            border-color: {theme['accent']};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        }}
        
        /* ==== CHAT INPUT - Style Claude ==== */
        .stChatInput {{
            position: relative !important;
            background: transparent !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            padding: 0 20px 20px 20px !important;
        }}
        
        [data-testid="stChatInput"] div div{{
            background: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            
        }}
        [data-testid="stChatInput"] {{
            position: relative !important;
            max-width: 800px !important;
            margin: 0 auto !important;
        }}
        
        [data-testid="stChatInput"] > div {{
            position: relative !important;
            background-color: {theme['bg_card']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 26px !important;
            padding: 6px 56px 6px 20px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }}
        
        [data-testid="stChatInput"]:focus-within > div {{
            border-color: {theme['accent']} !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        }}
        
        [data-testid="stChatInput"] input,
        [data-testid="stChatInputTextArea"],
        .stChatInput input {{
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            color: {theme['text_primary']} !important;
            padding: 10px 0 !important;
            font-size: 15px !important;
            outline: none !important;
            box-shadow: none !important;
            min-height: 24px !important;
        }}
        
        [data-testid="stChatInput"] input::placeholder {{
            color: {theme['text_secondary']} !important;
            opacity: 0.6 !important;
        }}
        
        [data-testid="stChatInput"] button,
        [data-testid="stChatInputSubmitButton"] {{
            position: absolute !important;
            right: 10px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            background: {theme['accent']} !important;
            border: none !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            min-height: 32px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}
        
        [data-testid="stChatInput"] button:hover {{
            background: {theme['accent_hover']} !important;
            transform: translateY(-50%) scale(1.05) !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
        }}
        
        [data-testid="stChatInput"] button:disabled {{
            opacity: 0.5 !important;
            cursor: not-allowed !important;
        }}
        
        [data-testid="stChatInput"] button svg {{
            color: white !important;
            fill: white !important;
            width: 16px !important;
            height: 16px !important;
        }}
        
        .stChatMessage,
        [data-testid="stChatMessage"] {{
            background-color: {theme['bg_card']} !important;
            background: {theme['bg_card']} !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            margin: 12px 0 !important;
            border: 1px solid {theme['border']} !important;
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div {{
            color: {theme['text_primary']} !important;
        }}
        
        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon"] {{
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            border-radius: 50% !important;
        }}
        
        /* ==== SCROLLBAR ==== */
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: {theme['bg_secondary']}; }}
        ::-webkit-scrollbar-thumb {{ 
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            border-radius: 4px;
        }}
        
        /* ==== SPINNER / LOADING ==== */
        .stSpinner > div {{
            border-color: {theme['accent']} !important;
        }}
        
        /* ==== ALERT / INFO / WARNING ==== */
        .stAlert {{
            background-color: {theme['bg_card']} !important;
            color: {theme['text_primary']} !important;
            border: 1px solid {theme['border']} !important;
        }}
        
        /* ==== CODE BLOCKS ==== */
        .stCodeBlock {{
            background-color: {theme['bg_secondary']} !important;
            border: 1px solid {theme['border']} !important;
        }}
        
        code {{
            background-color: {theme['bg_secondary']} !important;
            color: {theme['text_primary']} !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }}
        
        /* ==== SLIDER ==== */
        .stSlider [data-baseweb="slider"] {{
            background-color: {theme['bg_card']} !important;
        }}
        
        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background-color: {theme['accent']} !important;
        }}
        
        /* ==== CHECKBOX / RADIO ==== */
        .stCheckbox label {{
            color: {theme['text_primary']} !important;
        }}
        
        .stRadio label {{
            color: {theme['text_primary']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)
