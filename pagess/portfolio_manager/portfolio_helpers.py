# portfolio_helpers.py
"""
Fonctions helper pour les portfolios (à placer dans utils.py ou créer ce fichier)
"""

import streamlit as st
import matplotlib.pyplot as plt
from dataprovider import yahoo
from uiconfig import get_theme_colors
import pandas as pd
import plotly.graph_objects as go

def color_column(col, color):
    return [f"color: {color}" for _ in col]

def color_background_column(col, bg_color):
    return [f"background-color: {bg_color}" for _ in col]

def style_data_frame(df, background, color):
    styler = df.style
    styler.map(lambda x: f"'background-color': {background}; 'color':{color};")
    styler.hide()
    st.write(styler.to_html(), unsafe_allow_html=True)

# Définition des tooltips pour chaque métrique
METRIC_TOOLTIPS = {
    "Expected Return": "Rendement annuel attendu basé sur les données historiques",
    "Volatility": "Écart-type annualisé des rendements - mesure du risque total",
    "Sharpe Ratio": "Rendement excédentaire par unité de risque. >1 = bon, >2 = très bon",
    "Sortino Ratio": "Comme Sharpe mais ne pénalise que la volatilité négative. Meilleur pour les rendements asymétriques",
    "Calmar Ratio": "Rendement annuel / Max Drawdown. Mesure le rendement vs pire perte",
    "Treynor Ratio": "Rendement excédentaire par unité de risque systématique (beta)",
    "Information Ratio": "Alpha / Tracking Error. Mesure la compétence du gestionnaire vs benchmark",
    "Omega Ratio": "Gains pondérés vs pertes. >1 = plus de hausse que de baisse",
    "Sterling Ratio": "Rendement / Drawdown moyen. Comme Calmar mais utilise la moyenne",
    "Burke Ratio": "Pénalise les drawdowns multiples plus fortement",
    "Martin Ratio": "Rendement / Ulcer Index. Mesure les rendements ajustés de la douleur",
    "Gain-to-Pain": "Somme des gains / Somme absolue des pertes",
    "Max Drawdown": "Plus grande baisse pic-trou. Montre la pire perte historique",
    "Ulcer Index": "Mesure la profondeur et la durée des drawdowns. Plus bas = mieux",
    "VaR (95%)": "Perte maximale attendue dans 95% des cas. 5% de chance de perdre plus",
    "CVaR": "Perte moyenne dans les pires 5% des cas. Plus conservateur que VaR",
    "Modified VaR": "VaR ajusté pour skewness et kurtosis (Cornish-Fisher)",
    "Historical VaR": "VaR calculé directement à partir des quantiles historiques",
    "Downside Deviation": "Volatilité des rendements négatifs uniquement",
    "Tail Ratio": "95e percentile / 5e percentile. >1 = asymétrie positive",
    "Skewness": "Asymétrie de la distribution. Positif = plus de potentiel de hausse",
    "Kurtosis": "Épaisseur des queues. Valeurs élevées = plus d'événements extrêmes",
    "Alpha": "Rendement excédentaire vs benchmark ajusté du risque",
    "Beta": "Sensibilité aux mouvements du marché. 1 = suit le marché",
    "Tracking Error": "Écart-type des différences de rendements vs benchmark",
    "Positive Periods": "Pourcentage de périodes avec rendements positifs",
    "Negative Periods": "Pourcentage de périodes avec rendements négatifs",
}

def create_metric_card(label, value, tooltip, color=None, icon="📊"):
   
    """Crée une carte métrique avec tooltip"""
    theme = get_theme_colors()
    if color is None:
        color = theme['text_primary']
    
    return f"""
    <div class="metric-card" style="position: relative; padding: 16px; margin: 8px 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 20px;">{icon}</span>
            <div class="metric-label" style="flex: 1;">{label}</div>
            <span title="{tooltip}" style="cursor: help; font-size: 18px; opacity: 0.6;">ℹ️</span>
        </div>
        <div style="font-size: 24px; font-weight: 700; color: {color}; margin-top: 4px;">
            {value}
        </div>
        <div style="font-size: 11px; color: {theme['text_secondary']}; margin-top: 4px; opacity: 0.8;">
            {tooltip[:60]}...
        </div>
    </div>
    """

def render_advanced_metrics_section(portfolio, benchmark=None):
    """Affiche la section des métriques avancées avec un design moderne"""
    theme = get_theme_colors()
    st.markdown("---")
    st.markdown("## 📊 Portfolio Analytics")
    
    # Créer les catégories de métriques
    categories = {
        "🎯 Risk-Adjusted Returns": [
            ("Sharpe Ratio", portfolio.sharp_ratio, "📈", None),
            ("Sortino Ratio", portfolio.sortino_ratio(), "📉", None),
            ("Calmar Ratio", portfolio.calmar_ratio(), "⚡", None),
            ("Omega Ratio", portfolio.omega_ratio(), "🎲", None),
        ],
        "⚠️ Risk Metrics": [
            ("Volatility", portfolio.stdev, "📊", None),
            ("Downside Deviation", portfolio.downside_deviation(), "⬇️", None),
            ("Max Drawdown", portfolio.max_drawdown, "📉", "#EF4444"),
            ("Ulcer Index", portfolio.ulcer_index(), "🤕", None),
        ],
        "💰 Value at Risk": [
            ("VaR (95%)", portfolio.VAR(), "⚠️", "#F59E0B"),
            ("CVaR", portfolio.conditional_var(), "🔴", "#EF4444"),
            ("Modified VaR", portfolio.Cornish_Fisher_var(), "📐", "#F59E0B"),
            ("Historical VaR", portfolio.value_at_risk_historical(), "📊", "#F59E0B"),
        ],
        "📈 Distribution Metrics": [
            ("Skewness", portfolio.skewness, "↗️", None),
            ("Kurtosis", portfolio.kurtosis, "📊", None),
            ("Tail Ratio", portfolio.tail_ratio, "🎭", None),
            ("Positive Periods", portfolio.positive_periods, "✅", "#10B981"),
        ],
        "🔧 Advanced Ratios": [
            ("Sterling Ratio", portfolio.sterling_ratio(), "💎", None),
            ("Burke Ratio", portfolio.burke_ratio(), "🏛️", None),
            ("Martin Ratio", portfolio.martin_ratio(), "🎯", None),
            ("Gain-to-Pain", portfolio.gain_to_pain_ratio(), "⚖️", None),
        ],
    }
    
    # Ajouter les métriques benchmark si disponibles
    if benchmark is not None:
        try:
            categories["🎯 Benchmark Comparison"] = [
                ("Alpha", portfolio.alpha(benchmark) * 252, "⭐", None),
                ("Beta", portfolio.beta(benchmark), "📊", None),
                ("Treynor Ratio", portfolio.treynor_ratio(benchmark), "🎲", None),
                ("Information Ratio", portfolio.information_ratio(benchmark), "ℹ️", None),
                ("Tracking Error", portfolio.tracking_error(benchmark), "📍", None),
            ]
        except Exception as e:
            st.warning(f"Could not calculate benchmark metrics: {str(e)}")
    
    # Afficher chaque catégorie
    for category, metrics in categories.items():
        with st.expander(category, expanded=True):
            cols = st.columns(2)
            
            for idx, (label, value, icon, color) in enumerate(metrics):
                with cols[idx % 2]:
                    # Formater la valeur selon le type de métrique
                    if "Ratio" in label or label in ["Alpha", "Beta"]:
                        formatted_value = f"{value:.4f}"
                    elif "%" in label or label in ["Positive Periods", "Negative Periods"]:
                        formatted_value = f"{value:.2%}"
                    elif "VaR" in label or "$" in str(value):
                        formatted_value = f"${abs(value):,.2f}"
                    elif label in ["Volatility", "Downside Deviation", "Max Drawdown", "Tracking Error"]:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:.4f}"
                    
                    # Déterminer la couleur basée sur la valeur
                    if color is None:
                        if "Ratio" in label or label == "Alpha":
                            if value > 1:
                                color = "#10B981"
                            elif value > 0:
                                color = "#F59E0B"
                            else:
                                color = "#EF4444"
                        elif label == "Beta":
                            if 0.8 <= value <= 1.2:
                                color = "#10B981"
                            else:
                                color = "#F59E0B"
                        else:
                            color = theme['text_primary']
                    
                    tooltip = METRIC_TOOLTIPS.get(label, "")
                    st.markdown(
                        create_metric_card(label, formatted_value, tooltip, color, icon),
                        unsafe_allow_html=True
                    )



@st.cache_data(persist=True)
def cached_get_sectors_weights(assets, weights):
    """Cache les poids par secteur"""
    sector_weights = yahoo.get_sectors_weights(assets, weights)
    return sector_weights


def plot_composition_matplotlib(weights, assets):
    """
    Affiche 2 pie charts: composition assets + composition secteurs
    Version matplotlib (legacy)
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Asset composition
    axs[0].pie(
        weights, 
        labels=assets, 
        autopct='%1.1f%%',
        pctdistance=0.8, 
        startangle=90,
        textprops={'size': 'smaller'}
    )
    axs[0].set_title('Asset Composition')
    
    # Sector composition
    try:
        sector_weights = cached_get_sectors_weights(assets, weights)
        if sector_weights:
            axs[1].pie(
                sector_weights.values(), 
                labels=sector_weights.keys(), 
                autopct='%1.1f%%',
                pctdistance=0.8, 
                startangle=90,
                textprops={'size': 'smaller'}
            )
            axs[1].set_title('Sector Composition')
        else:
            axs[1].text(0.5, 0.5, 'Sector data unavailable', 
                       ha='center', va='center')
    except Exception as e:
        axs[1].text(0.5, 0.5, f'Error: {str(e)}', 
                   ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    
def display_portfolio_results(portfolio, assets, model, **kwargs):
    """Display portfolio metrics and save option avec design amélioré"""
    theme = get_theme_colors()
    
    st.markdown("---")
    st.markdown("### Portfolio Overview")
    
    # Metrics principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Expected Return",
            f"{portfolio.expected_return:.2%}",
            METRIC_TOOLTIPS["Expected Return"],
            icon="📈"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Volatility",
            f"{portfolio.stdev:.2%}",
            METRIC_TOOLTIPS["Volatility"],
            icon="⚠️"
        ), unsafe_allow_html=True)
    
    with col3:
        sharpe_color = "#10B981" if portfolio.sharp_ratio > 1 else "#F59E0B" if portfolio.sharp_ratio > 0 else "#EF4444"
        st.markdown(create_metric_card(
            "Sharpe Ratio",
            f"{portfolio.sharp_ratio:.3f}",
            METRIC_TOOLTIPS["Sharpe Ratio"],
            color=sharpe_color,
            icon="🎯"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "Max Drawdown",
            f"{portfolio.max_drawdown:.2%}",
            METRIC_TOOLTIPS["Max Drawdown"],
            color="#EF4444",
            icon="📉"
        ), unsafe_allow_html=True)
    
    # Weights table
    st.markdown("---")
    st.markdown("### Asset Allocation")
    colc1, colc2 = st.columns(2)
   
    with colc1:
         
    
         weights_df = pd.DataFrame({
        'Asset': assets,
        'Weight': [f"{w:.2%}" for w in portfolio.weights]
    })
         styler= weights_df.style
         styler.hide()
 
         st.write(styler.to_html(index=False), unsafe_allow_html=True)
    # Composition pie chart
    with colc2:
         
         fig = go.Figure(data=[go.Pie(
        labels=assets,
        values=portfolio.weights,
        hole=0.4,
        marker=dict(colors=['#6366F1', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981', '#3B82F6']),
    )])
         fig.update_layout(
        template=theme['plotly_template'],
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        font=dict(color=theme['text_primary'])
    )
         st.plotly_chart(fig, use_container_width=True)
    
    # Advanced metrics avec le nouveau design
    try:
        benchmark = None
        try:
            benchmark = create_benchmark("^GSPC", period="5y")
        except:
            pass
        
        render_advanced_metrics_section(portfolio, benchmark)
    except Exception as e:
        st.warning(f"Could not load advanced metrics: {str(e)}")
    
    # Save section
    st.markdown("---")
    st.markdown("### 💾 Save Portfolio")
    
    with st.form("save_portfolio_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Portfolio Name", placeholder="My Portfolio")
        
        with col2:
            amount = st.number_input("Initial Amount ($)", min_value=100.0, value=10000.0, step=100.0)
        
        submit = st.form_submit_button("Save Portfolio", use_container_width=True)
        
        if submit:
            if not name:
                st.error("❌ Please enter a portfolio name")
            else:
                try:
                    save_portfolio(portfolio, name, model=model, amount=amount, **kwargs)
                    st.success(f"✅ Portfolio '{name}' saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving portfolio: {str(e)}")

