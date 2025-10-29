# pagess/pricing.py
"""
Pricing Page - Standalone
Called from app3.py navigation
"""

import streamlit as st
from typing import Dict
from .pricing_strategy_pro import (
    PRICING_MATRIX,
    BUNDLE_ALL_ACCESS,
    LicenseType,
    BillingCycle,
    VolumeDiscountCalculator,
    LicenseVerification
)
from uiconfig import get_theme_colors

def render_pricing_page():
    """
    Pricing page principale
    À appeler depuis app3.py
    """
    
    theme = get_theme_colors()
    
    # Hero Section
    st.html(f"""
    <div style="
        background: {theme['gradient_primary']};
        padding: 3rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h1 style="color: white; font-size: 2.5rem; margin: 0;">
            💎 PyManager Pricing
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 1rem;">
            Choisissez le plan qui correspond à vos besoins
        </p>
        <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">
            ✨ Gratuit pour étudiants • 🎯 Essai 30 jours sans carte • 💰 Garantie satisfait ou remboursé
        </p>
    </div>
    """)
    
    # Toggle Annual/Monthly
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        billing_toggle = st.radio(
            "Cycle de facturation",
            ["Mensuel", "Annuel (-20%)", "2 ans (-30%)"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if billing_toggle == "Mensuel":
            billing_cycle = BillingCycle.MONTHLY
        elif billing_toggle == "Annuel (-20%)":
            billing_cycle = BillingCycle.YEARLY
        else:
            billing_cycle = BillingCycle.BIENNIAL
    
    st.markdown("")
    
    # Pricing Cards
    tabs = st.tabs([
        "🎓 Student/Academic",
        "👤 Individual", 
        "💼 Professional",
        "🏢 Institutional",
        "🎁 All-Access"
    ])
    
    # TAB 1: STUDENT/ACADEMIC
    with tabs[0]:
        render_student_academic_tab()
    
    # TAB 2: INDIVIDUAL
    with tabs[1]:
        render_individual_tab(billing_cycle)
    
    # TAB 3: PROFESSIONAL
    with tabs[2]:
        render_professional_tab(billing_cycle)
    
    # TAB 4: INSTITUTIONAL
    with tabs[3]:
        render_institutional_tab()
    
    # TAB 5: ALL-ACCESS BUNDLE
    with tabs[4]:
        render_all_access_tab(billing_cycle)
    
    # Comparison Table
    st.markdown("---")
    render_comparison_table()
    
    # FAQ
    st.markdown("---")
    render_faq()
    
    # Footer CTA
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>Des questions ?</h3>
        <p>Notre équipe est là pour vous aider</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("💬 Chat Support", use_container_width=True)
    with col2:
        st.button("📧 Email", use_container_width=True)
    with col3:
        st.button("📞 Appel", use_container_width=True)


def render_student_academic_tab():
    """Tab Student/Academic"""
    
    st.markdown("""
    ## 🎓 100% Gratuit pour Étudiants & Académiques
    
    **Nous soutenons l'éducation en offrant certaines fonctionalités PyManager gratuitement.
    """)
    
    col1, col2 = st.columns(2)
    
    # STUDENT
    with col1:
        st.markdown("""
        <div style="
            border: 2px solid #6366f1;
            border-radius: 12px;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            height: 100%;
        ">
            <h3 style="margin-top: 0; color: white;">Student License</h3>
            <div style="font-size: 2.5rem; font-weight: 700; margin: 1rem 0;">
                GRATUIT
            </div>
            <p style="opacity: 0.9;">Valable 1 an, renouvelable</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("**Inclus:**")
        for feat in PRICING_MATRIX[LicenseType.STUDENT]['features']:
            st.markdown(f"- {feat}")
        
        st.markdown("")
        
        with st.form("student_verify"):
            email = st.text_input("Email étudiant (.edu, .ac.*)")
            
            if st.form_submit_button("✅ Vérifier & Activer", use_container_width=True, type="primary"):
                if LicenseVerification.verify_student_email(email):
                    st.success("✅ Email vérifié ! Licence activée.")
                    st.balloons()
                else:
                    st.error("❌ Email non reconnu. Contactez support@pymanager.com")
        
        st.info("""
        📧 **Méthodes de vérification:**
        - Email universitaire (.edu, .ac.*)
        - GitHub Student Developer Pack
        - Carte étudiante (upload)
        """)
    
    # ACADEMIC
    with col2:
        st.markdown("""
        <div style="
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 2rem;
            background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
            color: white;
            height: 100%;
        ">
            <h3 style="margin-top: 0; color: white;">Academic License</h3>
            <div style="font-size: 2.5rem; font-weight: 700; margin: 1rem 0;">
                GRATUIT
            </div>
            <p style="opacity: 0.9;">Pour enseignants & chercheurs</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("**Inclus:**")
        for feat in PRICING_MATRIX[LicenseType.ACADEMIC]['features']:
            st.markdown(f"- {feat}")
        
        st.markdown("")
        
        if st.button("📧 Demander Licence Academic", use_container_width=True, type="primary"):
            st.info("""
            Envoyez un email à **academic@pymanager.com** avec:
            - Email institutionnel
            - Preuve d'affiliation (carte, lettre)
            - Nombre d'étudiants (pour licence classe)
            
            Réponse sous 48h.
            """)


def render_individual_tab(billing_cycle: BillingCycle):
    """Tab Individual"""
    
    pricing = PRICING_MATRIX[LicenseType.INDIVIDUAL]['price']
    
    st.markdown("## 👤 PyManager Individual")
    st.markdown("Pour investisseurs particuliers et passionnés de finance")
    
    # Pricing display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mensuel",
            f"${pricing[BillingCycle.MONTHLY]}/mois",
        )
    
    with col2:
        yearly_monthly = pricing[BillingCycle.YEARLY] / 12
        st.metric(
            "Annuel",
            f"${yearly_monthly:.2f}/mois",
            delta="Économisez 20%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Lifetime",
            f"${pricing[BillingCycle.LIFETIME]}",
            delta="Paiement unique"
        )
    
    # Continuity Discount Highlight
    st.info("""
    🎁 **Continuity Discount** (comme JetBrains)
    
    Plus vous restez, moins vous payez:
    - **Année 1:** Prix plein
    - **Année 2:** -20% automatique
    - **Année 3+:** -40% permanent
    
    Exemple: $19.99 → $15.99 (an 2) → $11.99 (an 3+)
    """)
    
    # Features
    st.markdown("### ✨ Tout ce dont vous avez besoin")
    
    cols = st.columns(2)
    features = PRICING_MATRIX[LicenseType.INDIVIDUAL]['features']
    
    for idx, feat in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f"- {feat}")
    
    # CTA
    st.markdown("")
    
    selected_price = pricing[billing_cycle]
    if billing_cycle == BillingCycle.YEARLY:
        selected_price_display = f"${selected_price/12:.2f}/mois (facturé ${selected_price}/an)"
    elif billing_cycle == BillingCycle.BIENNIAL:
        selected_price_display = f"${selected_price/24:.2f}/mois (facturé ${selected_price}/2 ans)"
    else:
        selected_price_display = f"${selected_price}/mois"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Prix sélectionné:** {selected_price_display}")
    
    with col2:
        if st.button("🚀 Essai 30 jours", use_container_width=True, type="primary"):
            initiate_trial(LicenseType.INDIVIDUAL, billing_cycle)


def render_professional_tab(billing_cycle: BillingCycle):
    """Tab Professional"""
    
    st.markdown("## 💼 PyManager Professional")
    st.markdown("Pour conseillers financiers, wealth managers et RIAs")
    
    # User count selector
    st.markdown("### Combien d'utilisateurs ?")
    num_users = st.slider("", 1, 50, 1, label_visibility="collapsed")
    
    # Calculate price with volume discount
    calc = VolumeDiscountCalculator()
    
    if billing_cycle == BillingCycle.YEARLY:
        total_price = calc.calculate_price(LicenseType.PROFESSIONAL, num_users, billing_cycle)
        price_per_user_month = total_price / num_users / 12
    else:
        base_monthly = PRICING_MATRIX[LicenseType.PROFESSIONAL]['price'][BillingCycle.MONTHLY]
        total_price = base_monthly * num_users
        price_per_user_month = base_monthly
    
    # Volume discount indicator
    volume_discount = 0
    if num_users >= 25:
        volume_discount = 20
    elif num_users >= 10:
        volume_discount = 15
    elif num_users >= 5:
        volume_discount = 10
    
    # Display pricing
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Par utilisateur/mois",
            f"${price_per_user_month:.2f}"
        )
    
    with col2:
        if billing_cycle == BillingCycle.YEARLY:
            st.metric(
                "Total annuel",
                f"${total_price:,.2f}",
                delta=f"-20% billing annuel"
            )
        else:
            st.metric(
                "Total mensuel",
                f"${total_price:,.2f}"
            )
    
    with col3:
        if volume_discount > 0:
            st.metric(
                "Réduction volume",
                f"-{volume_discount}%",
                delta=f"Économisez ${total_price * volume_discount / 100:,.0f}"
            )
    
    # Features
    st.markdown("### 💼 Fonctionnalités Pro")
    
    cols = st.columns(2)
    features = PRICING_MATRIX[LicenseType.PROFESSIONAL]['features']
    
    for idx, feat in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f"- {feat}")
    
    # CTA
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📞 Contacter Sales", use_container_width=True):
            show_contact_form("professional")
    
    with col2:
        if st.button("🚀 Essai 14 jours", use_container_width=True, type="primary"):
            initiate_trial(LicenseType.PROFESSIONAL, billing_cycle, num_users)


def render_institutional_tab():
    """Tab Institutional"""
    
    st.markdown("## 🏢 PyManager Terminal")
    st.markdown("**La solution institutionnelle complète**")
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 3rem;
        color: white;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">$9,999</h2>
        <p style="opacity: 0.8; margin: 0.5rem 0;">par siège / an</p>
        <p style="opacity: 0.6; font-size: 0.9rem;">Minimum 5 sièges</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.markdown("### 🎯 Conçu pour")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🏦 Banques d'investissement
        - 💼 Asset managers
        - 🏢 Family offices
        """)
    
    with col2:
        st.markdown("""
        - 📊 Hedge funds
        - 🏛️ Fonds de pension
        - 💰 Private equity
        """)
    
    st.markdown("### ⚡ Enterprise Features")
    
    features = PRICING_MATRIX[LicenseType.INSTITUTIONAL]['features']
    
    cols = st.columns(2)
    for idx, feat in enumerate(features):
        with cols[idx % 2]:
            st.markdown(f"- {feat}")
    
    st.info("""
    📊 **Pricing personnalisé** selon:
    - Nombre de sièges
    - Modules requis
    - Niveau de support
    - Développements custom
    - Intégrations systèmes existants
    """)
    
    # CTA
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📅 Demander une démo", use_container_width=True, type="primary"):
            show_demo_request()
    
    with col2:
        if st.button("💬 Discuter avec un expert", use_container_width=True):
            show_contact_form("institutional")


def render_all_access_tab(billing_cycle: BillingCycle):
    """Tab All-Access Bundle"""
    
    st.markdown("## 🎁 PyManager All-Access")
    st.markdown("Tous les produits PyManager actuels et futurs")
    
    bundle_pricing = BUNDLE_ALL_ACCESS['price']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mensuel",
            f"${bundle_pricing[BillingCycle.MONTHLY]}/mois"
        )
    
    with col2:
        yearly_monthly = bundle_pricing[BillingCycle.YEARLY] / 12
        st.metric(
            "Annuel",
            f"${yearly_monthly:.2f}/mois",
            delta="Économisez 20%"
        )
    
    with col3:
        biennial_monthly = bundle_pricing[BillingCycle.BIENNIAL] / 24
        st.metric(
            "2 ans",
            f"${biennial_monthly:.2f}/mois",
            delta="Économisez 30%"
        )
    
    st.success(BUNDLE_ALL_ACCESS['savings'])
    
    st.markdown("### 📦 Inclus dans All-Access")
    
    for item in BUNDLE_ALL_ACCESS['includes']:
        st.markdown(f"- ✅ {item}")
    
    st.markdown(BUNDLE_ALL_ACCESS['description'])
    
    if st.button("🚀 Get All-Access", use_container_width=True, type="primary"):
        initiate_checkout(LicenseType.INDIVIDUAL, billing_cycle, bundle=True)


def render_comparison_table():
    """Comparison table"""
    
    st.subheader("📊 Comparaison des Plans")
    
    import pandas as pd
    
    comparison_data = {
        "Feature": [
            "Portfolios",
            "AI Queries/jour",
            "Market Explorer",
            "Modèles ML/RL",
            "Black-Litterman",
            "Experiments Lab",
            "Export PDF",
            "Alertes",
            "Données temps réel",
            "Historique",
            "Support",
            "API Access",
            "White-label",
        ],
        "Student": ["10", "100", "✓", "✓", "✓", "✓", "✓", "✓", "✗", "5 ans", "Community", "✗", "✗"],
        "Individual": ["∞", "∞", "✓", "✓", "✓", "✓", "✓", "✓", "✗", "10 ans", "Email", "✗", "✗"],
        "Professional": ["∞", "∞", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "15 ans", "Priority", "✓", "✓"],
        "Institutional": ["∞", "∞", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "25 ans", "Dedicated", "✓", "✓"],
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_faq():
    """FAQ section"""
    
    st.subheader("❓ Questions Fréquentes")
    
    with st.expander("💳 Quels moyens de paiement acceptez-vous ?"):
        st.markdown("""
        - Carte bancaire (Visa, Mastercard, Amex)
        - PayPal
        - Virement SEPA (facturation annuelle)
        - Apple Pay / Google Pay
        """)
    
    with st.expander("🔄 Puis-je changer de plan ?"):
        st.markdown("""
        Oui, à tout moment:
        - **Upgrade:** Immédiat, prorata appliqué
        - **Downgrade:** Effectif à la fin du cycle de facturation
        - Pas de frais de changement
        """)
    
    with st.expander("❌ Comment annuler mon abonnement ?"):
        st.markdown("""
        En 1 clic dans Settings > Subscription:
        - Annulation immédiate
        - Pas d'engagement
        - Accès jusqu'à la fin de la période payée
        - Ré-activation possible à tout moment
        """)
    
    with st.expander("🎁 Y a-t-il un essai gratuit ?"):
        st.markdown("""
        Oui!
        - **Individual:** 30 jours gratuits
        - **Professional:** 14 jours gratuits
        - **Institutional:** 30 jours + démo personnalisée
        - Aucune carte bancaire requise
        - Toutes les fonctionnalités incluses
        """)
    
    with st.expander("📧 Vérification Student - quels emails sont acceptés ?"):
        st.markdown("""
        Emails automatiquement vérifiés:
        - `.edu` (US/International)
        - `.ac.uk`, `.ac.fr`, `.ac.*` (UK/France/Intl)
        - Domaines universitaires (ex: `univ-paris.fr`)
        
        Autre:
        - GitHub Student Developer Pack
        - Upload carte étudiante/certificat
        """)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def initiate_trial(license_type: LicenseType, billing_cycle: BillingCycle, num_users: int = 1):
    """Start free trial"""
    
    st.success(f"""
    ✅ **Essai gratuit activé!**
    
    - Durée: 30 jours
    - Accès complet à toutes les fonctionnalités
    - Aucune carte bancaire requise
    - Annulation possible à tout moment
    
    Un email de confirmation vous a été envoyé.
    """)
    
    if st.button("🚀 Commencer maintenant"):
        st.session_state['trial_active'] = True
        st.session_state['show_pricing'] = False
        st.rerun()


def initiate_checkout(license_type: LicenseType, billing_cycle: BillingCycle, 
                      num_users: int = 1, bundle: bool = False):
    """Initiate Stripe checkout"""
    
    # TODO: Integrate Stripe
    st.info("""
    ### 🚀 Checkout
    
    **TODO:** Intégrer Stripe Checkout
    
    1. Créer Stripe Products/Prices
    2. Générer session checkout
    3. Webhook pour activation
    
    **Pour l'instant:** support@pymanager.com
    """)


def show_contact_form(tier: str):
    """Contact form for sales"""
    
    with st.form("contact_sales"):
        st.markdown("### 📞 Contactez notre équipe Sales")
        
        name = st.text_input("Nom complet*")
        email = st.text_input("Email professionnel*")
        company = st.text_input("Entreprise")
        phone = st.text_input("Téléphone")
        
        users = st.number_input("Nombre d'utilisateurs estimé", min_value=1, value=5)
        
        message = st.text_area("Message / Besoins spécifiques")
        
        if st.form_submit_button("📧 Envoyer", type="primary"):
            st.success("""
            ✅ **Message envoyé!**
            
            Notre équipe vous contactera sous 24h.
            """)


def show_demo_request():
    """Demo request for institutional"""
    
    with st.form("demo_request"):
        st.markdown("### 📅 Demander une Démo")
        
        name = st.text_input("Nom complet*")
        email = st.text_input("Email professionnel*")
        company = st.text_input("Entreprise / Institution*")
        role = st.selectbox("Rôle", [
            "Portfolio Manager",
            "Analyst",
            "CTO",
            "Head of Trading",
            "Other"
        ])
        
        employees = st.selectbox("Taille de l'entreprise", [
            "1-10",
            "11-50",
            "51-200",
            "201-500",
            "500+"
        ])
        
        use_case = st.text_area("Cas d'usage / Besoins")
        
        if st.form_submit_button("📅 Programmer une Démo", type="primary"):
            st.success("""
            ✅ **Demande de démo enregistrée!**
            
            Un expert vous contactera sous 24h pour planifier.
            """)


if __name__ == "__main__":
    render_pricing_page()
