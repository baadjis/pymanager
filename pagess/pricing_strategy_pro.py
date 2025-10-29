# pricing_strategy_pro.py
"""
PyManager Professional Pricing Strategy
Inspired by JetBrains + Bloomberg

Structure:
1. FREE for Students/Academic (like JetBrains)
2. INDIVIDUAL for retail investors
3. PROFESSIONAL for advisors
4. INSTITUTIONAL for firms (like Bloomberg)
5. ALL-ACCESS bundle (like JetBrains All Products Pack)
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

# =============================================================================
# PRICING TIERS 
# =============================================================================

class LicenseType(Enum):
    """License types like JetBrains"""
    STUDENT = "student"           # Free with verification
    ACADEMIC = "academic"         # Free for teachers/researchers
    INDIVIDUAL = "individual"     # For personal use
    PROFESSIONAL = "professional" # For financial advisors
    INSTITUTIONAL = "institutional" # For firms (Bloomberg-style)
    NONPROFIT = "nonprofit"       # Free for NGOs

class BillingCycle(Enum):
    """Billing cycles with discounts"""
    MONTHLY = "monthly"
    YEARLY = "yearly"             # -20% discount
    BIENNIAL = "biennial"         # -30% discount
    LIFETIME = "lifetime"         # One-time payment

# =============================================================================
# PRICING MATRIX
# =============================================================================

PRICING_MATRIX = {
    # =========================================================================
    # STUDENT/ACADEMIC - FREE 
    # =========================================================================
    LicenseType.STUDENT: {
        "name": "PyManager Student",
        "tagline": "Apprenez la finance quantitative",
        "price": {
            BillingCycle.MONTHLY: 0,
            BillingCycle.YEARLY: 0,
        },
        "verification_required": True,
        "verification_method": "email_edu",  # .edu or .ac.* email
        "renewal_period": 365,  # Renew yearly with verification
        "limits": {
            "portfolios": 10,
            "ai_queries_per_day": 100,
            "data_history_years": 5,
            "experiments": True,
            "ml_rl_models": True,
            "black_litterman": True,
            "pdf_export": True,
            "alerts": True,
            "realtime_data": False,
            "team_users": 1,
            "api_access": False,
        },
        "features": [
            "✅ Tout PyManager (sauf temps réel)",
            "✅ Portfolios illimités",
            "✅ Tous les modèles ML/RL",
            "✅ AI Assistant illimité",
            "🎓 Parfait pour apprendre",
            "📚 Accès ressources éducatives",
            "🔄 Renouvelable chaque année"
        ],
        "restrictions": [
            "❌ Usage personnel uniquement",
            "❌ Pas d'usage commercial",
            "❌ Pas de données temps réel"
        ]
    },
    
    LicenseType.ACADEMIC: {
        "name": "PyManager Academic",
        "tagline": "Pour enseignants et chercheurs",
        "price": {
            BillingCycle.MONTHLY: 0,
            BillingCycle.YEARLY: 0,
        },
        "verification_required": True,
        "verification_method": "institution_email",
        "limits": {
            "portfolios": 999,
            "ai_queries_per_day": 999,
            "data_history_years": 10,
            "experiments": True,
            "ml_rl_models": True,
            "black_litterman": True,
            "pdf_export": True,
            "alerts": True,
            "realtime_data": False,
            "team_users": 25,  # Pour classe
            "api_access": True,
        },
        "features": [
            "✅ Tout PyManager",
            "👥 Jusqu'à 25 étudiants",
            "📊 Datasets recherche",
            "🔬 Accès API complet",
            "📚 Matériel pédagogique",
            "🎓 Licence classe"
        ]
    },
    
    # =========================================================================
    # INDIVIDUAL - Personal
    # =========================================================================
    LicenseType.INDIVIDUAL: {
        "name": "PyManager Individual",
        "tagline": "Pour investisseurs particuliers",
        "price": {
            BillingCycle.MONTHLY: 19.99,
            BillingCycle.YEARLY: 191.90,    # -20%
            BillingCycle.BIENNIAL: 335.83,  # -30%
            BillingCycle.LIFETIME: 499.00,
        },
        "continuity_discount": {
            "year_2": 0.80,  # -20% en année 2
            "year_3": 0.60,  # -40% en année 3+
        },
        "limits": {
            "portfolios": 999,
            "ai_queries_per_day": 999,
            "data_history_years": 10,
            "experiments": True,
            "ml_rl_models": True,
            "black_litterman": True,
            "pdf_export": True,
            "alerts": True,
            "realtime_data": False,
            "team_users": 1,
            "api_access": False,
            "priority_support": False,
        },
        "features": [
            "🚀 Tout PyManager",
            "📊 Portfolios illimités",
            "🤖 AI Assistant illimité",
            "🧪 Experiments complet",
            "📄 Export PDF",
            "🔔 Alertes temps réel",
            "💾 Historique 10 ans",
            "📉 Discount continuité"
        ]
    },
    
    # =========================================================================
    # PROFESSIONAL - For Financial Advisors
    # =========================================================================
    LicenseType.PROFESSIONAL: {
        "name": "PyManager Professional",
        "tagline": "Pour conseillers financiers",
        "price": {
            BillingCycle.MONTHLY: 99.99,
            BillingCycle.YEARLY: 959.90,    # -20%
            BillingCycle.BIENNIAL: 1679.83, # -30%
        },
        "per_user_pricing": True,
        "min_users": 1,
        "volume_discount": {
            "5_users": 0.90,   # -10%
            "10_users": 0.85,  # -15%
            "25_users": 0.80,  # -20%
        },
        "limits": {
            "portfolios": 999,
            "ai_queries_per_day": 999,
            "data_history_years": 15,
            "experiments": True,
            "ml_rl_models": True,
            "black_litterman": True,
            "pdf_export": True,
            "alerts": True,
            "realtime_data": True,
            "team_users": 999,
            "api_access": True,
            "priority_support": True,
            "white_label": True,
            "client_reporting": True,
            "compliance_tools": True,
        },
        "features": [
            "💼 Tout PyManager Pro",
            "📊 Données temps réel",
            "👥 Multi-utilisateurs",
            "🎨 White-label possible",
            "📋 Reporting clients auto",
            "🔒 Outils conformité",
            "🔧 API complète",
            "📞 Support prioritaire",
            "📈 Analytics avancés",
            "💾 Historique 15 ans",
            "🎓 Formation incluse"
        ]
    },
    
    # =========================================================================
    # INSTITUTIONAL
    # =========================================================================
    LicenseType.INSTITUTIONAL: {
        "name": "PyManager Terminal",
        "tagline": "La solution professionnelle complète",
        "price": {
            BillingCycle.YEARLY: 9999.00,  # Per seat, like Bloomberg $24k
        },
        "per_user_pricing": True,
        "min_users": 5,
        "enterprise_negotiation": True,
        "custom_contract": True,
        "limits": {
            "portfolios": 999,
            "ai_queries_per_day": 999,
            "data_history_years": 25,
            "experiments": True,
            "ml_rl_models": True,
            "black_litterman": True,
            "pdf_export": True,
            "alerts": True,
            "realtime_data": True,
            "team_users": 999,
            "api_access": True,
            "priority_support": True,
            "white_label": True,
            "client_reporting": True,
            "compliance_tools": True,
            "dedicated_server": True,
            "sla_guarantee": True,
            "custom_development": True,
        },
        "features": [
            "🏢 Solution Enterprise",
            "📊 Données institutionnelles",
            "🔒 Sécurité enterprise (SSO, SAML)",
            "⚡ Serveur dédié",
            "🎨 White-label complet",
            "🔧 Développements custom",
            "📞 Account manager dédié",
            "🎓 Formation on-site",
            "📜 SLA 99.9%",
            "🔐 Conformité réglementaire",
            "💾 Historique 25 ans",
            "🌐 Déploiement multi-régions",
            "👥 Utilisateurs illimités (selon contrat)"
        ]
    },
}

# =============================================================================
# ALL-ACCESS BUNDLE 
# =============================================================================

BUNDLE_ALL_ACCESS = {
    "name": "PyManager All-Access",
    "tagline": "Tous les modules PyManager + futures apps",
    "description": """
     accédez à:
    - PyManager Portfolio (core)
    - PyManager Research (à venir)
    - PyManager Trading (à venir)
    - PyManager Analytics Pro (à venir)
    - Tous les futurs produits
    """,
    "price": {
        BillingCycle.MONTHLY: 39.99,
        BillingCycle.YEARLY: 383.90,    # -20%
        BillingCycle.BIENNIAL: 671.83,  # -30%
    },
    "savings": "Économisez 40% vs achats séparés",
    "includes": [
        "PyManager Portfolio",
        "PyManager Research (Q2 2025)",
        "PyManager Trading (Q3 2025)",
        "PyManager Analytics Pro (Q4 2025)",
        "Accès anticipé nouvelles features",
        "Tous futurs modules"
    ]
}

# =============================================================================
# VERIFICATION SYSTEM (Student/Academic)
# =============================================================================

class LicenseVerification:
    """Verification system for free academic licenses"""
    
    @staticmethod
    def verify_student_email(email: str) -> bool:
        """Verify if email is from educational institution"""
        edu_domains = [
            '.edu', '.ac.uk', '.ac.fr', '.edu.au',
            'univ-', 'university', 'college',
            # Add more patterns
        ]
        
        email_lower = email.lower()
        return any(domain in email_lower for domain in edu_domains)
    
    @staticmethod
    def verify_github_student(github_username: str) -> bool:
        """Verify GitHub Student Developer Pack"""
        # Integration with GitHub API
        # Check if user has student pack
        pass
    
    @staticmethod
    def request_manual_verification(user_data: Dict) -> str:
        """Manual verification for edge cases"""
        # Upload student ID, enrollment certificate, etc.
        pass

# =============================================================================
# CONTINUITY DISCOUNT (JetBrains style)
# =============================================================================

class ContinuityDiscountCalculator:
    """
    JetBrains gives discounts for continued subscription
    Year 1: Full price
    Year 2: -20%
    Year 3+: -40%
    """
    
    @staticmethod
    def calculate_discount(user_id: str, license_type: LicenseType) -> float:
        """Calculate discount based on subscription age"""
        
        # Get user subscription history
        subscription_start = get_user_subscription_start(user_id)
        
        if not subscription_start:
            return 1.0  # No discount
        
        years_subscribed = (datetime.now() - subscription_start).days / 365
        
        pricing = PRICING_MATRIX[license_type]
        
        if 'continuity_discount' not in pricing:
            return 1.0
        
        discounts = pricing['continuity_discount']
        
        if years_subscribed >= 2:
            return discounts.get('year_3', 1.0)  # -40%
        elif years_subscribed >= 1:
            return discounts.get('year_2', 1.0)  # -20%
        
        return 1.0  # Full price year 1

# =============================================================================
# VOLUME DISCOUNT (Enterprise)
# =============================================================================

class VolumeDiscountCalculator:
    """Volume discounts for teams"""
    
    @staticmethod
    def calculate_price(license_type: LicenseType, num_users: int, 
                       billing_cycle: BillingCycle) -> float:
        """Calculate total price with volume discount"""
        
        pricing = PRICING_MATRIX[license_type]
        
        # Base price per user
        base_price = pricing['price'][billing_cycle]
        
        # Volume discount
        volume_discount = 1.0
        if 'volume_discount' in pricing:
            if num_users >= 25:
                volume_discount = pricing['volume_discount'].get('25_users', 1.0)
            elif num_users >= 10:
                volume_discount = pricing['volume_discount'].get('10_users', 1.0)
            elif num_users >= 5:
                volume_discount = pricing['volume_discount'].get('5_users', 1.0)
        
        total = base_price * num_users * volume_discount
        
        return total

# =============================================================================
# TRIAL SYSTEM
# =============================================================================

TRIAL_CONFIG = {
    LicenseType.INDIVIDUAL: {
        "trial_days": 30,
        "full_features": True,
        "credit_card_required": False,  # Like JetBrains
    },
    LicenseType.PROFESSIONAL: {
        "trial_days": 14,
        "full_features": True,
        "credit_card_required": False,
    },
    LicenseType.INSTITUTIONAL: {
        "trial_days": 30,
        "full_features": True,
        "credit_card_required": False,
        "demo_required": True,  # Sales call first
    }
}

# =============================================================================
# UPGRADE PATHS (Upsell Strategy)
# =============================================================================

UPGRADE_PATHS = {
    LicenseType.STUDENT: {
        "natural_upgrade": LicenseType.INDIVIDUAL,
        "trigger": "After graduation",
        "discount": 0.50,  # 50% off first year
        "duration": 365,
    },
    LicenseType.INDIVIDUAL: {
        "natural_upgrade": LicenseType.PROFESSIONAL,
        "trigger": "Becomes financial advisor",
        "discount": 0.80,  # 20% off
    },
    LicenseType.PROFESSIONAL: {
        "natural_upgrade": LicenseType.INSTITUTIONAL,
        "trigger": "Joins institution",
    }
}

# =============================================================================
# PRICING PAGE UI
# =============================================================================

def render_pricing():
    """Pricing page like JetBrains"""
    
    import streamlit as st
    
    st.title("💎 PyManager Pricing")
    
    # Tab selector
    tabs = st.tabs([
        "🎓 Student/Academic",
        "👤 Individual", 
        "💼 Professional",
        "🏢 Institutional"
    ])
    
    # STUDENT/ACADEMIC TAB
    with tabs[0]:
        st.markdown("""
        ## 🎓 Free for Students & Academics
        
        **Comme JetBrains**, nous offrons PyManager **gratuitement** 
        aux étudiants et enseignants du monde entier.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Student License")
            st.markdown("**100% Gratuit**")
            for feat in PRICING_MATRIX[LicenseType.STUDENT]['features']:
                st.markdown(f"- {feat}")
            
            st.button("✅ Vérifier mon email .edu", key="student")
        
        with col2:
            st.markdown("### Academic License")
            st.markdown("**100% Gratuit**")
            for feat in PRICING_MATRIX[LicenseType.ACADEMIC]['features']:
                st.markdown(f"- {feat}")
            
            st.button("✅ Vérifier mon email institutionnel", key="academic")
        
        st.info("""
        📧 **Vérification:** Email .edu, .ac.*, ou GitHub Student Pack
        
        🔄 **Renouvellement:** Annuel avec vérification
        """)
    
    # INDIVIDUAL TAB
    with tabs[1]:
        st.markdown("## 👤 Individual")
        
        pricing = PRICING_MATRIX[LicenseType.INDIVIDUAL]['price']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mensuel", f"${pricing[BillingCycle.MONTHLY]}")
        
        with col2:
            yearly_monthly = pricing[BillingCycle.YEARLY] / 12
            st.metric("Annuel", f"${yearly_monthly:.2f}/mois", "Économisez 20%")
        
        with col3:
            biennial_monthly = pricing[BillingCycle.BIENNIAL] / 24
            st.metric("2 ans", f"${biennial_monthly:.2f}/mois", "Économisez 30%")
        
        with col4:
            st.metric("Lifetime", f"${pricing[BillingCycle.LIFETIME]}", "Paiement unique")
        
        st.markdown("### Continuity Discount (comme JetBrains)")
        st.info("""
        - **Année 1:** Prix plein
        - **Année 2:** -20% de réduction
        - **Année 3+:** -40% de réduction permanente
        
        Plus vous restez, moins vous payez ! 📉
        """)
        
        st.markdown("### Features")
        for feat in PRICING_MATRIX[LicenseType.INDIVIDUAL]['features']:
            st.markdown(f"- {feat}")
        
        st.button("🚀 Essai gratuit 30 jours", key="trial_individual", type="primary")
    
    # PROFESSIONAL TAB
    with tabs[2]:
        st.markdown("## 💼 Professional")
        st.markdown("Pour conseillers financiers et wealth managers")
        
        num_users = st.slider("Nombre d'utilisateurs", 1, 50, 1)
        
        calc = VolumeDiscountCalculator()
        yearly_price = calc.calculate_price(
            LicenseType.PROFESSIONAL,
            num_users,
            BillingCycle.YEARLY
        )
        monthly_price = yearly_price / 12
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prix mensuel par utilisateur",
                f"${monthly_price/num_users:.2f}"
            )
        with col2:
            st.metric(
                "Total annuel",
                f"${yearly_price:,.2f}",
                f"Économisez {(1-calc.calculate_price(LicenseType.PROFESSIONAL, num_users, BillingCycle.YEARLY) / (PRICING_MATRIX[LicenseType.PROFESSIONAL]['price'][BillingCycle.MONTHLY] * 12 * num_users)) * 100:.0f}%"
            )
        
        st.markdown("### Features")
        for feat in PRICING_MATRIX[LicenseType.PROFESSIONAL]['features']:
            st.markdown(f"- {feat}")
        
        st.button("📞 Contacter Sales", key="contact_pro")
    
    # INSTITUTIONAL TAB
    with tabs[3]:
        st.markdown("## 🏢 PyManager Terminal")
        st.markdown("**La solution institutionnelle complète**")
        
        st.markdown("""
        ### Bloomberg Terminal Style
        
        **$9,999/an par siège**
        
        Solution complète pour:
        - 🏦 Banques d'investissement
        - 💼 Asset managers
        - 🏢 Family offices
        - 📊 Hedge funds
        """)
        
        st.markdown("### Enterprise Features")
        for feat in PRICING_MATRIX[LicenseType.INSTITUTIONAL]['features']:
            st.markdown(f"- {feat}")
        
        st.info("""
        📞 **Contrat sur mesure**
        
        Pricing personnalisé selon:
        - Nombre de sièges
        - Modules requis
        - Support niveau
        - Développements custom
        """)
        
        st.button("📅 Demander une démo", key="demo_institutional", type="primary")
    
    # ALL-ACCESS BUNDLE
    st.markdown("---")
    st.markdown("## 🎁 PyManager All-Access")
    st.markdown(bundle['tagline'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(BUNDLE_ALL_ACCESS['description'])
        for item in BUNDLE_ALL_ACCESS['includes']:
            st.markdown(f"- ✅ {item}")
    
    with col2:
        st.metric(
            "Prix mensuel",
            f"${BUNDLE_ALL_ACCESS['price'][BillingCycle.MONTHLY]}"
        )
        st.success(BUNDLE_ALL_ACCESS['savings'])
        st.button("🚀 Get All-Access", key="bundle")

# Helper
def get_user_subscription_start(user_id: str):
    """Get when user first subscribed"""
    # Implementation
    pass

bundle = BUNDLE_ALL_ACCESS
