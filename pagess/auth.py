# pagess/auth.py
"""
Page d'authentification - Login et Register
Support pour licensing (student/academic auto-detection)
"""

import streamlit as st
from uiconfig import get_theme_colors
import re
import time

# Import des fonctions database
try:
    from database.user import (
        create_user, 
        authenticate_user, 
        get_user,
        is_educational_email,
        verify_student
    )
except:
    from database import create_user, authenticate_user, get_user

def render_auth():
    """Page principale d'authentification"""
    theme = get_theme_colors()
    
    # Si déjà connecté, rediriger vers dashboard
    if st.session_state.get('user_id'):
        st.session_state.current_page = 'Dashboard'
        st.rerun()
    
    # Initialiser tab state
    if 'auth_tab' not in st.session_state:
        st.session_state.auth_tab = 'login'
    
    # CSS
    st.markdown(f"""
    <style>
        /* Container principal */
        .auth-container {{
            max-width: 450px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }}
        
        /* Header */
        .auth-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .auth-logo {{
            font-size: 64px;
            font-weight: 700;
            background: linear-gradient(135deg, #6366F1, #8B5CF6, #EC4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        
        .auth-title {{
            font-size: 28px;
            font-weight: 700;
            color: {theme['text_primary']};
            margin-bottom: 0.5rem;
        }}
        
        .auth-subtitle {{
            font-size: 14px;
            color: {theme['text_secondary']};
        }}
        
        /* Card */
        .auth-card {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}
        
        /* Override Streamlit styles */
        .stTextInput > div > div > input {{
            border-radius: 8px !important;
            border: 1px solid {theme['border']} !important;
            padding: 0.75rem 1rem !important;
            font-size: 14px !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {theme['accent']} !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1) !important;
        }}
        
        .stTextInput > label {{
            font-size: 13px !important;
            font-weight: 600 !important;
            color: {theme['text_primary']} !important;
            margin-bottom: 0.5rem !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            width: 100%;
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3) !important;
        }}
        
        /* Footer */
        .auth-footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid {theme['border']};
        }}
        
        .auth-footer-text {{
            font-size: 13px;
            color: {theme['text_secondary']};
        }}
        
        .auth-footer-link {{
            color: {theme['accent']};
            text-decoration: none;
            font-weight: 600;
        }}
        
        .auth-footer-link:hover {{
            text-decoration: underline;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Container principal
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="auth-header">
        <div class="auth-logo">Φ</div>
        <div class="auth-title">ΦManager</div>
        <div class="auth-subtitle">Portfolio & Market Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    
    # Tabs
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login", key="tab_login", use_container_width=True):
            st.session_state.auth_tab = 'login'
            st.rerun()
    
    with col2:
        if st.button("Register", key="tab_register", use_container_width=True):
            st.session_state.auth_tab = 'register'
            st.rerun()
    
    st.markdown("---")
    
    # Afficher le formulaire approprié
    if st.session_state.auth_tab == 'login':
        render_login_form(theme)
    else:
        render_register_form(theme)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close card
    
    # Footer
    st.markdown(f"""
    <div class="auth-footer">
        <p class="auth-footer-text">
            By continuing, you agree to our 
            <a href="#" class="auth-footer-link">Terms of Service</a> and 
            <a href="#" class="auth-footer-link">Privacy Policy</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close container


def render_login_form(theme):
    """Formulaire de login"""
    
    st.markdown("### 👋 Welcome Back")
    st.caption("Enter your credentials to access your account")
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username or Email",
            placeholder="Enter your username or email",
            key="login_username"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            remember_me = st.checkbox("Remember me", value=True)
        with col2:
            st.markdown(f'<a href="#" class="auth-footer-link" style="float: right; font-size: 13px;">Forgot password?</a>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Sign In", use_container_width=True)
        
        if submitted:
            handle_login(username, password, remember_me)


def render_register_form(theme):
    """Formulaire de registration avec détection student"""
    
    st.markdown("### 🚀 Create Account")
    st.caption("Join thousands of smart investors")
    
    # Afficher info student/academic
    st.info("""
    🎓 **Students & Academics**
    
    Use your `.edu` or `.ac.*` email to get **free premium access**!
    """)
    
    with st.form("register_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input(
                "First Name",
                placeholder="John",
                key="reg_first_name"
            )
        
        with col2:
            last_name = st.text_input(
                "Last Name",
                placeholder="Doe",
                key="reg_last_name"
            )
        
        username = st.text_input(
            "Username",
            placeholder="Choose a unique username",
            key="reg_username",
            help="3-20 characters, letters and numbers only"
        )
        
        email = st.text_input(
            "Email Address",
            placeholder="john.doe@example.com or john@university.edu",
            key="reg_email"
        )
        
        # Détecter email éducatif en temps réel
        if email and is_educational_email(email):
            st.success("🎓 Student/Academic email detected! You'll get free premium access.")
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a strong password",
            key="reg_password",
            help="Minimum 8 characters"
        )
        
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="reg_confirm_password"
        )
        
        terms = st.checkbox(
            "I agree to the Terms of Service and Privacy Policy",
            key="reg_terms"
        )
        
        # Option newsletter
        newsletter = st.checkbox(
            "Send me tips, trends, and updates (optional)",
            value=True,
            key="reg_newsletter"
        )
        
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        
        if submitted:
            handle_register(
                username, email, password, confirm_password,
                first_name, last_name, terms, newsletter
            )


def handle_login(username, password, remember_me):
    """Gère la tentative de login"""
    
    # Validation
    if not username or not password:
        st.error("⚠️ Please fill in all fields")
        return
    
    # Authentifier
    user = authenticate_user(username, password)
    
    if user:
        # Stocker en session
        st.session_state.user_id = user['_id']
        st.session_state.username = user['username']
        st.session_state.user_email = user['email']
        st.session_state.user_initial = user['username'][0].upper()
        
        # Nouvelles infos license
        st.session_state.license_type = user.get('license_type', 'free')
        st.session_state.subscription_status = user.get('subscription', {}).get('status', 'active')
        
        # Préférences
        if 'preferences' in user:
            prefs = user['preferences']
            if 'theme' in prefs:
                st.session_state.theme = prefs['theme']
        
        # Message de bienvenue avec license info
        license_type = user.get('license_type', 'free')
        
        if license_type == 'student':
            st.success(f"✅ Welcome back, {user['username']}! 🎓 (Student License)")
        elif license_type == 'academic':
            st.success(f"✅ Welcome back, {user['username']}! 👨‍🏫 (Academic License)")
        elif license_type in ['individual', 'professional']:
            st.success(f"✅ Welcome back, {user['username']}! 💼 ({license_type.title()})")
        else:
            st.success(f"✅ Welcome back, {user['username']}!")
        
        # Check subscription expiration
        subscription = user.get('subscription', {})
        if subscription.get('status') == 'trial':
            trial_ends = subscription.get('trial_ends_at')
            if trial_ends:
                import datetime
                days_left = (trial_ends - datetime.datetime.utcnow()).days
                if days_left > 0:
                    st.info(f"⏰ Trial: {days_left} days remaining")
        
        # Délai pour voir le message
        time.sleep(1)
        
        # Rediriger vers dashboard
        st.session_state.current_page = 'Dashboard'
        st.rerun()
    
    else:
        st.error("❌ Invalid username or password")
        st.info("💡 Tip: Username and password are case-sensitive")


def handle_register(username, email, password, confirm_password, 
                    first_name, last_name, terms, newsletter):
    """Gère la registration avec auto-détection student"""
    
    # Validation basique
    errors = []
    
    if not username or not email or not password:
        errors.append("All fields are required")
    
    if not terms:
        errors.append("You must accept the Terms of Service")
    
    # Validation username
    if username:
        if len(username) < 3 or len(username) > 20:
            errors.append("Username must be between 3 and 20 characters")
        if not re.match("^[a-zA-Z0-9_]+$", username):
            errors.append("Username can only contain letters, numbers, and underscores")
    
    # Validation email
    if email:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            errors.append("Invalid email address")
    
    # Validation password
    if password:
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        if password != confirm_password:
            errors.append("Passwords do not match")
    
    # Afficher erreurs
    if errors:
        for error in errors:
            st.error(f"⚠️ {error}")
        return
    
    # Détecter license type
    license_type = 'free'
    if is_educational_email(email):
        license_type = 'student'
    
    # Créer l'utilisateur
    user_id = create_user(
        username=username,
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
        license_type=license_type,
        marketing_emails=newsletter
    )
    
    if user_id:
        # Message de succès adapté au license type
        if license_type == 'student':
            st.success("✅ Account created successfully!")
            st.balloons()
            st.success("""
            🎓 **Student License Activated!**
            
            You have free access to all premium features:
            - ✨ Unlimited portfolios
            - 🤖 AI Assistant unlimited
            - 📊 All ML/RL models
            - 🧪 Experiments Lab
            
            Valid for 1 year (renewable annually with verification)
            """)
        else:
            st.success("✅ Account created successfully!")
            st.info("""
            🎉 Welcome to ΦManager!
            
            You can now login and start with:
            - 1 free portfolio
            - 10 AI queries/day
            - Basic analytics
            
            💎 Upgrade anytime for unlimited access!
            """)
        
        # Auto-switch to login tab
        time.sleep(3)
        st.session_state.auth_tab = 'login'
        st.rerun()
    
    else:
        st.error("❌ Registration failed")
        st.warning("⚠️ Username or email already exists")
        st.info("💡 Try a different username or login if you already have an account")


def validate_email(email):
    """Valide un email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_username(username):
    """Valide un username"""
    if len(username) < 3 or len(username) > 20:
        return False
    if not re.match("^[a-zA-Z0-9_]+$", username):
        return False
    return True


def validate_password(password):
    """Valide un password"""
    if len(password) < 8:
        return False
    
    # Peut ajouter plus de règles:
    # - Au moins une majuscule
    # - Au moins un chiffre
    # - Au moins un caractère spécial
    
    return True
