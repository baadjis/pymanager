# pages/dashboard.py
"""
Page Dashboard principale
"""

import streamlit as st
from database import get_portfolios


def render_dashboard():
    """Page Dashboard"""
    st.markdown("<h1>Welcome to Φ Manager</h1>", unsafe_allow_html=True)
    
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
    
    st.markdown("---")
    st.markdown("### Recent Portfolios")
    
    try:
        for pf in portfolios[:5]:
            with st.expander(f"📁 {pf['name']}"):
                col1, col2 = st.columns(2)
                col1.write(f"**Model:** {pf.get('model', 'N/A').title()}")
                col2.write(f"**Value:** ${pf.get('amount', 0):,.2f}")
    except:
        st.info("No portfolios yet. Create one in Portfolio Manager!")
