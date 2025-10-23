# pages/stock_explorer.py
"""
Page Stock Explorer
"""

import streamlit as st
from stock import Stock
from charts import create_candlestick_chart
from uiconfig import get_theme_colors
import pandas as pd


def render_stock_explorer():
    """Page Stock Explorer"""
    theme = get_theme_colors()
    st.markdown("<h1>Stock Explorer</h1>", unsafe_allow_html=True)
    
    ticker = st.text_input("Enter Ticker", placeholder="AAPL")
    
    if ticker:
        try:
            stock = Stock(ticker.upper())
            stock.get_data("1y")
            
            if not stock.data.empty:
                close = [p[0] for p in stock.data["Close"].values]
                value = close[-1]
                if len(close) > 1:
                    pch = (close[-1] - close[-2]) / close[-2] * 100
                    delta = close[-1] - close[-2]
                else:
                    pch, delta = 0, 0
                
                # Header
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div class="metric-label">{ticker.upper()}</div>
                        <div class="metric-value">${value:.2f}</div>
                        <div style="color: {'#10B981' if pch > 0 else '#EF4444'}; font-weight: 600; margin-top: 8px;">
                            {pch:+.2f}% (${delta:+.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if stock.infos:
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sector</div>
                            <div style="font-size: 16px; color: {theme['text_primary']}; margin-top: 8px;">
                                {stock.get_sector()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tabs
            tabs = st.tabs(["Chart", "Info", "Financials", "Dividends", "News"])
            
            # CHART TAB
            with tabs[0]:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
                    indicators = st.multiselect("Indicators", ["MA 50", "MA 200", "RSI", "Volume", "LRC"])
                
                stock.get_data(period)
                data = stock.data
                
                fig = create_candlestick_chart(data, indicators)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # INFO TAB
            with tabs[1]:
                if stock.infos:
                    st.write(stock.infos.get('longBusinessSummary', 'N/A'))
                else:
                    st.info("Info not available")
            
            # FINANCIALS TAB
            with tabs[2]:
                try:
                    financials = stock.get_financials()
                    if financials:
                        for key, df in financials.items():
                            st.markdown(f"**{key}**")
                            st.dataframe(df)
                except:
                    st.info("Financials not available")
            
            # DIVIDENDS TAB
            with tabs[3]:
                try:
                    divs = stock.get_dividends()
                    
                    
                    if not divs.empty:
                        divs_frame=pd.DataFrame({'Date':divs.index,'Dividends': divs.values})
                        #styler=divs_frame.style
                        #print(styler)
                        #styler.hide()
                        st.write(divs_frame.to_html(index=False), unsafe_allow_html=True)
                    
                    else:
                        st.info("No dividends")
                except:
                    st.info("Dividends not available")
            
            # NEWS TAB
            with tabs[4]:
                try:
                    news = stock.get_news()
                    for article in news[:10]:
                        st.subheader(article["content"]["title"])
                        st.html(article["content"].get("summary", ""))
                        st.link_button("Read", article["content"]["clickThroughUrl"]["url"])
                        st.divider()
                except:
                    st.info("News not available")
        
        except Exception as e:
            st.error(f"Error: {e}")
