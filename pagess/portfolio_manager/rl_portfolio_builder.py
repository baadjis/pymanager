import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from portfolio import Portfolio, get_log_returns
from dataprovider import yahoo
from database import save_portfolio
from .rl_portfolio import get_rl_portfolio_weights

user_id = st.session_state.user_id

def build_rl_portfolio(assets, data, theme):
    st.markdown("#### 🤖 Reinforcement Learning Portfolio")
    st.info("💡 L'agent RL apprend à optimiser l'allocation")
    
    try:
        returns_data = get_log_returns(data)
        
        if len(assets) > 1:
            if isinstance(data.columns, pd.MultiIndex):
                returns_df = pd.DataFrame()
                for asset in assets:
                    prices = data[('Adj Close', asset)]
                    returns_df[asset] = np.log(prices / prices.shift(1)).dropna()
            else:
                returns_df = returns_data if len(assets) == 1 else returns_data
        else:
            returns_df = pd.DataFrame({assets[0]: returns_data})
        
        if returns_df.empty or len(returns_df) < 100:
            st.error("❌ Insufficient data (need 100+ points)")
            return
        
        st.success(f"✅ Using {len(returns_df)} data points")
    except Exception as e:
        st.error(f"Error: {e}")
        return
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type = st.selectbox("Algorithm", ["actor_critic", "reinforce"])
        n_episodes = st.slider("Training Episodes", 10, 200, 50, 10)
    
    with col2:
        transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05) / 100
        initial_capital = st.number_input("Initial Capital ($)", 1000.0, 1000000.0, 10000.0, 1000.0)
    
    with st.expander("🔧 Advanced"):
        gamma = st.slider("Discount Factor (γ)", 0.9, 0.999, 0.99, 0.001)
        window_size = st.slider("Lookback Window", 5, 50, 20)
    
    if st.button("🚀 Train", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing...")
        progress_bar.progress(10)
        
        try:
            status_text.text(f"🤖 Training {agent_type.upper()}...")
            progress_bar.progress(30)
            
            with st.spinner(f"Training {n_episodes} episodes..."):
                weights, info = get_rl_portfolio_weights(
                    returns_data=returns_df,
                    agent_type=agent_type,
                    n_episodes=n_episodes,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    gamma=gamma
                )
            
            progress_bar.progress(80)
            status_text.text("✅ Training completed!")
            
            st.session_state.rl_weights = weights
            st.session_state.rl_info = info
            st.session_state.rl_assets = assets
            
            progress_bar.progress(100)
            st.success("🎉 RL Agent trained!")
            
            display_rl_results(weights, info, assets, returns_df, theme, data)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
    
    elif 'rl_weights' in st.session_state and st.session_state.rl_assets == assets:
        st.info("📊 Previous results")
        display_rl_results(st.session_state.rl_weights, st.session_state.rl_info, assets, returns_df, theme, data)


def display_rl_results(weights, info, assets, returns_df, theme, data):
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio", f"{info['final_sharpe']:.3f}")
    with col2:
        st.metric("Total Return", f"{info['final_return']:.2%}")
    with col3:
        st.metric("Episodes", info['n_episodes'])
    with col4:
        st.metric("Algorithm", info['method'].split('(')[1].rstrip(')').upper())
    
    history = info['training_history']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Sharpe', 'Returns', 'Capital', 'Reward'))
    fig.add_trace(go.Scatter(y=history['sharpe_ratios'], line=dict(color='#6366F1', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=history['episode_returns'], line=dict(color='#10B981', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(y=history['final_capitals'], line=dict(color='#F59E0B', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(y=np.cumsum(history['episode_rewards']), line=dict(color='#8B5CF6', width=2)), row=2, col=2)
    fig.update_layout(height=600, showlegend=False, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        weights_df = pd.DataFrame({'Asset': assets, 'Weight': [f"{w:.4f}" for w in weights], 'Percentage': [f"{w*100:.2f}%" for w in weights]})
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_weights = go.Figure(data=[go.Bar(x=assets, y=weights, marker=dict(color=weights, colorscale='Viridis', showscale=True))])
        fig_weights.update_layout(title="Weights", template='plotly_dark', height=400)
        st.plotly_chart(fig_weights, use_container_width=True)
    
    st.markdown("---")
    with st.form("save_rl"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", placeholder="My RL Portfolio")
        with col2:
            amount = st.number_input("Initial ($)", 100.0, value=10000.0, step=100.0)
        
        if st.form_submit_button("Save", use_container_width=True):
            if name:
                try:
                    portfolio = Portfolio(assets, data)
                    portfolio.set_weights(list(weights))
                    save_portfolio(user_id, portfolio, name, model="rl", amount=amount)
                    st.success(f"✅ Saved!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")
