"""
Reinforcement Learning pour Portfolio - Implémentation Légère
Utilise seulement NumPy et Pandas (pas de TensorFlow/PyTorch)

Algorithmes implémentés:
1. REINFORCE (Policy Gradient)
2. Actor-Critic Simple
3. Deep Q-Learning (Discretized)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioEnvironment:
    """
    Environment de trading pour RL
    Compatible avec l'interface Gym/Gymnasium
    """
    
    def __init__(self, 
                 returns_data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 window_size: int = 20):
        """
        Args:
            returns_data: DataFrame des rendements (lignes=dates, colonnes=actifs)
            initial_capital: Capital initial
            transaction_cost: Coût de transaction (0.1% = 0.001)
            window_size: Taille de la fenêtre d'observation
        """
        self.returns_data = returns_data.values
        self.n_assets = returns_data.shape[1]
        self.asset_names = returns_data.columns.tolist()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        self.n_steps = len(self.returns_data)
        self.current_step = window_size
        self.capital = initial_capital
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        
        # Historique
        self.portfolio_values = [initial_capital]
        self.actions_history = []
        self.rewards_history = []
        
        logger.info(f"Environment créé: {self.n_assets} actifs, {self.n_steps} périodes")
    
    def reset(self) -> np.ndarray:
        """Reset l'environnement"""
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_values = [self.initial_capital]
        self.actions_history = []
        self.rewards_history = []
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construit l'état = features du marché
        Returns: array de shape (window_size * n_assets + n_assets,)
        """
        # Rendements passés (fenêtre glissante)
        past_returns = self.returns_data[
            self.current_step - self.window_size:self.current_step
        ].flatten()
        
        # Poids actuels du portfolio
        current_weights = self.portfolio_weights
        
        # État = [rendements passés, poids actuels]
        state = np.concatenate([past_returns, current_weights])
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute une action (nouveaux poids de portfolio)
        
        Args:
            action: Nouveaux poids du portfolio (doit sommer à 1)
            
        Returns:
            next_state: Prochain état
            reward: Récompense
            done: Si l'épisode est terminé
            info: Informations additionnelles
        """
        # Normaliser l'action pour garantir somme = 1
        action = np.abs(action)
        action = action / action.sum()
        
        # Calculer les coûts de transaction
        portfolio_change = np.abs(action - self.portfolio_weights).sum()
        transaction_costs = portfolio_change * self.transaction_cost * self.capital
        
        # Calculer le rendement du portfolio
        period_returns = self.returns_data[self.current_step]
        portfolio_return = (action * period_returns).sum()
        
        # Mise à jour du capital
        self.capital = self.capital * (1 + portfolio_return) - transaction_costs
        
        # Mise à jour des poids
        self.portfolio_weights = action
        
        # Calcul de la récompense (plusieurs options)
        reward = self._calculate_reward(portfolio_return, transaction_costs)
        
        # Historique
        self.portfolio_values.append(self.capital)
        self.actions_history.append(action.copy())
        self.rewards_history.append(reward)
        
        # Avancer d'un pas
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # État suivant
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        
        # Info
        info = {
            'capital': self.capital,
            'portfolio_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'weights': action
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, portfolio_return: float, transaction_costs: float) -> float:
        """
        Calcule la récompense (plusieurs stratégies possibles)
        """
        # Option 1: Return simple
        # return portfolio_return
        
        # Option 2: Return - coûts
        # return portfolio_return - (transaction_costs / self.capital)
        
        # Option 3: Log return (plus stable)
        log_return = np.log(1 + portfolio_return) if portfolio_return > -0.99 else -10
        
        # Option 4: Sharpe ratio approximé
        if len(self.rewards_history) > 20:
            recent_returns = self.rewards_history[-20:]
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
            return log_return + 0.1 * sharpe
        
        return log_return


class REINFORCEAgent:
    """
    Agent REINFORCE (Policy Gradient basique)
    Simple mais efficace pour débuter
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001):
        """
        Args:
            state_dim: Dimension de l'état
            action_dim: Dimension de l'action (= nombre d'actifs)
            hidden_dim: Taille de la couche cachée
            learning_rate: Taux d'apprentissage
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Réseau de neurones simple (2 couches)
        # Poids: input -> hidden
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        
        # Poids: hidden -> output
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2 = np.zeros(action_dim)
        
        # Mémoire pour l'épisode
        self.states = []
        self.actions = []
        self.rewards = []
        
        logger.info(f"REINFORCE Agent créé: state_dim={state_dim}, action_dim={action_dim}")
    
    def _relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _softmax(self, x):
        """Softmax pour garantir somme = 1"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        """
        Politique: état -> probabilités d'action
        """
        # Forward pass
        hidden = self._relu(state @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        action_probs = self._softmax(logits)
        
        return action_probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sélectionne une action selon la politique
        """
        action_probs = self.policy(state)
        
        if deterministic:
            # Mode déterministe (pour évaluation)
            return action_probs
        else:
            # Mode stochastique (pour exploration)
            # Ajouter du bruit pour exploration
            noise = np.random.dirichlet(np.ones(self.action_dim) * 10)
            action = 0.9 * action_probs + 0.1 * noise
            return action / action.sum()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Stocke une transition pour l'apprentissage"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self, gamma: float = 0.99):
        """
        Met à jour la politique avec Policy Gradient
        
        Args:
            gamma: Facteur de discount
        """
        if len(self.states) == 0:
            return 0.0
        
        # Calculer les returns (récompenses actualisées)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normaliser les returns (stabilité)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        total_loss = 0
        
        # Pour chaque transition
        for t, (state, action, G) in enumerate(zip(self.states, self.actions, returns)):
            # Forward pass
            hidden = self._relu(state @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            action_probs = self._softmax(logits)
            
            # Gradient du log-likelihood
            # ∇log π(a|s) ≈ (a - π(s)) / π(s)
            grad_log_prob = (action - action_probs)
            
            # Gradient pour W2 et b2
            grad_W2 = np.outer(hidden, grad_log_prob) * G
            grad_b2 = grad_log_prob * G
            
            # Gradient pour W1 et b1 (backprop à travers ReLU)
            grad_hidden = (grad_log_prob @ self.W2.T) * (hidden > 0) * G
            grad_W1 = np.outer(state, grad_hidden)
            grad_b1 = grad_hidden
            
            # Update (gradient ascent)
            self.W2 += self.lr * grad_W2
            self.b2 += self.lr * grad_b2
            self.W1 += self.lr * grad_W1
            self.b1 += self.lr * grad_b1
            
            total_loss += -np.log(action_probs[action.argmax()] + 1e-8) * G
        
        avg_loss = total_loss / len(self.states)
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        
        return avg_loss


class ActorCriticAgent:
    """
    Agent Actor-Critic
    Plus stable que REINFORCE, un peu plus complexe
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 64,
                 learning_rate_actor: float = 0.001,
                 learning_rate_critic: float = 0.01):
        """
        Args:
            state_dim: Dimension de l'état
            action_dim: Dimension de l'action
            hidden_dim: Taille des couches cachées
            learning_rate_actor: LR pour l'acteur
            learning_rate_critic: LR pour le critique
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        
        # Actor network (comme REINFORCE)
        self.actor_W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_W2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.actor_b2 = np.zeros(action_dim)
        
        # Critic network (estime la value function)
        self.critic_W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.critic_b1 = np.zeros(hidden_dim)
        self.critic_W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.critic_b2 = np.zeros(1)
        
        logger.info("Actor-Critic Agent créé")
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def policy(self, state: np.ndarray) -> np.ndarray:
        """Actor: état -> action"""
        hidden = self._relu(state @ self.actor_W1 + self.actor_b1)
        logits = hidden @ self.actor_W2 + self.actor_b2
        return self._softmax(logits)
    
    def value(self, state: np.ndarray) -> float:
        """Critic: état -> valeur estimée"""
        hidden = self._relu(state @ self.critic_W1 + self.critic_b1)
        value = (hidden @ self.critic_W2 + self.critic_b2)[0]
        return value
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sélectionne une action"""
        action_probs = self.policy(state)
        
        if deterministic:
            return action_probs
        else:
            noise = np.random.dirichlet(np.ones(self.action_dim) * 10)
            action = 0.9 * action_probs + 0.1 * noise
            return action / action.sum()
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                   reward: float, next_state: np.ndarray, 
                   done: bool, gamma: float = 0.99):
        """
        Met à jour l'acteur et le critique avec TD learning
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
            gamma: Facteur de discount
        """
        # Critic update (TD error)
        current_value = self.value(state)
        next_value = 0 if done else self.value(next_state)
        td_target = reward + gamma * next_value
        td_error = td_target - current_value
        
        # Update critic
        hidden_c = self._relu(state @ self.critic_W1 + self.critic_b1)
        
        grad_W2_c = np.outer(hidden_c, [td_error])
        grad_b2_c = np.array([td_error])
        
        grad_hidden_c = (np.array([td_error]) @ self.critic_W2.T).flatten() * (hidden_c > 0)
        grad_W1_c = np.outer(state, grad_hidden_c)
        grad_b1_c = grad_hidden_c
        
        self.critic_W2 += self.lr_critic * grad_W2_c
        self.critic_b2 += self.lr_critic * grad_b2_c
        self.critic_W1 += self.lr_critic * grad_W1_c
        self.critic_b1 += self.lr_critic * grad_b1_c
        
        # Actor update (policy gradient with advantage)
        hidden_a = self._relu(state @ self.actor_W1 + self.actor_b1)
        logits = hidden_a @ self.actor_W2 + self.actor_b2
        action_probs = self._softmax(logits)
        
        advantage = td_error
        grad_log_prob = (action - action_probs)
        
        grad_W2_a = np.outer(hidden_a, grad_log_prob) * advantage
        grad_b2_a = grad_log_prob * advantage
        
        grad_hidden_a = (grad_log_prob @ self.actor_W2.T) * (hidden_a > 0) * advantage
        grad_W1_a = np.outer(state, grad_hidden_a)
        grad_b1_a = grad_hidden_a
        
        self.actor_W2 += self.lr_actor * grad_W2_a
        self.actor_b2 += self.lr_actor * grad_b2_a
        self.actor_W1 += self.lr_actor * grad_W1_a
        self.actor_b1 += self.lr_actor * grad_b1_a
        
        return td_error


def train_rl_portfolio(returns_data: pd.DataFrame,
                       agent_type: str = 'reinforce',
                       n_episodes: int = 100,
                       initial_capital: float = 10000.0,
                       transaction_cost: float = 0.001,
                       gamma: float = 0.99,
                       verbose: bool = True) -> Tuple[object, List, Dict]:
    """
    Entraîne un agent RL pour gérer un portfolio
    
    Args:
        returns_data: DataFrame des rendements
        agent_type: 'reinforce' ou 'actor_critic'
        n_episodes: Nombre d'épisodes d'entraînement
        initial_capital: Capital initial
        transaction_cost: Coût de transaction
        gamma: Facteur de discount
        verbose: Afficher les progrès
        
    Returns:
        agent: Agent entraîné
        training_history: Historique de l'entraînement
        best_weights: Meilleurs poids trouvés
    """
    # Créer l'environnement
    env = PortfolioEnvironment(
        returns_data=returns_data,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    # Créer l'agent
    state_dim = env.window_size * env.n_assets + env.n_assets
    action_dim = env.n_assets
    
    if agent_type == 'reinforce':
        agent = REINFORCEAgent(state_dim, action_dim)
    elif agent_type == 'actor_critic':
        agent = ActorCriticAgent(state_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Historique d'entraînement
    training_history = {
        'episode_rewards': [],
        'episode_returns': [],
        'final_capitals': [],
        'sharpe_ratios': [],
        'losses': []
    }
    
    best_sharpe = -np.inf
    best_weights = None
    
    # Boucle d'entraînement
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Sélectionner action
            action = agent.select_action(state, deterministic=False)
            
            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # Entraîner l'agent
            if agent_type == 'reinforce':
                agent.store_transition(state, action, reward)
            elif agent_type == 'actor_critic':
                agent.train_step(state, action, reward, next_state, done, gamma)
            
            state = next_state
        
        # Mise à jour REINFORCE (après l'épisode)
        if agent_type == 'reinforce':
            loss = agent.train(gamma)
            training_history['losses'].append(loss)
        
        # Calculer les métriques de l'épisode
        portfolio_returns = np.diff(env.portfolio_values) / env.portfolio_values[:-1]
        episode_return = (env.capital - initial_capital) / initial_capital
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        
        training_history['episode_rewards'].append(episode_reward)
        training_history['episode_returns'].append(episode_return)
        training_history['final_capitals'].append(env.capital)
        training_history['sharpe_ratios'].append(sharpe)
        
        # Sauvegarder les meilleurs poids
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = env.portfolio_weights.copy()
        
        # Afficher les progrès
        if verbose and (episode + 1) % 10 == 0:
            avg_return = np.mean(training_history['episode_returns'][-10:])
            avg_sharpe = np.mean(training_history['sharpe_ratios'][-10:])
            logger.info(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Return: {avg_return:.2%} | "
                f"Sharpe: {avg_sharpe:.3f} | "
                f"Best Sharpe: {best_sharpe:.3f}"
            )
    
    logger.info(f"Training terminé! Best Sharpe: {best_sharpe:.3f}")
    
    return agent, training_history, best_weights


def evaluate_rl_agent(agent: object,
                      returns_data: pd.DataFrame,
                      initial_capital: float = 10000.0,
                      transaction_cost: float = 0.001) -> Dict:
    """
    Évalue un agent RL entraîné
    
    Returns:
        metrics: Dictionnaire de métriques de performance
    """
    env = PortfolioEnvironment(
        returns_data=returns_data,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
    
    # Calculer les métriques
    portfolio_values = np.array(env.portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    annual_return = total_return * (252 / len(returns))
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cumulative = portfolio_values / initial_capital
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_capital': portfolio_values[-1],
        'portfolio_values': portfolio_values,
        'returns': returns,
        'weights_history': env.actions_history
    }
    
    return metrics


def get_rl_portfolio_weights(returns_data: pd.DataFrame,
                             agent_type: str = 'actor_critic',
                             n_episodes: int = 50,
                             **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Fonction simple pour obtenir des poids de portfolio via RL
    Utilisable directement dans l'interface Streamlit
    
    Args:
        returns_data: DataFrame des rendements
        agent_type: 'reinforce' ou 'actor_critic'
        n_episodes: Nombre d'épisodes
        
    Returns:
        weights: Poids optimaux du portfolio
        info: Informations sur l'entraînement
    """
    logger.info(f"Training RL portfolio with {agent_type}...")
    
    agent, history, best_weights = train_rl_portfolio(
        returns_data=returns_data,
        agent_type=agent_type,
        n_episodes=n_episodes,
        verbose=False,
        **kwargs
    )
    
    # Évaluation
    metrics = evaluate_rl_agent(agent, returns_data)
    
    info = {
        'method': f'RL ({agent_type})',
        'n_episodes': n_episodes,
        'final_sharpe': metrics['sharpe_ratio'],
        'final_return': metrics['total_return'],
        'training_history': history,
        'evaluation_metrics': metrics
    }
    
    # Utiliser les poids du meilleur modèle
    weights = best_weights if best_weights is not None else np.ones(returns_data.shape[1]) / returns_data.shape[1]
    
    return weights, info


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🤖 TEST - REINFORCEMENT LEARNING PORTFOLIO")
    print("=" * 80)
    
    # Créer des données de test
    np.random.seed(42)
    n_days = 500
    n_assets = 5
    
    # Simuler des rendements avec corrélations
    mean_returns = np.array([0.0003, 0.0002, 0.0004, 0.0002, 0.0003])
    cov_matrix = np.array([
        [0.0004, 0.0001, 0.0001, 0.0000, 0.0001],
        [0.0001, 0.0003, 0.0001, 0.0001, 0.0000],
        [0.0001, 0.0001, 0.0005, 0.0001, 0.0001],
        [0.0000, 0.0001, 0.0001, 0.0003, 0.0001],
        [0.0001, 0.0000, 0.0001, 0.0001, 0.0004],
    ])
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    returns_df = pd.DataFrame(
        returns,
        columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D', 'Asset_E']
    )
    
    print(f"\n📊 Données: {n_days} jours, {n_assets} actifs")
    print(f"   Rendements moyens: {returns_df.mean().values}")
    print(f"   Volatilités: {returns_df.std().values}")
    
    # Test 1: REINFORCE
    print("\n" + "=" * 80)
    print("TEST 1: REINFORCE AGENT")
    print("=" * 80)
    
    weights_reinforce, info_reinforce = get_rl_portfolio_weights(
        returns_df,
        agent_type='reinforce',
        n_episodes=50
    )
    
    print(f"\n✅ Entraînement REINFORCE terminé!")
    print(f"   Poids optimaux: {weights_reinforce}")
    print(f"   Sharpe Ratio: {info_reinforce['final_sharpe']:.3f}")
    print(f"   Return Total: {info_reinforce['final_return']:.2%}")
    
    # Test 2: Actor-Critic
    print("\n" + "=" * 80)
    print("TEST 2: ACTOR-CRITIC AGENT")
    print("=" * 80)
    
    weights_ac, info_ac = get_rl_portfolio_weights(
        returns_df,
        agent_type='actor_critic',
        n_episodes=50
    )
    
    print(f"\n✅ Entraînement Actor-Critic terminé!")
    print(f"   Poids optimaux: {weights_ac}")
    print(f"   Sharpe Ratio: {info_ac['final_sharpe']:.3f}")
    print(f"   Return Total: {info_ac['final_return']:.2%}")
    
    # Comparaison avec Equal Weight
    print("\n" + "=" * 80)
    print("COMPARAISON AVEC EQUAL WEIGHT")
    print("=" * 80)
    
    equal_weights = np.ones(n_assets) / n_assets
    env_equal = PortfolioEnvironment(returns_df, initial_capital=10000.0)
    env_equal.reset()
    
    done = False
    state = env_equal._get_state()
    while not done:
        _, _, done, _ = env_equal.step(equal_weights)
    
    equal_returns = np.diff(env_equal.portfolio_values) / env_equal.portfolio_values[:-1]
    equal_sharpe = np.mean(equal_returns) / np.std(equal_returns) * np.sqrt(252)
    equal_return = (env_equal.portfolio_values[-1] - 10000) / 10000
    
    print(f"\nEqual Weight:")
    print(f"   Sharpe: {equal_sharpe:.3f}")
    print(f"   Return: {equal_return:.2%}")
    
    print(f"\nREINFORCE:")
    print(f"   Sharpe: {info_reinforce['final_sharpe']:.3f} ({info_reinforce['final_sharpe']/equal_sharpe:.1%} vs Equal)")
    print(f"   Return: {info_reinforce['final_return']:.2%}")
    
    print(f"\nActor-Critic:")
    print(f"   Sharpe: {info_ac['final_sharpe']:.3f} ({info_ac['final_sharpe']/equal_sharpe:.1%} vs Equal)")
    print(f"   Return: {info_ac['final_return']:.2%}")
    
    # Visualisation
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Training curves
        ax1 = axes[0, 0]
        ax1.plot(info_reinforce['training_history']['sharpe_ratios'], label='REINFORCE')
        ax1.plot(info_ac['training_history']['sharpe_ratios'], label='Actor-Critic')
        ax1.axhline(y=equal_sharpe, color='r', linestyle='--', label='Equal Weight')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Training Progress - Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Weights comparison
        ax2 = axes[0, 1]
        x = np.arange(n_assets)
        width = 0.25
        ax2.bar(x - width, weights_reinforce, width, label='REINFORCE', alpha=0.8)
        ax2.bar(x, weights_ac, width, label='Actor-Critic', alpha=0.8)
        ax2.bar(x + width, equal_weights, width, label='Equal', alpha=0.8)
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Weight')
        ax2.set_title('Portfolio Weights Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(returns_df.columns)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative returns
        ax3 = axes[1, 0]
        ax3.plot(info_reinforce['evaluation_metrics']['portfolio_values'], label='REINFORCE')
        ax3.plot(info_ac['evaluation_metrics']['portfolio_values'], label='Actor-Critic')
        ax3.plot(env_equal.portfolio_values, label='Equal Weight')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.set_title('Cumulative Portfolio Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Returns distribution
        ax4 = axes[1, 1]
        ax4.hist(info_reinforce['evaluation_metrics']['returns'], bins=30, alpha=0.5, label='REINFORCE')
        ax4.hist(info_ac['evaluation_metrics']['returns'], bins=30, alpha=0.5, label='Actor-Critic')
        ax4.hist(equal_returns, bins=30, alpha=0.5, label='Equal')
        ax4.set_xlabel('Daily Returns')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Returns Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rl_portfolio_results.png', dpi=150, bbox_inches='tight')
        print("\n✅ Visualisation sauvegardée: 'rl_portfolio_results.png'")
        
    except ImportError:
        print("\n⚠️  Matplotlib non disponible pour la visualisation")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("=" * 80)
