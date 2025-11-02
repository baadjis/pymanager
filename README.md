# 📊 ΦManager - Portfolio & Market Intelligence Platform

> Plateforme moderne de gestion de portefeuille avec intelligence artificielle intégrée

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/mongodb-7.0+-green.svg)](https://www.mongodb.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 À propos

**ΦManager** (Phi Manager) est une plateforme complète de gestion de portefeuille et d'analyse de marché, combinant:

- 📊 **Portfolio Manager** - Construction et optimisation (Markowitz, ML, RL, Black-Litterman)
- 📈 **Market Explorer** - Données temps réel avec indicateurs techniques avancés
- 🤖 **AI Assistant** - Conseiller intelligent avec Claude AI et MCP Server
- 🔍 **Stock Screener** - Filtres avancés et recherche personnalisée
- 🧪 **Experiments Lab** - Backtesting, comparaison de modèles, ML/RL training
- 📱 **Interface Moderne** - Design glassmorphism avec thème sombre

---

## ✨ Fonctionnalités Principales

### 🏠 Dashboard
- Vue d'ensemble de tous vos portefeuilles
- Graphiques de performance en temps réel
- Métriques clés (rendement, volatilité, Sharpe, Sortino, Calmar)
- Actualités du marché
- Allocation d'actifs et secteurs

### 💼 Portfolio Manager

**Onglet 1: Build Portfolio**

Modèles disponibles :
- **Markowitz** - Optimisation Sharpe/Risk/Return/Unsafe
- **Discretionary** - Poids manuels personnalisés
- **Naive** - Equal-weighted (1/n)
- **Beta Weighted** - Basé sur les bêtas
- **ML** - PCA, ICA, HRP (Hierarchical Risk Parity)
- **RL** - REINFORCE, Actor-Critic (Reinforcement Learning)
- **Black-Litterman** - Intégration de vues personnelles

**Onglet 2: My Portfolios**
- Liste de tous vos portfolios sauvegardés
- Métriques en temps réel
- Vue rapide P&L

**Onglet 3: Portfolio Details**
- Holdings avec P&L détaillé
- Asset & Sector allocation (pie charts)
- Analytics avancés (Sharpe, Sortino, Calmar, Max DD, VaR, CVaR)
- Informations détaillées

**Onglet 4: Experiments Lab** 🆕
- **Model Comparison** - Comparez 10+ modèles simultanément
- **Backtesting** - Simple backtest & Walk-forward optimization
- **ML/RL Training** - Entraînez vos modèles avec hyperparamètres
- **Export** - CSV, JSON, PDF (bientôt)

### 📈 Market 

**Onglet 1: oveview**
- Vue global du marché
- indices principaux
- marché principaux


**Onglet 2: explore**
- Données en temps réel (Yahoo Finance)
- Graphiques interactifs (Candlestick, Line, Area)
- Indicateurs techniques (SMA, EMA, RSI, MACD, Bollinger Bands, Volume)
- Comparaison multi-actions
- Informations fondamentales (P/E, Market Cap, Dividendes)
- Actualités intégrées

**Onglet 3: Screen**
Filtres disponibles :
- Regions
- Secteur et industrie
- Capitalisation boursière (Small/Mid/Large cap)
- Ratios financiers (P/E, PEG, P/B, Debt/Equity)
- Rendement dividende
- Performance (YTD, 1M, 3M, 6M, 1Y)
- Volume et liquidité
- Export des résultats

### 🤖 AI Assistant 🆕

**Architecture:**
```
AI Assistant
├── MCP Server (Données Internes)
│   ├── GET: portfolios, transactions, watchlist
│   └── WRITE: save_portfolio, add_to_watchlist (avec confirmation)
├── Claude AI (Analyses & Recommandations)
│   ├── Analyse de portfolio
│   ├── Recherche d'entreprises
│   └── Éducation financière
└── Yahoo Finance (Données de Marché)
    ├── Prix en temps réel
    ├── Infos fondamentales
    └── Recherche de tickers
```

**Capacités:**
- 📊 Analyse de portfolios avec recommandations détaillées
- 🔍 Recherche d'entreprises par nom ou ticker
- 🏗️ Création de portfolios optimisés automatiquement
- 📈 Comparaison d'actions
- 🎓 Éducation financière (knowledge base + web search)
- ⏮️ Backtesting assisté (en développement)
- 💬 Contexte conversationnel (mémorise les échanges)

**Exemples de commandes:**
```
"Analyse mon portfolio et donne des recommandations"
"Recherche Apple et dis-moi si c'est un bon investissement"
"Crée un portfolio growth avec AAPL, MSFT, GOOGL"
"Compare Tesla et Ford"
"Explique-moi le ratio de Sharpe"
```

---

## 🚀 Installation Rapide

### Prérequis

```bash
Python 3.9+
MongoDB 7.0+
Git (optionnel)
```

### Installation en 5 minutes

```bash
# 1. Cloner le projet
git clone https://github.com/baadjis/pymanager.git
cd pymanager

# 2. Environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env
nano .env  # Éditer avec vos clés

# 5. Initialiser la base de données
python database.py

# 6. Créer un utilisateur
python -c "from database import create_user; create_user('admin', 'admin@example.com', 'password123')"

# 7. Démarrer (mode simple)
streamlit run app3.py

# OU avec AI Assistant (mode complet)
./start.sh  # Linux/Mac
start.bat   # Windows
```

Accédez à : **http://localhost:8501**

---

## ⚙️ Configuration

### 1. MongoDB

**Option A: Local**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb-org

# macOS
brew install mongodb-community@7.0

# Windows
# Télécharger depuis mongodb.com/try/download/community
```

**Option B: MongoDB Atlas (Cloud - Gratuit)**
1. Créez un compte sur [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Créez un cluster M0 (gratuit)
3. Obtenez votre connection string
4. Configurez dans `.streamlit/secrets.toml`

### 2. Secrets Configuration

Créez `.streamlit/secrets.toml` :

```toml
# MongoDB
MONGODB_URI = "mongodb://localhost:27017/"
# Ou pour Atlas:
# MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/pymanager_db"

# AI Assistant (optionnel mais recommandé)
ANTHROPIC_API_KEY = "sk-ant-api03-votre-cle-ici"
MCP_SERVER_URL = "http://localhost:8000"
USE_MCP = true
```

**Obtenir une clé Anthropic:**
1. Allez sur https://console.anthropic.com/
2. Créez un compte (gratuit avec crédits)
3. Générez une API key
4. Copiez dans secrets.toml

⚠️ **Ajoutez `secrets.toml` à votre `.gitignore` !**

### 3. Variables d'Environnement (.env)

```env
# MongoDB
MONGO_PASSWORD=your_secure_password

# Anthropic Claude AI
ANTHROPIC_API_KEY=sk-ant-api03-your-key

# Application
STREAMLIT_SERVER_PORT=8501
MCP_SERVER_PORT=8000
```

---

## 📁 Structure du Projet

```
pymanager/
│
├── app3.py                      # 🎯 Point d'entrée principal
├── database/   
|    ├──  __init__.py            # 💾 MongoDB Multi-User + Auth
|    ├── database.py 
|    ├── portfolios.py
|    ├── tasks.py
|    ├── user.py
|    ├── transaction.py
|         
├── mcp/ 
|   ├──  __init__.py             # 🤖 MCP Server (AI Assistant)
|   ├── mcp_server.py  
|   ├── handlers.py
|   ├──helpers.py       
|   ├── test_mcp.py
├── start_server.py            # 🧪 Tests automatiques MCP
├── ml/                          # machine learning module
|   ├──  __init__.py 
|   ├── timeseries_predictors.py                 
│
├── portfolio.py                 # 📊 Classe Portfolio (core)
├── factory.py                   # 🏭 Factory pour créer portfolios
├── viz.py                       # 📈 Visualisations
├── utils.py                     # 🔧 Utilitaires
│
├── uiconfig.py                  # 🎨 Configuration UI
├── styles.py                    # 💅 Styles CSS
├── sidebar_collapsible.py       # 📐 Sidebar navigation
│
├── dataprovider/
│   ├── __init__.py
│   └── yahoo.py                # 📡 Yahoo Finance API
│
├── pagess/                      # 📄 Modules des pages
│   ├── __init__.py
│   ├── auth.py                 # 🔐 Authentification     
|   ├── pricing.py             
│   ├── dashboard.py            # 🏠 Dashboard
│   ├── portfolio_manager/      # 💼 Portfolio Manager
|   |    ├── portfolio_manager.py 
│   │    |
│   |    ├── portfolio_builder.py    # 🏗️ Construction portfolios
│   |    ├── portfolio_helpers.py    # 🔧 Helpers portfolio
│   │    |
│   |    ├── ml_portfolio.py         # 🤖 ML core (PCA, ICA, HRP)
│   |    ├── ml_portfolio_builder.py # 🏗️ ML UI builder
│   |    ├── ml_rl_training.py       # 🎓 ML/RL training UI
│   │    |
│   |    ├── rl_portfolio_simple.py  # 🎮 RL core (REINFORCE, AC)
│   |    ├── rl_portfolio_builder.py # 🏗️ RL UI builder
│   │    |
│   |    ├── bl_portfolio.py         # 📊 Black-Litterman core
│   |    ├── bl_portfolio_builder.py # 🏗️ BL UI builder
│   │    |
│   |    ├── experiments_tab.py      # 🧪 Experiments (backtesting)
│   |    ├── backtesting.py          # ⏮️ Backtesting engine
│   ├──market/
|   |    ├── __init__.py
│   |    ├── market.py               # 📈 Market Explorer
│   |    ├── screener.py             # 🔍 Stock Screener
|   |    ├── explorer.py
│   └──ai/                         # 🤖 AI Assistant
|       ├── __init__.py
|       ├── ai_assistant.py
|       ├── handlers.py
|                         
|                  
│
├── .streamlit/
│   ├── config.toml             # ⚙️ Config Streamlit
│   └── secrets.toml            # 🔑 API Keys (gitignored)
│
├── requirements.txt             # 📦 Dépendances Python
├── .env.example                # 📝 Template variables env
├── .gitignore                  # 🚫 Fichiers à ignorer
│
├── start.sh                    # 🚀 Script démarrage Linux/Mac
├── start.bat                   # 🚀 Script démarrage Windows
│
├── Dockerfile                  # 🐳 Docker image
├── docker-compose.yml          # 🐳 Docker orchestration
├── docker-entrypoint.sh        # 🐳 Docker entrypoint
│
├── docs/                       # 📚 Documentation
│   ├── README_AI_ASSISTANT.md  # 🤖 Guide AI complet
│   └── ...
│
└── tests/                      # 🧪 Tests
    └── ...
```

---

## 🎯 Utilisation

### Démarrage

**Mode Simple (sans AI):**
```bash
streamlit run app3.py
```

**Mode Complet (avec AI Assistant):**

**Terminal 1 - MCP Server:**
```bash
python mcp_server.py
```

**Terminal 2 - Streamlit:**
```bash
streamlit run app3.py
```

**Ou utilisez le script:**
```bash
./start.sh      # Linux/Mac
start.bat       # Windows
```

### Premiers Pas

1. **Connexion** - Créez un compte ou connectez-vous
2. **Dashboard** - Vue d'ensemble (vide au début)
3. **Portfolio Manager** → Build Portfolio
   - Choisissez des tickers (ex: AAPL, MSFT, GOOGL)
   - Sélectionnez un modèle (Markowitz recommandé)
   - Sauvegardez votre portfolio
4. **AI Assistant** - Essayez : "Analyse mon portfolio"

---

## 💬 Guide AI Assistant

### Commandes Disponibles

#### 📊 Analyse de Portfolio
```
"Analyse mon portfolio"
"Donne-moi des recommandations sur mes investissements"
"Mon portfolio est-il bien diversifié?"
```

**Résultat:** Analyse complète avec points forts, faiblesses, recommandations

#### 🔍 Recherche d'Entreprise
```
"Recherche Apple"
"Analyse l'action TSLA"
"Donne-moi des infos sur Microsoft"
```

**Résultat:** Analyse fondamentale, valorisation, recommandation Achat/Vente

#### 🏗️ Création de Portfolio
```
"Crée un portfolio growth avec AAPL, MSFT, GOOGL"
"Construis un portfolio balanced avec 5 actions tech"
```

**Résultat:** Portfolio optimisé avec poids calculés + option de sauvegarde

#### 📈 Comparaison
```
"Compare Apple et Microsoft"
"Tesla vs Ford, laquelle est meilleure?"
```

**Résultat:** Tableau comparatif des métriques

#### 🎓 Éducation
```
"Explique-moi le ratio de Sharpe"
"Qu'est-ce que le modèle Black-Litterman?"
"Comment fonctionne la diversification?"
```

**Résultat:** Explication pédagogique avec exemples

### Human-in-the-Loop

Pour les opérations sensibles (WRITE), l'IA demande confirmation :

```
User: "Crée un portfolio avec AAPL, MSFT"

AI: ✅ Portfolio créé!
    Poids: AAPL 55%, MSFT 45%
    
    💾 Sauvegarder?
    Répondez "Oui, sauvegarde sous [NOM]"

User: "Oui, sauvegarde sous Tech Portfolio"

AI: [Affiche confirmation]
    🔧 Action MCP à exécuter
    save_portfolio(...)
    
    [✅ Confirmer] [❌ Annuler]
```

---

## 🧪 Tests & Validation

### Test Automatique du MCP Server

```bash
# Installation de colorama (si pas déjà fait)
pip install colorama

# Lancer les tests
python test_mcp.py
```

**Résultat attendu:**
```
╔═══════════════════════════════════════════════════╗
║      PyManager MCP Server Tests                   ║
╚═══════════════════════════════════════════════════╝

✓ PASS - Health Check
✓ PASS - List Tools
✓ PASS - Get Portfolios
✓ PASS - Get Transactions
✓ PASS - Get Watchlist
✓ PASS - Calculate Metrics
✓ PASS - Get Allocation
✓ PASS - Validation Errors
✓ PASS - Performance

Results: 11/11 tests passed (100.0%)
🎉 All tests passed!
```

### Tests Manuels

**Test 1: MCP Server Health**
```bash
curl http://localhost:8000/health
```

**Test 2: MCP API Docs**
Ouvrez : http://localhost:8000/docs

**Test 3: Database**
```python
from database import get_portfolios
portfolios = list(get_portfolios("your_user_id"))
print(portfolios)
```

**Test 4: Yahoo Finance**
```python
from dataprovider import yahoo
info = yahoo.get_ticker_info("AAPL")
print(info)
```

---

## 🐳 Déploiement Docker

### Démarrage Rapide Docker

```bash
# 1. Configuration
cp .env.example .env
nano .env  # Éditer avec vos clés

# 2. Build et démarrer
docker-compose up -d

# 3. Vérifier
docker-compose ps
docker-compose logs -f streamlit

# 4. Accéder
# Streamlit: http://localhost:8501
# MCP API: http://localhost:8000/docs
```

### Commandes Docker Utiles

```bash
# Voir les logs
docker-compose logs -f

# Redémarrer un service
docker-compose restart streamlit

# Rebuild après modification
docker-compose up -d --build

# Shell dans un container
docker-compose exec streamlit bash

# Arrêter tout
docker-compose down

# Nettoyer (⚠️ supprime les données)
docker-compose down -v
```

### Services Docker

Le `docker-compose.yml` lance 3 services :

1. **mongodb** - Base de données (port 27017)
2. **mcp-server** - MCP Server (port 8000)
3. **streamlit** - Application UI (port 8501)

---

## 🎨 Personnalisation

### Thème Couleurs

`.streamlit/config.toml` :
```toml
[theme]
primaryColor = "#6366f1"           # Indigo
backgroundColor = "#0f172a"         # Dark slate
secondaryBackgroundColor = "#1e293b"  # Slate
textColor = "#f1f5f9"              # Light
font = "sans serif"
```

### Indicateurs Techniques par Défaut

`pagess/market.py` :
```python
DEFAULT_INDICATORS = ['SMA_50', 'SMA_200', 'RSI', 'MACD']
```

### Knowledge Base AI

`pagess/ai_assistant.py` :
```python
KNOWLEDGE_BASE = {
    "votre_sujet": {
        "title": "Votre Titre",
        "content": """Votre contenu..."""
    }
}
```

---

## 🐛 Troubleshooting

### Problème: MCP Server Offline

**Symptôme:** 🔴 MCP Server Offline dans sidebar

**Solutions:**
1. Vérifier que le serveur tourne :
```bash
python mcp_server.py
```

2. Tester la connexion :
```bash
curl http://localhost:8000/health
```

3. Vérifier `secrets.toml` :
```toml
MCP_SERVER_URL = "http://localhost:8000"  # Pas de slash final
USE_MCP = true
```

### Problème: Claude AI ne répond pas

**Symptôme:** "⚙️ Configurez ANTHROPIC_API_KEY"

**Solutions:**
1. Vérifier la clé dans `secrets.toml`
2. Tester la clé :
```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.content[0].text)
```

3. Obtenir une nouvelle clé sur https://console.anthropic.com/

### Problème: MongoDB Connection Failed

**Symptôme:** "pymongo.errors.ServerSelectionTimeoutError"

**Solutions:**
1. Vérifier MongoDB actif :
```bash
# Linux
systemctl status mongod

# macOS
brew services list | grep mongodb

# Windows
sc query MongoDB
```

2. Tester connexion :
```python
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
print(client.server_info())
```

3. Vérifier URI dans `secrets.toml`

### Problème: Port Already in Use

**Symptôme:** "Address already in use"

**Solutions:**
```bash
# Linux/Mac - Trouver le processus
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr ":8501"
taskkill /F /PID <PID>

# Ou changer le port
streamlit run app3.py --server.port 8502
```

### Plus de Solutions

Consultez la documentation complète : `docs/README_AI_ASSISTANT.md`

---

## 📚 Documentation

### Guides Principaux

- **[AI Assistant Guide](docs/README_AI_ASSISTANT.md)** - Documentation complète AI
- **[Installation Guide](#installation-rapide)** - Installation pas à pas
- **[Configuration Guide](#configuration)** - Configuration détaillée
- **[Docker Guide](#déploiement-docker)** - Déploiement Docker

### API Documentation

- **MCP Server API:** http://localhost:8000/docs (quand le serveur tourne)
- **[Yahoo Finance API](https://github.com/ranaroussi/yfinance)** - Documentation yfinance
- **[Streamlit API](https://docs.streamlit.io/)** - Documentation Streamlit
- **[Claude API](https://docs.anthropic.com/)** - Documentation Anthropic



---

## 🗺️ Roadmap

### ✅ Version 1.0 (Actuelle)
- Dashboard complet
- Portfolio Manager (10+ modèles)
- Market Explorer
- Screener
- AI Assistant avec MCP
- Experiments Lab
- Auth multi-utilisateurs

### 🚧 Version 1.1 (Court terme - Q1 2025)
- [ ] Alertes en temps réel
- [ ] Watchlist interactive améliorée
- [ ] Export PDF avec graphiques
- [ ] Mode hors ligne avec cache
- [ ] AI: Comparaison multi-actions
- [ ] AI: Backtesting assisté

### 🎯 Version 1.2 (Moyen terme - Q2 2025)
- [ ] Options et dérivés
- [ ] Calendrier économique
- [ ] Analyse de corrélation avancée
- [ ] Portfolio optimization multi-objectifs
- [ ] AI: Génération de rapports PDF

### 🚀 Version 2.0 (Long terme - 2025)
- [ ] Trading automatisé (paper trading)
- [ ] Mobile app (iOS/Android)
- [ ] Intégration courtiers (Alpaca, IB)
- [ ] Social features (partage stratégies)
- [ ] API publique
- [ ] Marketplace de stratégies

---

## 🤝 Contribution

### Comment Contribuer

1. **Fork** le projet
2. Créez une **feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit**: `git commit -m 'Add AmazingFeature'`
4. **Push**: `git push origin feature/AmazingFeature`
5. Ouvrez une **Pull Request**

### Guidelines

- **Code:** Suivre PEP 8
- **Docstrings:** Pour toutes les fonctions
- **Tests:** Pour les nouvelles features
- **Documentation:** Mettre à jour README si nécessaire

### Idées de Contribution

**Facile** (Good First Issue):
- 📝 Améliorer la documentation
- 🐛 Corriger des bugs mineurs
- 🌍 Traduire l'interface
- 🎨 Améliorer le design

**Moyen:**
- 📊 Nouveaux indicateurs techniques
- 🔧 Nouvelles sources de données
- 🧪 Tests automatisés
- 📈 Nouveaux graphiques

**Avancé:**
- 🤖 Nouveaux modèles ML/RL
- 🔄 Intégration courtiers
- 📱 Version mobile
- 🚀 Optimisations performance

---

## 📊 Technologies Utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Frontend** | Streamlit 1.28+, Plotly 5.17+, HTML/CSS (Glassmorphism) |
| **Backend** | Python 3.9+, FastAPI 0.104+ (MCP Server) |
| **Database** | MongoDB 7.0+ (PyMongo 4.5+) |
| **Data Processing** | pandas 2.0+, numpy 1.24+ |
| **Financial Data** | yfinance 0.2.28+ |
| **AI/ML** | Anthropic Claude API, scikit-learn, PyTorch (RL) |
| **Optimization** | scipy, cvxpy (Markowitz) |
| **Visualization** | Plotly, Matplotlib |
| **Testing** | requests, colorama, pytest (planned) |
| **Deployment** | Docker, docker-compose, nginx (optional) |

---

## 📄 License

MIT License - voir [LICENSE](LICENSE)

```
MIT License

Copyright (c) 2024 PyManager Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Remerciements

### Librairies Open Source

- **[yfinance](https://github.com/ranaroussi/yfinance)** - Données Yahoo Finance
- **[Streamlit](https://streamlit.io/)** - Framework UI incroyable
- **[Anthropic](https://www.anthropic.com/)** - Claude AI
- **[Plotly](https://plotly.com/)** - Graphiques interactifs
- **[MongoDB](https://www.mongodb.com/)** - Base de données NoSQL
- **[FastAPI](https://fastapi.tiangolo.com/)** - Framework API moderne

### Contributeurs

Merci à tous les contributeurs du projet ! 🎉

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- Liste générée automatiquement -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## 📞 Support & Contact

### 💬 Support

- **Issues GitHub:** [github.com/baadjis/pymanager/issues](https://github.com/baadjis/pymanager/issues)
- **Discussions:** [github.com/baadjis/pymanager/discussions](https://github.com/baadjis/pymanager/discussions)
- **Documentation:** `docs/` folder
- **Email:** support@pymanager.dev (si configuré)

### 🌐 Liens

- **GitHub:** [@baadjis](https://github.com/baadjis)
- **Project:** [github.com/baadjis/pymanager](https://github.com/baadjis/pymanager)
- **Discord:** [Lien à venir]
- **Website:** [À venir]

---

## ⭐ Star History

Si vous aimez PyManager, mettez une étoile sur GitHub ! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=baadjis/pymanager&type=Date)](https://star-history.com/#baadjis/pymanager&Date)

---

## 📈 Statistiques

![GitHub stars](https://img.shields.io/github/stars/baadjis/pymanager?style=social)
![GitHub forks](https://img.shields.io/github/forks/baadjis/pymanager?style=social)
![GitHub issues](https://img.shields.io/github/issues/baadjis/pymanager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/baadjis/pymanager)
![GitHub last commit](https://img.shields.io/github/last-commit/baadjis/pymanager)

---

## 🎯 Philosophie du Projet

**ΦManager** (Phi Manager) tire son nom de la lettre grecque **Φ** (Phi), symbole du **nombre d'or** et de l'**harmonie parfaite**.

Notre vision :

- 🎨 **Design Élégant** - Interface intuitive et visuellement agréable
- 🧠 **Intelligence Accessible** - IA pour démocratiser l'analyse financière
- 🔓 **Open Source** - Transparence et collaboration
- 📊 **Données Ouvertes** - Accès gratuit aux données de marché
- 🚀 **Innovation Continue** - Technologies modernes et performantes
- 🌍 **Finance pour Tous** - Rendre l'investissement accessible

Le nombre d'or (≈1.618) représente l'équilibre parfait, tout comme un portfolio optimal.

---

<div align="center">

**Construit avec ❤️ par la communauté**

[⭐ Star](https://github.com/baadjis/pymanager) • [🐛 Bug Report](https://github.com/baadjis/pymanager/issues) • [✨ Feature Request](https://github.com/baadjis/pymanager/issues)

---

### 🎓 Citation

Si vous utilisez PyManager dans votre recherche ou projet, veuillez citer :

```bibtex
@software{pymanager2024,
  title = {ΦManager: AI-Powered Portfolio Management Platform},
  author = {PyManager Contributors},
  year = {2024},
  url = {https://github.com/baadjis/pymanager},
  version = {1.0.0}
}
```

---

### 🌟 Showcase

Votre projet utilise PyManager ? Montrez-le nous !

Créez une issue avec le tag `showcase` pour être featured ici.

---

### 📺 Captures d'Écran

#### Dashboard
![Dashboard](docs/images/dashboard.png)

#### Portfolio Manager
![Portfolio Manager](docs/images/portfolio-manager.png)

#### AI Assistant
![AI Assistant](docs/images/ai-assistant.png)

#### Market Explorer
![Market Explorer](docs/images/market-explorer.png)

#### Experiments Lab
![Experiments Lab](docs/images/experiments-lab.png)

---

### 🎬 Demo Video

[![PyManager Demo](https://img.youtube.com/vi/DEMO_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)

---

### 🔗 Liens Rapides

| Ressource | Lien |
|-----------|------|
| 📖 Documentation | [docs/](docs/) |
| 🤖 AI Assistant Guide | [README_AI_ASSISTANT.md](docs/README_AI_ASSISTANT.md) |
| 🐛 Report Bug | [Issues](https://github.com/baadjis/pymanager/issues) |
| 💡 Request Feature | [Issues](https://github.com/baadjis/pymanager/issues) |
| 💬 Discussions | [Discussions](https://github.com/baadjis/pymanager/discussions) |
| 🚀 Changelog | [CHANGELOG.md](CHANGELOG.md) |
| 📜 License | [LICENSE](LICENSE) |

---

### 📋 Checklist de Démarrage

Après installation, vérifiez que tout fonctionne :

- [ ] MongoDB connecté (`python -c "from database import get_portfolios; print('OK')"`)
- [ ] Streamlit lance (`streamlit run app3.py`)
- [ ] MCP Server actif (`curl http://localhost:8000/health`)
- [ ] Yahoo Finance fonctionne (testez dans Market Explorer)
- [ ] Claude AI configuré (testez dans AI Assistant)
- [ ] Utilisateur créé et connexion possible
- [ ] Premier portfolio créé avec succès
- [ ] Tests MCP passent (`python test_mcp.py`)

---

### 🎯 Quick Start Commands

```bash
# Installation complète
git clone https://github.com/baadjis/pymanager.git && cd pymanager && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Démarrage rapide (après installation)
./start.sh

# Avec Docker
docker-compose up -d && docker-compose logs -f

# Tests
python test_mcp.py

# Nettoyage
docker-compose down -v  # Docker
deactivate && rm -rf venv  # Local
```

---

### 💡 Tips & Astuces

**Performance:**
- Utilisez le cache Streamlit pour les données fréquentes
- Limitez l'historique des données à ce qui est nécessaire
- Activez la compression dans MongoDB

**Sécurité:**
- Ne commitez JAMAIS les secrets.toml ou .env
- Changez les mots de passe par défaut
- Utilisez HTTPS en production
- Limitez l'accès à MongoDB

**Development:**
- Utilisez `streamlit run app3.py --server.runOnSave true` pour hot reload
- Activez les logs debug : `--logger.level=debug`
- Utilisez MongoDB Compass pour explorer la DB visuellement

**Production:**
- Utilisez MongoDB Atlas pour la scalabilité
- Configurez un reverse proxy (nginx)
- Activez les backups automatiques MongoDB
- Monitorer avec Grafana + Prometheus

---

### 🔐 Sécurité

#### Reporting Security Issues

Si vous découvrez une faille de sécurité, **NE PAS** créer une issue publique.

Contactez-nous directement : security@pymanager.dev

#### Security Best Practices

1. **API Keys**
   - Stockez dans `.streamlit/secrets.toml` (gitignored)
   - Utilisez des variables d'environnement en production
   - Régénérez régulièrement les clés

2. **MongoDB**
   - Activez l'authentification
   - Utilisez des mots de passe forts
   - Limitez les IPs autorisées (whitelist)
   - Chiffrez les connexions (TLS/SSL)

3. **Application**
   - Validez toutes les entrées utilisateur
   - Utilisez HTTPS en production
   - Implémentez rate limiting
   - Logs des actions critiques

---

### 🏆 Contributors

Un grand merci à tous nos contributeurs !

<a href="https://github.com/baadjis/pymanager/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=baadjis/pymanager" />
</a>

Rejoignez-nous et contribuez ! 🚀

---

### 📊 Project Stats

- **Lines of Code:** ~15,000+
- **Files:** 50+
- **Models Supported:** 10+ (Markowitz, ML, RL, BL, etc.)
- **Indicators:** 20+ (Technical Analysis)
- **Languages:** Python 100%
- **First Release:** December 2024
- **Latest Version:** 1.0.0

---

### 🌍 Community

Rejoignez notre communauté grandissante !

- **GitHub Stars:** ![Stars](https://img.shields.io/github/stars/baadjis/pymanager?style=social)
- **Forks:** ![Forks](https://img.shields.io/github/forks/baadjis/pymanager?style=social)
- **Contributors:** ![Contributors](https://img.shields.io/github/contributors/baadjis/pymanager)
- **Discord:** [Lien à venir]
- **Twitter:** [@pymanager](#) (exemple)

---

### 🎉 Sponsoring

Aimez PyManager ? Supportez le projet !

- ⭐ **Star le projet** sur GitHub
- 🐛 **Reportez des bugs** pour améliorer la qualité
- 📖 **Contribuez à la documentation**
- 💻 **Contribuez au code**


---

### 📅 Changelog

#### Version 1.0.0 (December 2024)

**✨ Features:**
- 🎯 Multi-portfolio management
- 📊 10+ optimization models (Markowitz, ML, RL, BL)
- 🤖 AI Assistant with Claude API
- 🔧 MCP Server for internal data
- 📈 Advanced technical indicators
- 🧪 Experiments Lab (backtesting, model comparison)
- 🔐 Multi-user authentication
- 🎨 Modern glassmorphism UI

**🐛 Bug Fixes:**
- Fixed MongoDB connection issues
- Improved error handling in portfolio calculations
- Fixed ticker search with special characters
- Corrected timezone handling for market data

**📚 Documentation:**
- Complete README with all features
- AI Assistant integration guide
- Docker deployment guide
- Test suite documentation

---

### 🎓 Learning Resources

**Pour les Débutants:**
- [Introduction à l'investissement](docs/guides/investing-101.md) (à venir)
- [Comprendre les ratios financiers](docs/guides/financial-ratios.md) (à venir)
- [Guide des indicateurs techniques](docs/guides/technical-indicators.md) (à venir)

**Pour les Avancés:**
- [Théorie Moderne du Portfolio](docs/guides/modern-portfolio-theory.md) (à venir)
- [Machine Learning en Finance](docs/guides/ml-finance.md) (à venir)
- [Reinforcement Learning pour Trading](docs/guides/rl-trading.md) (à venir)

**Tutoriels Vidéo:**
- [ ] Installation complète (10 min)
- [ ] Créer son premier portfolio (5 min)
- [ ] Utiliser l'AI Assistant (8 min)
- [ ] Backtesting avancé (15 min)
- [ ] Déployer en production (12 min)

---

### 🔮 Vision Future

Notre vision pour PyManager :

1. **Plateforme Complète** - Couvrir tous les aspects de l'investissement
2. **Intelligence Avancée** - IA qui apprend de vos décisions
3. **Communauté Active** - Partage de stratégies et d'insights
4. **Open & Transparent** - Code source ouvert et auditable
5. **Accessible à Tous** - Gratuit et facile à utiliser

**Long terme (2025-2026):**
- Marketplace de stratégies
- Trading automatisé régulé
- Mobile apps natives
- API publique complète
- Intégration courtiers majeurs

---

### ⚖️ Disclaimer

**PyManager est un outil éducatif et d'analyse.**

⚠️ **Important:**
- Ce logiciel est fourni "tel quel" sans garantie
- Ne constitue PAS un conseil financier
- Les performances passées ne garantissent pas les résultats futurs
- Investir comporte des risques de perte en capital
- Consultez toujours un conseiller financier professionnel
- Les créateurs ne sont pas responsables des pertes financières

**Utilisation à vos propres risques.**

---

### 📜 Legal

**Copyright © 2024 PyManager Contributors**

Ce projet est sous licence MIT. Vous êtes libre de :
- ✅ Utiliser commercialement
- ✅ Modifier
- ✅ Distribuer
- ✅ Utiliser en privé

Conditions :
- 📋 Inclure la licence et le copyright
- 📋 Inclure l'avis de non-responsabilité

Voir [LICENSE](LICENSE) pour les détails complets.

---

### 🎊 Final Words

Merci d'utiliser **ΦManager** ! 

Nous espérons que cette plateforme vous aidera à :
- 📈 Optimiser vos investissements
- 🧠 Apprendre la finance
- 🤝 Partager avec la communauté
- 💰 Atteindre vos objectifs financiers

**Ensemble, rendons la finance accessible à tous !** 🌍

---

<div align="center">

**ΦManager** - Your Portfolio, Perfectly Balanced

![Phi Symbol](https://via.placeholder.com/100x100/6366f1/ffffff?text=Φ)



[⭐ Star](https://github.com/baadjis/pymanager) • [🍴 Fork](https://github.com/baadjis/pymanager/fork) • [📖 Docs](docs/) • [💬 Chat](https://github.com/baadjis/pymanager/discussions)

</div>

---

<div align="center">
<sub>Built with Python 🐍 • Streamlit 🎈 • MongoDB 🍃 • Claude AI 🤖</sub>
</div>
