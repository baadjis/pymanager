# 📊 ΦManager - Portfolio & Market Intelligence Platform

> Plateforme moderne de gestion de portefeuille avec intelligence artificielle intégrée

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 À propos

**ΦManager** (Phi Manager) est une plateforme complète de gestion de portefeuille et d'analyse de marché, combinant:

- 📊 **Gestion de Portfolio** - Créez et suivez plusieurs portefeuilles
- 📈 **Analyse de Marché** - Données en temps réel et graphiques interactifs
- 🤖 **AI Assistant** - Conseiller intelligent avec Claude AI
- 🔍 **Stock Explorer** - Recherche et analyse approfondie d'actions
- 📉 **Stock Screener** - Filtrez les actions selon vos critères
- 📱 **Interface Moderne** - Design responsive avec thème sombre élégant

---

## ✨ Fonctionnalités

### 🏠 Dashboard
- Vue d'ensemble de tous vos portefeuilles
- Graphiques de performance en temps réel
- Métriques clés (rendement, volatilité, Sharpe ratio)
- Actualités du marché
- Positions principales et allocation d'actifs

### 💼 Portfolio Manager
- **Multi-portfolio** - Gérez plusieurs stratégies simultanément
- **Modèles de gestion** - Growth, Income, Balanced
- **Tracking en temps réel** - Prix et valorisation actualisés
- **Historique complet** - Toutes vos transactions
- **Métriques avancées** - Performance, risque, diversification
- **Export de données** - CSV, Excel

### 📈 Market Explorer
- **Données en temps réel** - Prix, volumes, variations
- **Graphiques interactifs** - Candlestick, lignes, aires
- **Indicateurs techniques** - SMA, EMA, RSI, MACD, Bollinger Bands
- **Comparaison multi-actions** - Analysez plusieurs titres
- **Informations fondamentales** - P/E, market cap, dividendes
- **Actualités intégrées** - News liées aux actions suivies

### 🔍 Stock Screener
- **Filtres personnalisables**:
  - Secteur et industrie
  - Capitalisation boursière
  - Ratios financiers (P/E, PEG, P/B)
  - Rendement dividende
  - Performance YTD, 1M, 3M, 1Y
  - Volume et liquidité
- **Résultats en temps réel**
- **Tri et export** des résultats
- **Sauvegarde de filtres** favoris

### 🤖 AI Assistant
- **Agent conversationnel** alimenté par Claude AI
- **Analyse de portfolio** automatique
- **Recherche d'entreprises** approfondie
- **Recommandations personnalisées**
- **Éducation financière** interactive
- **Screening intelligent** guidé
- **Génération de rapports** détaillés
- **Accès aux données internes** via MCP (Model Context Protocol)

---

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- MongoDB 7.0+ (local ou Atlas)
- pip (gestionnaire de packages Python)
- Git (optionnel)

### Étape 1: Cloner le projet

```bash
# Via Git
git clone https://github.com/baadjis/pymanager.git
cd pymanager

# Ou télécharger le ZIP depuis GitHub
```

### Étape 2: Installer MongoDB

**Option A: MongoDB local**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community@7.0

# Windows
# Télécharger depuis https://www.mongodb.com/try/download/community
```

**Option B: MongoDB Atlas (Cloud - Gratuit)**
1. Créez un compte sur [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Créez un cluster gratuit (M0)
3. Obtenez votre connection string
4. Configurez dans `.streamlit/secrets.toml`

### Étape 3: Créer un environnement virtuel (recommandé)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Étape 4: Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- streamlit - Framework UI
- pymongo - Driver MongoDB
- pandas, numpy - Traitement de données
- plotly - Graphiques interactifs
- yfinance - Données de marché Yahoo Finance
- anthropic - API Claude AI (optionnel)
- fastapi, uvicorn - Serveur MCP (optionnel)

### Étape 5: Initialiser la base de données

```bash
# Créer les collections et indexes
python database.py
```

Cela crée:
- Collection `users` avec indexes sur email/username
- Collection `portfolios` avec indexes
- Collection `watchlists`
- Collection `alerts`
- Collection `transactions`

### Étape 6: Configuration

#### Configuration de base

Créez `.streamlit/secrets.toml`:

```toml
# MongoDB Connection
MONGODB_URI = "mongodb://localhost:27017/"
# Ou pour Atlas:
# MONGODB_URI = "mongodb+srv://user:pass@cluster.mongodb.net/pymanager_db"

# AI Assistant (optionnel)
ANTHROPIC_API_KEY = "sk-ant-votre-cle-ici"
MCP_SERVER_URL = "http://localhost:8000"
```

⚠️ **Important**: Ajoutez `secrets.toml` à votre `.gitignore` !

#### Configuration MongoDB Atlas

Si vous utilisez MongoDB Atlas:

1. **Connection string** format:
```
mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority
```

2. **Whitelist IP**: Ajoutez votre IP dans Network Access
3. **Database User**: Créez un utilisateur avec droits ReadWrite

### Étape 7: Migration (si vous avez une ancienne DB)

Si vous migrez depuis l'ancien schéma single-user:

```bash
# Créer une backup d'abord!
python migrate_to_multiuser.py
```

Le script:
- ✅ Crée un utilisateur par défaut (admin/admin123)
- ✅ Migre tous les portfolios
- ✅ Migre la watchlist
- ✅ Migre les transactions
- ✅ Crée une backup automatique

### Étape 8: Créer un utilisateur

```python
# Via Python shell
from database import create_user

user_id = create_user(
    username="votre_nom",
    email="votre@email.com",
    password="votre_mot_de_passe",
    first_name="Prénom",
    last_name="Nom"
)
print(f"User créé: {user_id}")
```

### Étape 9: Lancer l'application

#### Sans AI Assistant (basique)

```bash
streamlit run app3.py
```

Ouvrez votre navigateur sur: `http://localhost:8501`

#### Avec AI Assistant (complet)

**Terminal 1 - Serveur MCP:**
```bash
python mcp_server.py
```

**Terminal 2 - Application Streamlit:**
```bash
streamlit run app3.py
```

#### Script de démarrage automatique

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```bash
start.bat
```

#### Docker (Production)

```bash
# Build et lancer
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down
```

---

## 📁 Structure du projet

```
pymanager/
│
├── app3.py                      # Point d'entrée principal
├── database.py                  # 🆕 MongoDB Multi-User
├── migrate_to_multiuser.py     # 🆕 Script de migration
├── mcp_server.py               # Serveur MCP pour AI Assistant
├── uiconfig.py                  # Configuration UI et session
├── styles.py                    # Styles CSS personnalisés
├── sidebar_collapsible.py       # Sidebar avec navigation
├── requirements.txt             # 🆕 Dépendances (avec pymongo)
│
├── .streamlit/
│   ├── config.toml             # Configuration Streamlit
│   └── secrets.toml            # 🆕 Clés API + MongoDB URI
│
├── dataprovider/
│   ├── __init__.py
│   └── yahoo.py                # Interface Yahoo Finance API
│
├── pagess/                      # Modules des pages
│   ├── __init__.py
│   ├── dashboard.py            # Page Dashboard
│   ├── portfolio.py            # Page Portfolio Manager
│   ├── market.py               # Page Market Explorer
│   ├── screener.py             # Page Stock Screener
│   └── ai_assistant.py         # 🆕 Page AI Assistant
│
├── docs/                        # Documentation
│   ├── README_AI_INTEGRATION.md
│   ├── CI_CD_SETUP.md          # 🆕 Guide CI/CD
│   └── ...
│
├── tests/                       # 🆕 Tests
│   ├── conftest.py
│   ├── test_database.py
│   ├── test_auth.py
│   └── ...
│
├── .github/                     # 🆕 CI/CD
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── security.yml
│
├── docker-compose.yml           # 🆕 Docker orchestration
├── Dockerfile                   # 🆕 Container principal
└── .gitignore                  # Fichiers à ignorer
```

---

## 💻 Utilisation

### Créer un portfolio

1. Allez sur **Portfolio** dans la sidebar
2. Cliquez sur **"Créer un nouveau portfolio"**
3. Choisissez un nom et un modèle de gestion:
   - **Growth** - Croissance du capital
   - **Income** - Génération de revenus
   - **Balanced** - Équilibre risque/rendement
4. Ajoutez vos actions avec quantités et prix
5. Suivez la performance en temps réel

### Explorer le marché

1. Allez sur **Market** dans la sidebar
2. Entrez un symbole boursier (ex: AAPL, MSFT, TSLA)
3. Sélectionnez la période d'analyse
4. Choisissez les indicateurs techniques
5. Analysez les graphiques et métriques

### Utiliser le screener

1. Allez sur **Stock Screener**
2. Définissez vos critères de filtrage:
   - Secteur, capitalisation
   - Ratios financiers
   - Performance passée
3. Cliquez **"Lancer la recherche"**
4. Triez et exportez les résultats

### Interroger l'AI Assistant

1. Allez sur **AI Assistant**
2. Tapez votre question en langage naturel:

**Exemples:**
```
"Analyse mon portfolio et donne-moi des recommandations"
"Recherche Apple (AAPL) et dis-moi si c'est un bon investissement"
"Trouve-moi des actions technologiques avec P/E < 20"
"Explique-moi le ratio de Sharpe"
"Génère un rapport de performance"
```

3. L'AI répond avec analyses et recommandations
4. Suivez les suggestions ou posez des questions de suivi

---

## 🎨 Personnalisation

### Thème

Modifiez les couleurs dans `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6366f1"      # Couleur principale
backgroundColor = "#0f172a"    # Fond de page
secondaryBackgroundColor = "#1e293b"  # Fond widgets
textColor = "#f1f5f9"         # Couleur du texte
```

### Indicateurs par défaut

Dans `pagess/market.py`, modifiez la liste des indicateurs:

```python
DEFAULT_INDICATORS = ['SMA_50', 'SMA_200', 'RSI', 'MACD']
```

### Modèles de portfolio

Dans `pagess/portfolio.py`, ajoutez vos propres modèles:

```python
PORTFOLIO_MODELS = {
    'growth': {'description': '...', 'allocation': {...}},
    'income': {'description': '...', 'allocation': {...}},
    'balanced': {'description': '...', 'allocation': {...}},
    # Ajoutez le vôtre
    'aggressive': {'description': '...', 'allocation': {...}}
}
```

---

## 🔧 Configuration avancée

### Base de données

Par défaut, SQLite est utilisé (`pymanager.db`).

Pour PostgreSQL:

```python
# Dans database.py
DATABASE_URL = "postgresql://user:password@localhost/pymanager"
```

### Cache

Activez le cache Streamlit pour de meilleures performances:

```python
@st.cache_data(ttl=300)  # Cache 5 minutes
def get_market_data(ticker):
    return yahoo.get_ticker_data(ticker)
```

### Sources de données

Ajoutez d'autres sources dans `dataprovider/`:

```python
# dataprovider/alpha_vantage.py
class AlphaVantageProvider:
    def get_ticker_data(self, ticker):
        # Votre implémentation
        pass
```

---

## 🧪 Tests

### Tests manuels

```bash
# Test base de données
python database.py

# Test Yahoo Finance
python -c "from dataprovider import yahoo; print(yahoo.get_ticker_info('AAPL'))"

# Test MCP Server
curl http://localhost:8000/
```

### Tests automatisés

```bash
# Tests AI Assistant
python tests/test_mcp_integration.py --full

# Tests unitaires (si pytest installé)
pytest tests/
```

---

## 🐛 Dépannage

### Problème: "Module not found"

```bash
# Réinstaller les dépendances
pip install -r requirements.txt

# Vérifier l'installation
pip list
```

### Problème: "Database locked"

```bash
# Fermer toutes les instances de l'app
# Supprimer le fichier de lock
rm pymanager.db-wal pymanager.db-shm

# Relancer
streamlit run app3.py
```

### Problème: Données Yahoo Finance ne se chargent pas

```bash
# Mettre à jour yfinance
pip install --upgrade yfinance

# Vérifier la connexion
ping finance.yahoo.com
```

### Problème: Port déjà utilisé

```bash
# Linux/Mac
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr ":8501"
taskkill /F /PID <PID>

# Ou utiliser un autre port
streamlit run app3.py --server.port 8502
```

### Problème: AI Assistant ne répond pas

1. Vérifiez la clé API dans `.streamlit/secrets.toml`
2. Vérifiez que le serveur MCP tourne (si activé)
3. Consultez les logs du serveur MCP
4. Testez la connexion API:

```python
import anthropic
client = anthropic.Anthropic(api_key="votre-cle")
# Si erreur, la clé est invalide
```

---

## 📚 Documentation

- **[Guide d'intégration AI](docs/README_AI_INTEGRATION.md)** - Configuration complète de l'AI Assistant
- **[API Yahoo Finance](https://github.com/ranaroussi/yfinance)** - Documentation yfinance
- **[Streamlit Docs](https://docs.streamlit.io/)** - Documentation Streamlit
- **[Claude API](https://docs.anthropic.com/)** - Documentation Anthropic

---

## 🤝 Contribution

Les contributions sont les bienvenues! Voici comment participer:

### Pour contribuer

1. **Fork** le projet
2. Créez votre **feature branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Committez** vos changements:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** vers la branche:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Ouvrez une **Pull Request**

### Guidelines

- Code Python: Suivre PEP 8
- Docstrings pour toutes les fonctions
- Tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation

### Idées de contribution

- 🔧 Nouvelles sources de données (Alpha Vantage, IEX Cloud)
- 📊 Nouveaux indicateurs techniques
- 🎨 Thèmes personnalisés
- 🌍 Internationalisation (i18n)
- 📱 Version mobile optimisée
- 🧪 Tests automatisés
- 📖 Tutoriels et exemples

---

## 🗺️ Roadmap

### Version actuelle (v1.0)
- ✅ Dashboard avec métriques principales
- ✅ Gestion multi-portfolio
- ✅ Exploration de marché avec indicateurs techniques
- ✅ Stock screener avec filtres avancés
- ✅ AI Assistant avec Claude AI
- ✅ Base de données SQLite

### Prochaines versions

#### v1.1 (Court terme)
- [ ] Alertes en temps réel (prix, variations)
- [ ] Watchlist améliorée avec notifications
- [ ] Export avancé (PDF, Excel avec graphiques)
- [ ] Mode hors ligne avec cache

#### v1.2 (Moyen terme)
- [ ] Backtesting de stratégies
- [ ] Optimisation de portfolio (Markowitz)
- [ ] Analyse de corrélation entre actifs
- [ ] Calendrier économique intégré

#### v2.0 (Long terme)
- [ ] Support des options et dérivés
- [ ] Trading automatisé (paper trading)
- [ ] Application mobile (iOS/Android)
- [ ] Intégration courtiers (Alpaca, Interactive Brokers)
- [ ] Fonctionnalités sociales (partage de stratégies)
- [ ] API publique pour développeurs

---

## 📊 Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Frontend** | Streamlit, Plotly, HTML/CSS |
| **Backend** | Python, FastAPI (MCP Server) |
| **Base de données** | SQLite (PostgreSQL compatible) |
| **Data** | pandas, numpy, yfinance |
| **AI** | Anthropic Claude API |
| **Charts** | Plotly, Matplotlib |
| **Testing** | pytest, requests |

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2024 PyManager

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Remerciements

- **[yfinance](https://github.com/ranaroussi/yfinance)** - Données de marché Yahoo Finance
- **[Streamlit](https://streamlit.io/)** - Framework UI incroyable
- **[Anthropic](https://www.anthropic.com/)** - Claude AI
- **[Plotly](https://plotly.com/)** - Graphiques interactifs
- **Communauté Open Source** - Pour tous les packages utilisés

---

## 📞 Support & Contact

### Bugs et demandes de fonctionnalités

Ouvrez une issue sur GitHub: [github.com/baadjis/pymanager/issues](https://github.com/baadjis/pymanager/issues)

### Questions

- **Discussions GitHub**: Pour les questions générales
- **Email**: support@pymanager.dev (si configuré)

### Réseaux sociaux

- GitHub: [@baadjis](https://github.com/baadjis)
- Twitter: [@pymanager](https://twitter.com/pymanager) (exemple)

---

## ⭐ Star History

Si vous aimez PyManager, n'hésitez pas à mettre une étoile sur GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=baadjis/pymanager&type=Date)](https://star-history.com/#baadjis/pymanager&Date)

---

## 📈 Statistiques

![GitHub stars](https://img.shields.io/github/stars/baadjis/pymanager?style=social)
![GitHub forks](https://img.shields.io/github/forks/baadjis/pymanager?style=social)
![GitHub issues](https://img.shields.io/github/issues/baadjis/pymanager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/baadjis/pymanager)

---

## 🎯 Philosophie du projet

**ΦManager** (Phi Manager) tire son nom de la lettre grecque Φ (Phi), symbole du nombre d'or et de l'harmonie parfaite. Notre philosophie:

- 🎨 **Design élégant** - Interface intuitive et visuellement agréable
- 🧠 **Intelligence** - IA pour démocratiser l'analyse financière
- 🔓 **Open Source** - Transparence et collaboration
- 📊 **Données ouvertes** - Accès gratuit aux données de marché
- 🚀 **Innovation** - Technologies modernes et performantes
- 🌍 **Accessibilité** - Finance pour tous

---

<div align="center">

**Construit avec ❤️ par la communauté**

[⭐ Star](https://github.com/baadjis/pymanager) • [🐛 Report Bug](https://github.com/baadjis/pymanager/issues) • [✨ Request Feature](https://github.com/baadjis/pymanager/issues)

**ΦManager** - Your Portfolio, Perfectly Balanced

</div>
