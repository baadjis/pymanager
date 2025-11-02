# ⚡ PyManager - Quick Reference

Aide-mémoire rapide pour PyManager - Toutes les commandes essentielles.

---

## 🚀 Démarrage rapide

### Sans Docker (Recommandé pour dev)

```bash
# Setup initial
python3 -m venv venv
venv/bin/pip install -r requirements.txt
chmod +x start.sh

# Démarrer
./start.sh dev

# Accéder
open http://localhost:8501
```

### Avec Docker (Production)

```bash
# Setup initial
cp .env.example .env
nano .env  # Configurer

# Démarrer
make up

# Accéder
open http://localhost:8501
```

---

## 📝 Commandes start.sh

| Commande | Description |
|----------|-------------|
| `./start.sh dev` | Démarrer mode dev |
| `./start.sh test` | Démarrer mode test |
| `./start.sh prod` | Démarrer mode production |
| `./start.sh dev status` | Vérifier statut |
| `./start.sh dev stop` | Arrêter services |
| `./start.sh dev logs` | Voir les logs |
| `Ctrl+C` | Arrêter (cleanup auto) |

---

## 🐳 Commandes Docker (Makefile)

### Gestion services

| Commande | Description |
|----------|-------------|
| `make up` | Démarrer |
| `make down` | Arrêter |
| `make restart` | Redémarrer |
| `make status` | Statut |
| `make logs` | Voir logs |
| `make shell` | Shell container |

### Nettoyage (économiser espace)

| Commande | Description |
|----------|-------------|
| `make clean` | Nettoyer containers/volumes |
| `make prune` | ⚠️ Nettoyage complet Docker |
| `make prune-images` | Supprimer images non utilisées |
| `make disk-usage` | Voir utilisation disque |

### Database

| Commande | Description |
|----------|-------------|
| `make backup` | Backup MongoDB |
| `make restore` | Restaurer backup |
| `make shell-db` | MongoDB shell |

---

## 💾 Base de données MongoDB

### Commandes essentielles

```bash
# Via Python
venv/bin/python database.py  # Initialiser

# Créer user
venv/bin/python -c "from database import create_user; create_user('admin', 'admin@example.com', 'password123')"

# Migration
venv/bin/python migrate_to_multiuser.py
```

### MongoDB CLI

```bash
# Se connecter
mongosh

# Ou avec authentification
mongosh -u admin -p

# Commandes utiles
show dbs
use pymanager_db
show collections
db.users.find()
db.portfolios.find()
```

---

## 🔑 Authentification

### Structure user

```python
{
    "username": "admin",
    "email": "admin@example.com",
    "password": "hashed",  # Auto-hashé
    "created_at": datetime,
    "is_active": true
}
```

### Créer user via Python

```python
from database import create_user, authenticate_user

# Créer
user_id = create_user("john", "john@example.com", "securepass")

# Authentifier
user = authenticate_user("john", "securepass")
if user:
    print(f"Logged in: {user['username']}")
```

---

## 📊 Collections MongoDB

| Collection | Description |
|------------|-------------|
| `users` | Utilisateurs |
| `portfolios` | Portfolios par user |
| `watchlists` | Watchlist par user |
| `alerts` | Alertes par user |
| `transactions` | Transactions par user |

---

## 🌐 URLs des services

| Service | URL | Health Check |
|---------|-----|--------------|
| **Streamlit** | http://localhost:8501 | http://localhost:8501/_stcore/health |
| **MCP Server** | http://localhost:8000 | http://localhost:8000/ |
| **MongoDB** | localhost:27017 | `mongosh --eval "db.adminCommand('ping')"` |

---

## 📁 Structure fichiers

```
pymanager/
├── app3.py                 # Point d'entrée
├── database.py             # MongoDB multi-user
├── mcp_server.py          # Serveur MCP
├── start.sh               # Script démarrage
├── Makefile               # Docker commands
│
├── venv/                  # Virtual environment
│
├── pagess/                # Pages Streamlit
│   ├── dashboard.py
│   ├── portfolio.py
│   ├── market.py
│   └── ai_assistant.py
│
├── logs/                  # Logs
│   ├── mcp_server.log
│   └── streamlit.log
│
├── backups/               # Backups MongoDB
│
└── .streamlit/
    ├── config.toml
    └── secrets.toml       # API keys
```

---

## ⚙️ Configuration

### .streamlit/secrets.toml

```toml
# MongoDB
MONGODB_URI = "mongodb://localhost:27017/"

# AI Assistant
ANTHROPIC_API_KEY = "sk-ant-..."
MCP_SERVER_URL = "http://localhost:8000"
```

### .env (Docker)

```bash
ENVIRONMENT=development
MONGO_PASSWORD=secure_password
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 🧪 Tests

```bash
# Sans Docker
venv/bin/pytest tests/ -v

# Avec Docker
make test
make test-coverage
```

---

## 📝 Logs

### Localisation

```bash
# Sans Docker
logs/mcp_server.log
logs/streamlit.log

# Avec Docker
docker logs pymanager-app
docker logs pymanager-mcp
docker logs pymanager-mongo
```

### Consulter

```bash
# Temps réel
tail -f logs/*.log

# Filtrer erreurs
grep -i error logs/*.log

# Dernières 100 lignes
tail -n 100 logs/streamlit.log
```

---

## 🐛 Dépannage rapide

### MongoDB ne démarre pas

```bash
# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongodb

# Vérifier
mongosh --eval "db.adminCommand('ping')"
```

### Port déjà utilisé

```bash
# Trouver et tuer
lsof -ti:8501 | xargs kill -9  # Streamlit
lsof -ti:8000 | xargs kill -9  # MCP Server
```

### Streamlit crash

```bash
# Vérifier logs
cat logs/streamlit.log

# Redémarrer
./start.sh dev restart
```

### MCP Server crash

```bash
# Vérifier logs
cat logs/mcp_server.log

# Tester manuellement
venv/bin/python mcp_server.py
```

### Import errors

```bash
# Réinstaller dépendances
venv/bin/pip install -r requirements.txt --force-reinstall
```

---

## 🔐 Sécurité

### Checklist

- [ ] `.streamlit/secrets.toml` dans `.gitignore`
- [ ] `.env` dans `.gitignore`
- [ ] Mots de passe MongoDB changés (pas "changeme")
- [ ] API keys configurées
- [ ] Permissions fichiers correctes (`chmod 600 secrets.toml`)
- [ ] MongoDB authentification activée en prod
- [ ] HTTPS via reverse proxy (production)

---

## 🚀 Workflow développement

### Journée type

```bash
# Matin
./start.sh dev                    # Démarrer

# Développement
# (modifier le code)

# Tester
curl http://localhost:8501/_stcore/health

# Voir changements
# Streamlit auto-reload

# Soir
^C                                # Arrêter (Ctrl+C)
```

### Déploiement

```bash
# 1. Tests
venv/bin/pytest tests/ -v

# 2. Backup
venv/bin/python -c "from database import ..." # Backup

# 3. Commit
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push

# 4. Déployer
./start.sh prod
```

---

## 📊 Métriques

### Monitoring

```bash
# Statut services
./start.sh dev status

# Avec Docker
make health

# Ressources système
top | grep python
top | grep mongod
```

### Performance

| Métrique | Sans Docker | Avec Docker |
|----------|-------------|-------------|
| Temps démarrage | ~10s | ~30s |
| RAM utilisée | ~350MB | ~800MB |
| CPU idle | ~5% | ~10% |
| Espace disque | ~500MB | ~3GB |

---

## 🔄 Commandes Git

```bash
# Cloner
git clone https://github.com/baadjis/pymanager.git

# Mettre à jour
git pull origin main

# Nouvelle branche
git checkout -b feature/ma-feature

# Commit
git add .
git commit -m "feat: description"
git push origin feature/ma-feature
```

---

## 📚 Documentation

### Fichiers de doc

| Fichier | Contenu |
|---------|---------|
| `README.md` | Vue d'ensemble projet |
| `START_GUIDE.md` | Guide start.sh |
| `DOCKER_GUIDE.md` | Guide Docker/Makefile |
| `CI_CD_SETUP.md` | Guide CI/CD |
| `README_AI_INTEGRATION.md` | Guide AI Assistant |
| `QUICK_REFERENCE.md` | Ce fichier |

### Liens utiles

- MongoDB: https://docs.mongodb.com/
- Streamlit: https://docs.streamlit.io/
- Anthropic: https://docs.anthropic.com/
- FastAPI: https://fastapi.tiangolo.com/

---

## 🎯 Commandes par cas d'usage

### Je veux juste tester l'app

```bash
# Setup
python3 -m venv venv
venv/bin/pip install streamlit pymongo yfinance pandas plotly
chmod +x start.sh

# Démarrer MongoDB
brew services start mongodb-community  # macOS
# ou
sudo systemctl start mongodb           # Linux

# Créer DB et user
venv/bin/python database.py
venv/bin/python -c "from database import create_user; create_user('test', 'test@test.com', 'test123')"

# Lancer
./start.sh dev
```

### Je veux développer

```bash
# Setup complet
python3 -m venv venv
venv/bin/pip install -r requirements.txt
chmod +x start.sh

# Configurer
cp .env.example .env
nano .streamlit/secrets.toml

# Initialiser DB
venv/bin/python database.py
venv/bin/python migrate_to_multiuser.py  # Si migration

# Développer
./start.sh dev

# Tester
venv/bin/pytest tests/ -v
```

### Je veux déployer en production

```bash
# Option 1: Sans Docker
./start.sh prod

# Option 2: Avec Docker
cp .env.example .env
nano .env  # Configurer production
make deploy-prod
make health

# Monitoring
make logs -f
```

### Je veux économiser de l'espace

```bash
# Avec Docker
make disk-usage          # Voir utilisation
make prune-containers    # Léger
make prune-images        # Moyen
make prune              # ⚠️ Complet

# Sans Docker
rm -rf logs/*.log
rm -rf backups/backup_2024*
rm -rf __pycache__
```

---

## 🆘 Aide d'urgence

### Tout est cassé, comment recommencer ?

```bash
# 1. Arrêter tout
./start.sh dev stop
# ou
make down

# 2. Nettoyer
rm -rf logs/*.log
rm -f .*.pid

# 3. Backup (si données importantes)
make backup  # Avec Docker
# ou
mongodump --out=./backup_manual

# 4. Redémarrer
./start.sh dev
# ou
make up
```

### Services ne répondent pas

```bash
# 1. Vérifier statut
./start.sh dev status
curl http://localhost:8501/_stcore/health
curl http://localhost:8000/

# 2. Consulter logs
tail -50 logs/streamlit.log
tail -50 logs/mcp_server.log

# 3. Redémarrer
./start.sh dev restart
# ou
make restart
```

### Données perdues

```bash
# Restaurer backup
make restore  # Avec Docker

# Ou manuellement
mongorestore --drop ./backups/backup_YYYYMMDD_HHMMSS/

# Vérifier
mongosh
> use pymanager_db
> db.users.find()
```

---

## 💡 Tips & Tricks

### 1. Alias utiles

Ajoutez à votre `~/.bashrc` ou `~/.zshrc`:

```bash
alias pm-start='cd ~/pymanager && ./start.sh dev'
alias pm-stop='cd ~/pymanager && ./start.sh dev stop'
alias pm-logs='cd ~/pymanager && tail -f logs/*.log'
alias pm-status='cd ~/pymanager && ./start.sh dev status'
alias pm-backup='cd ~/pymanager && make backup'
```

### 2. VS Code setup

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true
}
```

### 3. Watch mode pour dev

```bash
# Terminal 1: App
./start.sh dev

# Terminal 2: Watch logs
watch -n 2 tail -20 logs/streamlit.log

# Terminal 3: Watch status
watch -n 5 './start.sh dev status'
```

### 4. Debug mode

Dans `.streamlit/config.toml`:
```toml
[runner]
fastRerun = true

[logger]
level = "debug"
```

### 5. Performance monitoring

```bash
# Script de monitoring
watch -n 1 'echo "=== Services ===" && \
  ./start.sh dev status && \
  echo "\n=== Resources ===" && \
  ps aux | grep -E "streamlit|mcp_server|mongod" | grep -v grep'
```

---

## 🔢 Ports par défaut

| Port | Service | Modifiable |
|------|---------|------------|
| 27017 | MongoDB | ✅ (mongod.conf) |
| 8000 | MCP Server | ✅ (mcp_server.py) |
| 8501 | Streamlit | ✅ (start.sh) |

---

## 📞 Support

### Debug checklist

1. [ ] MongoDB running? `mongosh --eval "db.adminCommand('ping')"`
2. [ ] Logs sans erreur? `tail logs/*.log`
3. [ ] Ports libres? `lsof -i :8501 && lsof -i :8000`
4. [ ] Config OK? `cat .streamlit/secrets.toml`
5. [ ] Dépendances OK? `venv/bin/pip list`
6. [ ] Espace disque? `df -h`

### Obtenir de l'aide

1. Consulter logs: `logs/*.log`
2. Vérifier GitHub Issues
3. Consulter documentation: `/docs`
4. Poser une question (GitHub Discussions)

---

## 🎓 Pour aller plus loin

### Fonctionnalités avancées

```bash
# CI/CD
# Voir CI_CD_SETUP.md

# Kubernetes
# Voir CI_CD_SETUP.md#kubernetes

# Monitoring
# Voir CI_CD_SETUP.md#monitoring

# Sécurité
# Voir CI_CD_SETUP.md#security
```

### Personnalisation

- Ajouter des pages: `pagess/ma_page.py`
- Modifier le thème: `.streamlit/config.toml`
- Ajouter des agents AI: `pagess/ai_assistant.py`
- Custom MCP tools: `mcp_server.py`

---

## 📋 Checklist complète

### Setup initial

- [ ] Python 3.9+ installé
- [ ] MongoDB installé et running
- [ ] Repo cloné
- [ ] Virtual env créé: `python3 -m venv venv`
- [ ] Dépendances installées: `venv/bin/pip install -r requirements.txt`
- [ ] start.sh exécutable: `chmod +x start.sh`
- [ ] MongoDB initialisé: `venv/bin/python database.py`
- [ ] User créé: `venv/bin/python -c "from database import create_user; ..."`
- [ ] Config créée: `.streamlit/secrets.toml`
- [ ] Test démarrage: `./start.sh dev`

### Production

- [ ] Backup configuré
- [ ] MongoDB sécurisé (auth activée)
- [ ] Secrets configurés (pas de défaut)
- [ ] HTTPS configuré (reverse proxy)
- [ ] Monitoring activé
- [ ] Logs rotatés
- [ ] CI/CD configuré
- [ ] Tests automatisés

---

## 🚀 Quick Commands

```bash
# Développement
./start.sh dev              # Tout démarrer
./start.sh dev status       # Vérifier
./start.sh dev logs         # Voir logs
^C                          # Arrêter

# Docker
make up                     # Démarrer
make status                 # Vérifier
make logs                   # Voir logs
make down                   # Arrêter
make prune                  # Nettoyer

# Database
venv/bin/python database.py              # Init
venv/bin/python migrate_to_multiuser.py  # Migrer
make backup                                # Backup

# Tests
venv/bin/pytest tests/ -v   # Tester
make test                   # Tester (Docker)
```

---

**⚡ Référence rapide complète! Gardez ce fichier à portée de main!**

*Dernière mise à jour: Octobre 2024*
