# 🚀 PyManager - Guide start.sh

Guide d'utilisation du script de démarrage intelligent `start.sh`.

---

## 📋 Vue d'ensemble

Le script `start.sh` gère automatiquement:
- ✅ Vérification et démarrage de MongoDB
- ✅ Démarrage du MCP Server
- ✅ Démarrage de Streamlit (mode headless)
- ✅ Monitoring continu des services
- ✅ Cleanup automatique à l'arrêt

---

## 🎯 Utilisation de base

### Démarrage simple

```bash
# Mode développement (par défaut)
./start.sh

# Ou explicitement
./start.sh dev
```

### Environnements disponibles

```bash
# Développement
./start.sh dev

# Test
./start.sh test

# Production
./start.sh prod
```

---

## 📝 Première utilisation

### 1. Rendre le script exécutable

```bash
chmod +x start.sh
```

### 2. Vérifier les prérequis

Le script vérifie automatiquement:
- ✅ Virtual environment (`venv/`)
- ✅ Python dans venv
- ✅ Streamlit installé
- ✅ MongoDB installé

### 3. Lancer

```bash
./start.sh dev
```

**Sortie attendue:**
```
========================================
🚀 PyManager Startup - dev
========================================
ℹ Checking prerequisites...
✓ Prerequisites OK

========================================
Starting Services
========================================
ℹ Checking MongoDB...
✓ MongoDB already running
ℹ Starting MCP Server...
✓ MCP Server started (PID: 12345)
ℹ Starting Streamlit...
✓ Streamlit started (PID: 12346)

========================================
✅ PyManager Started Successfully
========================================

✓ Environment: dev
✓ MongoDB: Running
✓ MCP Server: http://localhost:8000
✓ Streamlit: http://localhost:8501

ℹ Logs:
  - MCP Server: logs/mcp_server.log
  - Streamlit: logs/streamlit.log

⚠ Press Ctrl+C to stop all services
```

---

## 🔧 Commandes avancées

### Vérifier le statut

```bash
./start.sh dev status
```

**Sortie:**
```
========================================
Service Status
========================================
✓ MongoDB: Running
✓ MCP Server: Running (PID: 12345)
✓ Streamlit: Running (PID: 12346)
```

### Arrêter les services

```bash
./start.sh dev stop
```

### Redémarrer

```bash
./start.sh dev restart
```

### Voir les logs

```bash
./start.sh dev logs
```

---

## 🌍 Différences par environnement

### Development (dev)

```bash
./start.sh dev
```

- MongoDB local
- Streamlit headless: true
- Logs détaillés
- Auto-reload activé

### Test (test)

```bash
./start.sh test
```

- Base de données de test
- Ports différents (optionnel)
- Logs de debug

### Production (prod)

```bash
./start.sh prod
```

- Mode sécurisé
- Streamlit headless: true
- Stats désactivées
- Logs minimaux

---

## 📊 Monitoring

### Services surveillés

Le script monitore automatiquement:

1. **MCP Server** - Vérifie toutes les 5 secondes
   - Si crash → Affiche logs et arrête
   
2. **Streamlit** - Vérifie toutes les 5 secondes
   - Si crash → Affiche logs et arrête

3. **MongoDB** - Vérifié au démarrage

### Consulter les logs

```bash
# En temps réel
tail -f logs/mcp_server.log
tail -f logs/streamlit.log

# Ou via le script
./start.sh dev logs
```

---

## 🐛 Troubleshooting

### Problème: "MongoDB not installed"

**Solution:**
```bash
# macOS
brew install mongodb-community@7.0

# Ubuntu/Debian
sudo apt-get install mongodb

# Vérifier
mongod --version
```

### Problème: "Virtual environment not found"

**Solution:**
```bash
# Créer le venv
python3 -m venv venv

# Installer les dépendances
venv/bin/pip install -r requirements.txt
```

### Problème: "Port already in use"

Le script gère automatiquement:
- Détecte les ports utilisés
- Tue les processus si nécessaire

**Manuellement:**
```bash
# Tuer MCP Server (port 8000)
lsof -ti:8000 | xargs kill -9

# Tuer Streamlit (port 8501)
lsof -ti:8501 | xargs kill -9
```

### Problème: "MongoDB failed to start"

**Solution macOS:**
```bash
# Démarrer manuellement
brew services start mongodb-community

# Ou
mongod --config /usr/local/etc/mongod.conf --fork
```

**Solution Linux:**
```bash
# Démarrer manuellement
sudo systemctl start mongodb

# Vérifier
sudo systemctl status mongodb
```

### Problème: Service crash immédiatement

**Diagnostic:**
```bash
# Vérifier les logs
cat logs/mcp_server.log
cat logs/streamlit.log

# Vérifier la config
cat .streamlit/secrets.toml
```

---

## 🔄 Workflow recommandé

### Développement quotidien

```bash
# Matin - Démarrer
./start.sh dev

# Laisser tourner pendant le dev
# Les logs défilent automatiquement

# Soir - Arrêter (Ctrl+C)
^C  # Le script nettoie tout automatiquement
```

### Redémarrage rapide

```bash
# Ctrl+C pour arrêter
# Puis relancer
./start.sh dev
```

### Changement d'environnement

```bash
# Passer de dev à prod
^C  # Arrêter dev
./start.sh prod  # Démarrer prod
```

---

## 📁 Fichiers créés

Le script crée ces fichiers:

```
pymanager/
├── logs/
│   ├── mcp_server.log        # Logs MCP Server
│   └── streamlit.log          # Logs Streamlit
├── .mcp_server.pid           # PID du MCP Server
└── .streamlit.pid            # PID de Streamlit
```

### Nettoyer les fichiers

```bash
# Supprimer les PID files
rm -f .mcp_server.pid .streamlit.pid

# Nettoyer les logs
rm -f logs/*.log

# Ou tout nettoyer
rm -rf logs/ .*.pid
```

---

## ⚙️ Configuration

### Variables d'environnement

Le script utilise:

```bash
ENVIRONMENT=dev              # dev, test, prod
MONGODB_URI=mongodb://...    # URI MongoDB
ANTHROPIC_API_KEY=sk-ant-... # API Key Claude
```

Configurez dans `.streamlit/secrets.toml`:

```toml
MONGODB_URI = "mongodb://localhost:27017/"
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Ports personnalisés

Modifiez dans `start.sh`:

```bash
MCP_SERVER_PORT=8000    # Changez ici
STREAMLIT_PORT=8501     # Changez ici
```

---

## 🚦 Statut des services

### Codes de sortie

- `0` - Succès
- `1` - Erreur (service failed to start)

### Vérifier si tout tourne

```bash
# Via le script
./start.sh dev status

# Ou manuellement
curl http://localhost:8000/        # MCP Server
curl http://localhost:8501/_stcore/health  # Streamlit
mongosh --eval "db.adminCommand('ping')"   # MongoDB
```

---

## 💡 Astuces

### 1. Lancer en arrière-plan

```bash
# Détacher du terminal
nohup ./start.sh dev > /dev/null 2>&1 &

# Voir les logs
tail -f logs/streamlit.log
```

### 2. Automatiser au démarrage système

**macOS (launchd):**

Créez `~/Library/LaunchAgents/com.pymanager.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" 
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pymanager</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/pymanager/start.sh</string>
        <string>prod</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

Puis:
```bash
launchctl load ~/Library/LaunchAgents/com.pymanager.plist
```

**Linux (systemd):**

Créez `/etc/systemd/system/pymanager.service`:

```ini
[Unit]
Description=PyManager Service
After=network.target mongodb.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/pymanager
ExecStart=/path/to/pymanager/start.sh prod
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Puis:
```bash
sudo systemctl enable pymanager
sudo systemctl start pymanager
```

### 3. Monitoring avec watch

```bash
# Surveiller le statut
watch -n 2 './start.sh dev status'
```

### 4. Logs structurés

```bash
# Filtrer les erreurs
grep -i error logs/streamlit.log

# Suivre uniquement les erreurs
tail -f logs/*.log | grep -i error
```

---

## 🔐 Sécurité

### En production

1. **Ne pas lancer en root** ✅
2. **Utiliser HTTPS** (via reverse proxy)
3. **Limiter les IPs** (firewall)
4. **Logs sécurisés** (permissions 640)

### Permissions fichiers

```bash
# Scripts exécutables
chmod +x start.sh

# Logs lisibles uniquement par le user
chmod 640 logs/*.log

# PID files
chmod 644 .*.pid
```

---

## 📊 Performance

### Ressources utilisées

| Service | CPU | RAM |
|---------|-----|-----|
| MongoDB | ~5% | ~100MB |
| MCP Server | ~2% | ~50MB |
| Streamlit | ~10% | ~200MB |
| **Total** | ~17% | ~350MB |

### Optimisations

Le script utilise:
- ✅ Streamlit headless (pas de browser)
- ✅ Stats désactivées
- ✅ Monitoring léger (5s intervals)

---

## 🆘 Aide rapide

### Commandes essentielles

```bash
./start.sh dev          # Démarrer
./start.sh dev status   # Statut
./start.sh dev stop     # Arrêter
./start.sh dev logs     # Logs
Ctrl+C                  # Arrêter (dans terminal actif)
```

### Diagnostic rapide

```bash
# Tout en un
./start.sh dev status

# Services individuels
curl http://localhost:8000/
curl http://localhost:8501/_stcore/health
mongosh --eval "db.adminCommand('ping')"
```

---

**🚀 Script start.sh prêt! Démarrage intelligent et monitoring automatique!**
