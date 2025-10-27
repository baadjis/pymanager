# 🐳 PyManager - Guide Docker

Guide rapide pour gérer PyManager avec Docker et économiser l'espace disque.

---

## 🚀 Démarrage rapide

### 1. Configuration initiale

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer et configurer
nano .env  # ou vim .env
```

### 2. Démarrer avec Docker

```bash
# Build et démarrer
make build
make up

# Ou en une commande
make deploy

# Avec logs en temps réel
make quick-start
```

### 3. Vérifier le statut

```bash
make status
make health
```

---

## 📋 Commandes principales

### Gestion des services

```bash
make up          # Démarre les services
make down        # Arrête et supprime les containers
make restart     # Redémarre
make stop        # Arrête sans supprimer
make start       # Redémarre les services arrêtés
```

### Logs

```bash
make logs        # Tous les logs
make logs-app    # Streamlit uniquement
make logs-mcp    # MCP Server uniquement
make logs-db     # MongoDB uniquement
```

### Shell access

```bash
make shell       # Shell Streamlit
make shell-mcp   # Shell MCP Server
make shell-db    # MongoDB shell
```

---

## 💾 Gestion de la base de données

### Backup

```bash
# Créer un backup
make backup

# Les backups sont dans ./backups/backup_YYYYMMDD_HHMMSS/
```

### Restore

```bash
# Restaurer le dernier backup
make restore
```

---

## 🧹 Nettoyage (Économiser l'espace disque)

### Nettoyage léger

```bash
# Supprimer containers et volumes du projet
make clean

# Supprimer uniquement les containers arrêtés
make prune-containers

# Supprimer uniquement les images non utilisées
make prune-images

# Supprimer uniquement les volumes non utilisés
make prune-volumes
```

### Nettoyage complet ⚠️

```bash
# ATTENTION: Supprime TOUTES les données Docker non utilisées
make prune

# Vérifier l'espace libéré
make disk-usage
```

### Analyser l'utilisation disque

```bash
# Vue d'ensemble Docker
make disk-usage

# Taille des images
make size

# Liste des volumes
make volumes
```

**Astuce:** Lancer `make prune` régulièrement pour libérer de l'espace!

---

## 🧪 Tests

```bash
# Lancer les tests
make test

# Tests avec couverture
make test-coverage
```

---

## 🔧 Développement

### Mode développement

```bash
# Rebuild à chaque changement
make dev

# Ou avec logs
make dev-logs
```

### Rebuild complet

```bash
# Rebuild tout de zéro
make rebuild
```

### Mise à jour

```bash
# Mettre à jour les images
make update
```

---

## 📊 Monitoring

### Statut des services

```bash
# Statut simple
make status

# Health check complet
make health

# Statistiques CPU/RAM
make stats

# Watch mode (rafraîchit toutes les 2s)
make watch
```

### Accéder aux services

- **Streamlit**: http://localhost:8501
- **MCP Server**: http://localhost:8000
- **MongoDB**: localhost:27017

---

## 🚀 Production

### Déploiement production

```bash
# Déploiement avec backup automatique
make deploy-prod
```

**Workflow de déploiement:**
1. ✅ Backup de la base de données
2. ✅ Build des nouvelles images
3. ✅ Arrêt des anciens containers
4. ✅ Démarrage des nouveaux containers

---

## 💡 Astuces

### 1. Économiser l'espace disque

```bash
# Chaque semaine
make prune-containers  # Supprimer containers arrêtés
make prune-images      # Supprimer images non utilisées

# Chaque mois
make prune             # Nettoyage complet
```

### 2. Build sans cache

Le Makefile build toujours avec `--no-cache` pour économiser l'espace.

### 3. Volumes MongoDB

Les données MongoDB sont dans un volume Docker nommé `pymanager_mongodb_data`.

Pour supprimer complètement:
```bash
make clean  # Supprime tout, y compris les volumes
```

### 4. Logs volumineux

Si les logs prennent trop de place:

```bash
# Voir la taille
docker system df

# Nettoyer les logs
docker container prune
```

### 5. Images multiples

Docker garde plusieurs versions d'images. Nettoyez régulièrement:

```bash
make clean-images  # Supprime les images du projet
make prune-images  # Supprime toutes les images non utilisées
```

---

## 🔍 Troubleshooting

### Problème: Port déjà utilisé

```bash
# Changer les ports dans .env
STREAMLIT_PORT=8502
MCP_PORT=8001
MONGODB_PORT=27018

# Ou tuer le processus
lsof -ti:8501 | xargs kill -9
```

### Problème: MongoDB ne démarre pas

```bash
# Vérifier les logs
make logs-db

# Vérifier les permissions du volume
docker volume inspect pymanager_mongodb_data

# Recréer le volume
make clean
make up
```

### Problème: Build échoue

```bash
# Rebuild complet
make rebuild

# Vérifier les logs
make logs
```

### Problème: Container crash

```bash
# Vérifier les logs
make logs

# Redémarrer
make restart

# Health check
make health
```

### Problème: Manque d'espace disque

```bash
# Voir l'utilisation
make disk-usage

# Nettoyer
make prune

# Supprimer les backups anciens
rm -rf backups/backup_2024*
```

---

## 📚 Commandes avancées

### Accès direct aux containers

```bash
# Logs d'un container spécifique
docker logs -f pymanager-app

# Shell avec commande
docker exec pymanager-app python database.py

# Copier des fichiers
docker cp pymanager-app:/app/logs/streamlit.log ./
```

### Inspecter un volume

```bash
# Lister les volumes
make volumes

# Inspecter
docker volume inspect pymanager_mongodb_data

# Backup manuel d'un volume
docker run --rm -v pymanager_mongodb_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/mongodb_backup.tar.gz /data
```

### Network debugging

```bash
# Voir le réseau
docker network inspect pymanager-network

# Tester la connexion inter-containers
docker exec pymanager-app ping mongodb
docker exec pymanager-app curl http://mcp-server:8000
```

---

## 🔐 Sécurité

### Bonnes pratiques

1. **Ne jamais commiter .env** ✅
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Changer les mots de passe par défaut** ✅
   ```bash
   nano .env  # Modifier MONGO_PASSWORD
   ```

3. **Utiliser des secrets Docker en production** ✅
   ```bash
   docker secret create mongo_password password.txt
   ```

4. **Limiter les ressources** ✅
   ```yaml
   # Dans docker-compose.yml
   services:
     streamlit-app:
       deploy:
         resources:
           limits:
             cpus: '1'
             memory: 1G
   ```

---

## 📦 Structure des volumes

```
Docker Volumes:
├── pymanager_mongodb_data  # Données MongoDB
└── ./backups              # Backups (host)
```

### Backup complet

```bash
# MongoDB
make backup

# Code + config
tar czf pymanager_backup.tar.gz \
  app3.py database.py pagess/ \
  .streamlit/ docker-compose.yml
```

---

## 🎯 Workflow recommandé

### Développement quotidien

```bash
# Matin
make quick-start        # Démarre tout avec logs

# Pendant le dev
make logs              # Suivre les logs
make restart           # Après changements

# Soir
make down              # Arrêter
```

### Déploiement hebdomadaire

```bash
# 1. Backup
make backup

# 2. Mise à jour
make update

# 3. Tests
make test

# 4. Déploiement
make deploy-prod

# 5. Vérification
make health
```

### Maintenance mensuelle

```bash
# 1. Backup
make backup

# 2. Nettoyage Docker
make prune

# 3. Vérifier l'espace
make disk-usage

# 4. Supprimer vieux backups
find backups/ -type d -mtime +30 -delete
```

---

## 📊 Métriques de performance

### Surveiller les ressources

```bash
# Stats en temps réel
make stats

# OU
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### Limites recommandées

| Service | CPU | RAM | Disk |
|---------|-----|-----|------|
| Streamlit | 1 core | 1GB | 500MB |
| MCP Server | 0.5 core | 512MB | 100MB |
| MongoDB | 1 core | 1GB | 5GB+ |

---

## 🆘 Support

### Commandes de diagnostic

```bash
# Tout-en-un
make info

# Statut détaillé
make health

# Logs complets
make logs > debug.log
```

### Réinitialisation complète

```bash
# ATTENTION: Supprime TOUT
make clean          # Containers + volumes
make clean-images   # + Images
make prune          # + Tout Docker

# Redémarrer de zéro
make deploy
```

---

## 📝 Checklist déploiement

### Avant déploiement

- [ ] Backup créé: `make backup`
- [ ] Tests passent: `make test`
- [ ] .env configuré correctement
- [ ] Secrets sécurisés (pas de mots de passe par défaut)
- [ ] Espace disque suffisant: `make disk-usage`

### Après déploiement

- [ ] Services démarrés: `make status`
- [ ] Health check OK: `make health`
- [ ] Logs sans erreur: `make logs`
- [ ] Application accessible: http://localhost:8501
- [ ] MCP Server répond: http://localhost:8000

---

## 🎓 Ressources

- **Docker Docs**: https://docs.docker.com/
- **Docker Compose**: https://docs.docker.com/compose/
- **MongoDB Docker**: https://hub.docker.com/_/mongo
- **Streamlit**: https://docs.streamlit.io/

---

## 💾 Comparaison taille

### Sans optimisation

```
Docker Images:     ~5GB
Build Cache:       ~2GB
Containers:        ~500MB
Volumes:           ~1GB
Total:             ~8.5GB
```

### Avec optimisation (ce Makefile)

```
Docker Images:     ~2GB  (--no-cache)
Build Cache:       0GB   (supprimé)
Containers:        ~500MB
Volumes:           ~1GB
Total:             ~3.5GB  ✅ -60% d'espace!
```

**Astuce:** Lancer `make prune` régulièrement maintient Docker à ~3GB.

---

## 🔄 Commandes rapides (cheat sheet)

| Commande | Description |
|----------|-------------|
| `make` ou `make help` | Affiche l'aide |
| `make up` | Démarre |
| `make down` | Arrête |
| `make logs` | Voir les logs |
| `make status` | Statut |
| `make backup` | Backup DB |
| `make prune` | Nettoyage complet |
| `make health` | Health check |
| `make shell` | Shell Streamlit |

---

**🐳 Docker configuré et optimisé! Profitez de PyManager sans remplir votre disque!**
