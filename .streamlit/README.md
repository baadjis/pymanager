# 📁 Dossier .streamlit

Ce dossier contient la configuration de votre application Streamlit ΦManager.

## 📄 Fichiers

### **config.toml** ⚙️
Configuration principale de l'application :
- Thème par défaut
- Port du serveur (8501)
- Options de performance
- Paramètres de sécurité

### **secrets.toml** 🔐
**⚠️ IMPORTANT : NE PAS COMMIT CE FICHIER**

Contient toutes vos clés API et secrets :
- OpenAI API key
- Database credentials
- Email SMTP config
- API keys (Alpha Vantage, Polygon, etc.)

**Ajoutez à .gitignore :**
```
.streamlit/secrets.toml
.streamlit/credentials.toml
```

### **credentials.toml** 👤
Credentials Streamlit (généré automatiquement)

## 🚀 Configuration rapide

### 1. OpenAI API (pour AI Assistant)

Dans `secrets.toml`, remplacez :
```toml
[openai]
api_key = "sk-your-actual-key-here"
```

Obtenez votre clé sur : https://platform.openai.com/api-keys

### 2. Database (optionnel)

Si vous utilisez PostgreSQL/MySQL :
```toml
[database]
host = "localhost"
port = 5432
name = "phimanager"
user = "your-username"
password = "your-password"
```

### 3. Accès aux secrets dans le code

```python
import streamlit as st

# Accéder aux secrets
openai_key = st.secrets["openai"]["api_key"]
db_host = st.secrets["database"]["host"]

# Vérifier si une clé existe
if "openai" in st.secrets:
    # Utiliser OpenAI
    pass
```

## 🎨 Personnalisation du thème

### Thème Dark (actuel)
```toml
[theme]
primaryColor = "#6366F1"        # Violet
backgroundColor = "#0B0F19"     # Noir bleuté
secondaryBackgroundColor = "#131720"
textColor = "#F8FAFC"
```

### Thème Light (alternative)
```toml
[theme]
primaryColor = "#6366F1"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#F1F5F9"
textColor = "#1E293B"
```

### Thèmes prédéfinis

**🟣 Purple (actuel)**
```toml
primaryColor = "#6366F1"
```

**🔵 Blue**
```toml
primaryColor = "#3B82F6"
```

**🟢 Green**
```toml
primaryColor = "#10B981"
```

**🔴 Red**
```toml
primaryColor = "#EF4444"
```

**🟠 Orange**
```toml
primaryColor = "#F59E0B"
```

## ⚡ Optimisations performance

### Activer le cache agressif
```toml
[server]
enableStaticServing = true
maxUploadSize = 200
```

### Fast reruns
```toml
[runner]
fastReruns = true
magicEnabled = true
```

## 🔒 Sécurité

### Protection CSRF
```toml
[server]
enableXsrfProtection = true
enableCORS = false
```

### Masquer les erreurs en production
```toml
[client]
showErrorDetails = false  # Mettre à false en production
```

## 📊 Analytics (optionnel)

Ajoutez dans `secrets.toml` :
```toml
[analytics]
google_analytics_id = "GA-XXXXXXXXX"
```

Puis dans votre code :
```python
# components/analytics.py
import streamlit.components.v1 as components

def inject_ga():
    ga_id = st.secrets.get("analytics", {}).get("google_analytics_id")
    if ga_id:
        ga_code = f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{ga_id}');
        </script>
        """
        components.html(ga_code, height=0)
```

## 🌍 Déploiement

### Streamlit Cloud

1. **Pushez votre code sur GitHub** (sans secrets.toml)
2. **Connectez-vous à** https://share.streamlit.io
3. **Ajoutez vos secrets** dans l'interface web :
   - Settings → Secrets
   - Copiez le contenu de votre `secrets.toml`

### Docker

Créez un `.env` :
```bash
OPENAI_API_KEY=your-key
DATABASE_HOST=localhost
```

Dans votre `Dockerfile` :
```dockerfile
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
```

### Heroku

Ajoutez `setup.sh` :
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

## 🐛 Troubleshooting

### Secrets non trouvés
```python
# Vérifier si les secrets existent
if "openai" not in st.secrets:
    st.error("⚠️ OpenAI API key not found in secrets")
    st.stop()
```

### Port déjà utilisé
```bash
# Changer le port dans config.toml
[server]
port = 8502
```

### Cache problèmes
```bash
# Effacer le cache
streamlit cache clear
```

## 📝 Checklist de sécurité

- [ ] `secrets.toml` ajouté à `.gitignore`
- [ ] Clés API valides et actives
- [ ] CORS désactivé en production
- [ ] XSRF protection activée
- [ ] Error details masqués en production
- [ ] HTTPS activé (en production)
- [ ] Secrets stockés dans variables d'environnement (production)

## 💡 Tips

1. **Environnements multiples** : Créez `secrets.dev.toml` et `secrets.prod.toml`
2. **Rotation des clés** : Changez régulièrement vos API keys
3. **Monitoring** : Activez les analytics pour suivre l'usage
4. **Backup** : Sauvegardez vos configs (sauf secrets)

## 🔗 Ressources

- [Streamlit Config Docs](https://docs.streamlit.io/library/advanced-features/configuration)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [Streamlit Theming](https://docs.streamlit.io/library/advanced-features/theming)