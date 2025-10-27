# 📚 Knowledge Enhancement Guide

Guide complet pour la knowledge base enrichie de PyManager

---

## 🎯 Vue d'ensemble

La Knowledge Enhancement enrichit l'AI Assistant avec 3 sources de connaissances :

1. **RAG (Retrieval-Augmented Generation)** - Documents locaux
2. **Web Search** - DuckDuckGo + Wikipedia
3. **Economic Data** - FRED API

### Architecture

```
Question Utilisateur
    ↓
1. KB Hardcodée (instantané)
    ↓ (si pas trouvé)
2. RAG - Documents locaux (rapide)
    ↓ (si pas trouvé)
3. Web Search (moyen)
    ↓
4. Synthèse avec Claude (final)
    ↓
Réponse Enrichie
```

---

## 📦 Installation

### 1. Dépendances

```bash
pip install sentence-transformers duckduckgo-search wikipedia-api fredapi
```

**Détail des packages:**
- `sentence-transformers` - Embeddings pour RAG (~200MB)
- `duckduckgo-search` - Recherche web gratuite (~2MB)
- `wikipedia-api` - API Wikipedia (~1MB)
- `fredapi` - Données économiques FRED (~1MB)

### 2. Initialisation

```bash
python scripts/init_knowledge.py
```

Ce script :
- ✅ Crée la structure de dossiers
- ✅ Génère des documents d'exemple
- ✅ Initialise le RAG
- ✅ Teste les APIs

---

## 🔧 Configuration

### 1. FRED API Key (Optionnel mais recommandé)

**Obtenir une clé gratuite:**
1. Allez sur https://fred.stlouisfed.org/
2. Créez un compte (gratuit)
3. Obtenez votre API key

**Configurer:**

`.streamlit/secrets.toml`:
```toml
FRED_API_KEY = "votre-cle-fred-ici"
```

Ou variable d'environnement:
```bash
export FRED_API_KEY="votre-cle"
```

### 2. Pas de Configuration Requise

✅ **DuckDuckGo** - Pas d'API key nécessaire
✅ **Wikipedia** - Pas d'API key nécessaire
✅ **RAG** - Fonctionne localement (CPU)

---

## 📚 RAG (Retrieval-Augmented Generation)

### Qu'est-ce que le RAG ?

RAG permet à l'IA de chercher dans vos documents locaux et d'enrichir ses réponses.

**Pipeline:**
```
Document → Découpage → Embeddings → Index → Recherche → Réponse
```

### Ajouter des Documents

**Formats supportés:**
- `.txt` - Fichiers texte
- `.md` - Markdown
- `.json` - JSON structuré

**Emplacement:**
```
knowledge/documents/
├── sharpe_ratio.md
├── diversification.md
├── markowitz.md
└── votre_document.txt
```

### Réindexer

```bash
python -c "from knowledge.rag_engine import SimpleRAG; rag = SimpleRAG(); rag.clear_index(); rag.add_documents_from_folder()"
```

### API RAG

```python
from knowledge.rag_engine import SimpleRAG

# Initialiser
rag = SimpleRAG()

# Ajouter un document
rag.add_document(
    text="Le ratio de Sharpe mesure...",
    metadata={"title": "Sharpe Ratio", "category": "metrics"}
)

# Rechercher
results = rag.search("Qu'est-ce que le Sharpe?", top_k=3)

for result in results:
    print(f"{result['metadata']['title']}: {result['score']:.3f}")
    print(result['text'][:200])

# Statistiques
stats = rag.get_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Categories: {stats['categories']}")
```

### Format JSON pour Documents

```json
{
  "title": "Ratio de Sharpe",
  "content": "Le ratio de Sharpe est...",
  "category": "metrics",
  "tags": ["risk", "performance", "ratios"]
}
```

---

## 🌐 Web Search

### Sources Disponibles

1. **DuckDuckGo**
   - Recherche web générale
   - Gratuit, pas de limite
   - Pas d'API key

2. **Wikipedia**
   - Articles encyclopédiques
   - Français et autres langues
   - Très fiable pour finance

### API Web Search

```python
from knowledge.web_search import WebSearchEngine

search = WebSearchEngine()

# Recherche multi-sources
results = search.search(
    "ratio de sharpe",
    sources=['all'],  # ou ['duckduckgo', 'wikipedia']
    max_results=5
)

# Résultats DuckDuckGo
if 'duckduckgo' in results['sources']:
    for r in results['sources']['duckduckgo']['results']:
        print(f"{r['title']}: {r['snippet']}")

# Résultats Wikipedia
if 'wikipedia' in results['sources']:
    wiki = results['sources']['wikipedia']
    if wiki.get('found'):
        print(f"Wikipedia: {wiki['title']}")
        print(wiki['summary'])

# Recherche simplifiée
synthesized = search.search_financial_term("diversification")
print(synthesized)
```

### Cache

Les résultats sont mis en cache pendant 24h :

```python
# Localisation du cache
knowledge/cache/

# Effacer le cache
search.clear_cache()
```

---

## 📊 FRED Economic Data

### Indicateurs Disponibles

**Taux d'Intérêt:**
- `DFF` - Federal Funds Rate
- `DGS10` - 10-Year Treasury
- `T10Y2Y` - Yield Curve Spread

**Inflation:**
- `CPIAUCSL` - Consumer Price Index
- `PCEPI` - PCE Price Index

**Croissance:**
- `GDP` - Gross Domestic Product
- `UNRATE` - Unemployment Rate

**Marchés:**
- `SP500` - S&P 500 Index
- `VIXCLS` - VIX Volatility Index

[Liste complète](https://fred.stlouisfed.org/categories)

### API FRED

```python
from knowledge.fed_data import FREDDataProvider

# Initialiser
fed = FREDDataProvider(api_key="votre-cle")

# Dernière valeur
fed_rate = fed.get_latest_value('DFF')
print(f"Fed Funds Rate: {fed_rate}%")

# Série temporelle
import pandas as pd
data = fed.get_indicator('UNRATE', start_date='2020-01-01')
print(data.tail())

# Résumé économique
summary = fed.get_economic_summary()
for name, data in summary['indicators'].items():
    print(f"{name}: {data['value']}")

# Indicateurs de récession
recession = fed.get_recession_indicators()
print(f"Risk Level: {recession['overall_signal']}")

# Inflation
inflation = fed.get_inflation_data(months=12)
print(f"YoY Inflation: {inflation['yoy_inflation']:.1f}%")

# Contexte formaté pour l'IA
context = fed.format_economic_context()
print(context)
```

---

## 🤖 Utilisation dans l'AI Assistant

### Exemples de Requêtes

**1. Éducation (RAG + Web):**
```
"Explique-moi le ratio de Sharpe"
→ Cherche dans RAG
→ Si pas trouvé, web search
→ Synthèse avec Claude
```

**2. Économie (FRED):**
```
"Quel est le taux d'intérêt actuel?"
→ Récupère via FRED
→ Contexte économique
→ Réponse avec Claude
```

**3. Recherche (Web):**
```
"Qu'est-ce que la théorie des jeux en finance?"
→ Web search (pas dans RAG)
→ Wikipedia + DuckDuckGo
→ Synthèse
```

### Pipeline Complet

```python
def handle_education_query(prompt: str) -> str:
    # 1. KB hardcodée (instantané)
    if prompt in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[prompt]
    
    # 2. RAG (rapide)
    rag_results = rag.search(prompt, top_k=3)
    if rag_results:
        context = build_context(rag_results)
        return synthesize_with_claude(prompt, context, "RAG")
    
    # 3. Web Search (moyen)
    web_results = web_search.search(prompt)
    if web_results:
        context = build_context(web_results)
        return synthesize_with_claude(prompt, context, "Web")
    
    # 4. Fallback
    return "Aucune information trouvée"
```

---

## 🧪 Tests

### Test RAG

```bash
python knowledge/rag_engine.py
```

**Résultat attendu:**
```
🔧 Testing RAG Engine

📚 Adding sample documents...
   ✓ Added 5 documents

📊 Index Stats:
  Total documents: 5
  Categories: {'metrics': 1, 'strategy': 1, 'theory': 2, 'risk': 1}

🔍 Testing search...
  Query: Qu'est-ce que le ratio de Sharpe?
    1. Ratio de Sharpe (score: 0.856)
    2. Value at Risk (VaR) (score: 0.423)
```

### Test Web Search

```bash
python knowledge/web_search.py
```

### Test FRED

```bash
export FRED_API_KEY="votre-cle"
python knowledge/fed_data.py
```

### Test Intégration

```bash
python scripts/init_knowledge.py
```

---

## 📈 Performance

### Temps de Réponse

| Source | Temps Moyen | Mise en Cache |
|--------|-------------|---------------|
| KB Hardcodée | <1ms | N/A |
| RAG | 50-200ms | Oui (embeddings) |
| Web Search | 1-3s | Oui (24h) |
| FRED API | 200-500ms | Non |
| Claude Synthesis | 1-5s | Non |

### Optimisations

**1. Cache Agressif**
```python
# Web search cached 24h
# Augmenter si besoin
search = WebSearchEngine(cache_ttl=86400*7)  # 7 jours
```

**2. RAG Batch Processing**
```python
# Charger plusieurs documents en une fois
rag.add_documents_from_folder()  # Plus rapide que un par un
```

**3. Modèle Léger**
```python
# Utiliser modèle léger pour RAG
rag = SimpleRAG(model_name="all-MiniLM-L6-v2")  # 80MB
# Au lieu de "all-mpnet-base-v2" (420MB)
```

---

## 🎓 Meilleures Pratiques

### 1. Organisation des Documents

**Structure recommandée:**
```
knowledge/documents/
├── metrics/
│   ├── sharpe_ratio.md
│   ├── sortino_ratio.md
│   └── var_cvar.md
├── theories/
│   ├── markowitz.md
│   ├── black_litterman.md
│   └── apt.md
├── strategies/
│   ├── diversification.md
│   ├── dollar_cost_averaging.md
│   └── rebalancing.md
└── basics/
    ├── stocks_101.md
    ├── bonds_101.md
    └── etfs_101.md
```

### 2. Format des Documents

**Template Markdown:**
```markdown
# Titre Principal

## Introduction
Brève introduction (2-3 lignes)

## Définition
Définition claire et concise

## Formule (si applicable)
```
Formule mathématique
```

## Utilisation Pratique
Comment l'utiliser dans PyManager

## Exemples
Exemples concrets avec chiffres

## Limites
Ce qu'il faut savoir

## Ressources
- Liens externes
- Articles de référence
```

### 3. Métadonnées Riches

```json
{
  "title": "Ratio de Sharpe",
  "content": "...",
  "category": "metrics",
  "difficulty": "intermediate",
  "tags": ["risk", "performance", "ratios"],
  "related": ["sortino_ratio", "information_ratio"],
  "last_updated": "2024-01-15"
}
```

### 4. Mise à Jour Régulière

```bash
# Script de mise à jour mensuel
#!/bin/bash
# update_knowledge.sh

cd knowledge/documents/

# Télécharger nouvelles définitions
curl -o inflation_latest.md https://...

# Réindexer
cd ../..
python -c "from knowledge.rag_engine import SimpleRAG; rag = SimpleRAG(); rag.clear_index(); rag.add_documents_from_folder()"

echo "Knowledge base updated!"
```

---

## 🔐 Sécurité & Privacy

### Données Locales (RAG)

✅ **Avantages:**
- Données restent sur votre machine
- Pas d'envoi à des serveurs tiers
- Contrôle total

⚠️ **Attention:**
- Ne pas indexer de données sensibles
- Les documents sont lus par Claude pour synthèse

### Web Search

⚠️ **Ce qui est envoyé:**
- Requêtes de recherche à DuckDuckGo
- Requêtes à Wikipedia
- User-agent: "PyManager/1.0"

✅ **Ce qui n'est PAS envoyé:**
- Vos données de portfolio
- Informations personnelles
- Données des utilisateurs

### FRED API

✅ **Sécurité:**
- API publique, données non sensibles
- Pas de données personnelles envoyées
- Seulement l'API key transmise

🔒 **Protection de l'API key:**
```toml
# .streamlit/secrets.toml (gitignored)
FRED_API_KEY = "votre-cle"
```

---

## 🐛 Troubleshooting

### Problème 1: RAG ne trouve rien

**Symptôme:**
```python
results = rag.search("sharpe ratio")
# → []
```

**Solutions:**

1. **Vérifier l'index:**
```python
stats = rag.get_stats()
print(stats)  # Doit avoir des documents
```

2. **Réindexer:**
```python
rag.clear_index()
rag.add_documents_from_folder()
```

3. **Baisser le seuil:**
```python
results = rag.search("sharpe ratio", min_score=0.2)  # Au lieu de 0.4
```

### Problème 2: Web Search ne retourne rien

**Symptôme:**
```python
results = search.search("sharpe ratio")
# → {'error': '...'}
```

**Solutions:**

1. **Vérifier internet:**
```bash
ping duckduckgo.com
```

2. **Vérifier les dépendances:**
```bash
pip install --upgrade duckduckgo-search wikipedia-api
```

3. **Tester individuellement:**
```python
# Test DuckDuckGo
from duckduckgo_search import DDGS
results = list(DDGS().text("test", max_results=1))
print(results)

# Test Wikipedia
import wikipediaapi
wiki = wikipediaapi.Wikipedia('fr', 'PyManager/1.0')
page = wiki.page("Sharpe ratio")
print(page.exists())
```

### Problème 3: FRED API Error

**Symptôme:**
```
FREDError: API key is not valid
```

**Solutions:**

1. **Vérifier la clé:**
```python
import os
print(os.getenv('FRED_API_KEY'))  # Doit afficher votre clé
```

2. **Tester la clé:**
```bash
curl "https://api.stlouisfed.org/fred/series?series_id=DFF&api_key=VOTRE_CLE&file_type=json"
```

3. **Régénérer la clé:**
- https://fred.stlouisfed.org/
- Account Settings → API Keys → Create New Key

### Problème 4: sentence-transformers trop lent

**Symptôme:**
```
Le premier search prend 30+ secondes
```

**Solutions:**

1. **C'est normal la première fois** (téléchargement du modèle)

2. **Utiliser un modèle plus léger:**
```python
# Au lieu de all-mpnet-base-v2 (420MB)
rag = SimpleRAG(model_name="all-MiniLM-L6-v2")  # 80MB
```

3. **Pré-charger le modèle:**
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Problème 5: Mémoire insuffisante

**Symptôme:**
```
MemoryError ou OOM (Out of Memory)
```

**Solutions:**

1. **Utiliser modèle léger:**
```python
rag = SimpleRAG(model_name="paraphrase-MiniLM-L3-v2")  # 60MB
```

2. **Limiter le batch size:**
```python
# Dans rag_engine.py, modifier:
embedding = self.model.encode(text, batch_size=8)  # Au lieu de 32
```

3. **Augmenter swap (Linux):**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📊 Monitoring

### Statistiques RAG

```python
from knowledge.rag_engine import SimpleRAG

rag = SimpleRAG()
stats = rag.get_stats()

print(f"""
📊 RAG Statistics
─────────────────
Total Documents: {stats['total_documents']}
Categories: {stats['categories']}
Model: {stats['model']}
""")
```

### Statistiques Web Search

```python
from knowledge.web_search import WebSearchEngine
from pathlib import Path

search = WebSearchEngine()
cache_files = list(Path("knowledge/cache").glob("*.json"))

print(f"""
🌐 Web Search Statistics
─────────────────────────
Cached Queries: {len(cache_files)}
Cache Size: {sum(f.stat().st_size for f in cache_files) / 1024:.1f} KB
""")
```

### Logs

```python
import logging

# Activer logs détaillés
logging.basicConfig(level=logging.DEBUG)

# Logs dans fichier
logging.basicConfig(
    filename='knowledge.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 🚀 Cas d'Usage Avancés

### 1. Multi-langue

```python
# RAG en anglais
rag_en = SimpleRAG(knowledge_dir="knowledge_en")

# Wikipedia multilingue
wiki_en = wikipediaapi.Wikipedia('en', 'PyManager/1.0')
wiki_fr = wikipediaapi.Wikipedia('fr', 'PyManager/1.0')
```

### 2. Sources Personnalisées

```python
# Ajouter vos propres sources
from bs4 import BeautifulSoup
import requests

def scrape_investopedia(topic):
    url = f"https://www.investopedia.com/terms/{topic[0]}/{topic}.asp"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extraire contenu
    content = soup.find('div', class_='article-content')
    
    # Ajouter au RAG
    rag.add_document(
        text=content.text,
        metadata={'source': 'investopedia', 'topic': topic}
    )

# Peupler avec vos topics
topics = ['sharpe-ratio', 'modern-portfolio-theory']
for topic in topics:
    scrape_investopedia(topic)
```

### 3. Mise à Jour Automatique

```python
# Cron job pour mise à jour quotidienne
# crontab: 0 2 * * * /path/to/update_knowledge.py

import schedule
import time

def update_economic_data():
    """Mise à jour données économiques"""
    fed = FREDDataProvider(api_key)
    summary = fed.get_economic_summary()
    
    # Sauvegarder dans un document
    with open('knowledge/documents/economic_update.md', 'w') as f:
        f.write(fed.format_economic_context())
    
    # Réindexer
    rag = SimpleRAG()
    rag.add_documents_from_folder()

# Exécuter tous les jours à 2h
schedule.every().day.at("02:00").do(update_economic_data)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

---

## 📚 Ressources Recommandées

### Documents à Ajouter

**Basics:**
- Introduction à l'investissement
- Types d'actifs (actions, obligations, etc.)
- Ratios financiers de base

**Intermediate:**
- Théories de portfolio (Markowitz, APT, CAPM)
- Analyse technique vs fondamentale
- Gestion des risques

**Advanced:**
- Modèles quantitatifs
- Machine Learning en finance
- Stratégies algorithmiques

### Sources de Contenu

**Gratuites:**
- [Investopedia](https://www.investopedia.com/)
- [Wikipedia Finance](https://en.wikipedia.org/wiki/Portal:Finance)
- [FRED Blog](https://fredblog.stlouisfed.org/)
- [Khan Academy Finance](https://www.khanacademy.org/economics-finance-domain)

**Académiques:**
- [SSRN](https://www.ssrn.com/)
- [arXiv Finance](https://arxiv.org/list/q-fin/recent)
- Papers de recherche (avec permission)

---

## 🔮 Roadmap

### Court Terme (1-2 mois)

- [ ] Support PDF pour RAG
- [ ] Interface UI pour gérer documents
- [ ] Analytics détaillés
- [ ] Export de la knowledge base

### Moyen Terme (3-6 mois)

- [ ] Multi-langue (EN, FR, ES)
- [ ] Sources supplémentaires (Brave Search, Bing)
- [ ] Fine-tuning du modèle embeddings
- [ ] Clustering automatique des documents

### Long Terme (6-12 mois)

- [ ] Knowledge graph
- [ ] Fact-checking automatique
- [ ] Version collaborative (multi-users)
- [ ] Marketplace de knowledge bases

---

## 💡 Contributing

Vous avez des documents financiers de qualité à partager ?

1. Fork le projet
2. Ajoutez vos documents dans `knowledge/documents/`
3. Format: Markdown avec métadonnées
4. Pull Request avec description

**Critères:**
- ✅ Contenu original ou sous licence appropriée
- ✅ Format Markdown propre
- ✅ Références citées
- ✅ Langue française correcte
- ✅ Pas de contenu promotionnel

---

## 📄 License

Knowledge Enhancement code: MIT License

**Note sur le contenu:**
- Documents d'exemple: MIT License
- Vos propres documents: Votre choix de licence
- Contenu scraped: Respecter les licences sources

---

## 🆘 Support

**Issues GitHub:**
https://github.com/baadjis/pymanager/issues

**Documentation:**
- RAG: `knowledge/rag_engine.py`
- Web Search: `knowledge/web_search.py`
- FRED: `knowledge/fed_data.py`

**Tests:**
```bash
python scripts/init_knowledge.py
```

---

<div align="center">

**Knowledge Enhancement v1.0**

RAG + Web Search + Economic Data

Made with 🧠 and ☕

</div>
|
