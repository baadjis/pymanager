# 📚 Enhanced RAG v3.0 - Guide d'Utilisation

## 🚀 Quick Start

### Installation

```bash
# Core (obligatoire)
pip install sentence-transformers numpy

# PDF support (recommandé)
pip install PyPDF2 pdfminer.six

# Better reranking (optionnel mais améliore qualité)
pip install sentence-transformers  # Inclut cross-encoder

# Web search (optionnel)
pip install duckduckgo-search wikipedia-api
```

### Utilisation Basique

```python
from knowledge.rag_engine import SimpleRAG

# Initialiser
rag = SimpleRAG()

# Recherche
results = rag.search("Qu'est-ce que le ratio de Sharpe?", top_k=3)

for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Title: {r['metadata']['title']}")
    print(f"Text: {r['text'][:200]}...\n")
```

---

## 📄 Ajouter des PDFs

### PDF Unique

```python
rag = SimpleRAG()

# Ajouter un PDF
rag.add_pdf("knowledge/pdfs/finance_course.pdf", category='course')

# Statistiques
stats = rag.get_stats()
print(f"Total docs: {stats['total_documents']}")
print(f"PDFs: {stats['sources'].get('pdf', 0)}")
```

### Batch PDFs

```python
# Ajouter tous les PDFs d'un dossier
rag.add_pdfs_from_folder(
    "knowledge/pdfs", 
    recursive=True,      # Chercher dans sous-dossiers
    category='finance'
)

# Output:
# 📚 Found 15 PDFs to process
# 📄 Processing PDF: cours_finance.pdf
#   Split into 45 chunks
# ✅ PDF indexed: cours_finance.pdf (43 chunks)
# ...
# ✅ Processed 15/15 PDFs successfully
```

---

## 🔍 Recherche Avancée

### Recherche Hybride (Recommandée)

```python
# Hybride: sémantique + mots-clés
results = rag.search(
    "Comment calculer le ratio de Sharpe?",
    top_k=5,
    hybrid=True,  # Meilleure précision
    min_score=0.3
)
```

### Recherche avec Filtres

```python
# Filtrer par catégorie
results = rag.search(
    "risque portfolio",
    category_filter='risk',  # Seulement catégorie 'risk'
    top_k=3
)

# Filtrer par source
results = rag.search(
    "optimisation",
    source_filter='pdf',  # Seulement PDFs
    top_k=3
)

# Combiner filtres
results = rag.search(
    "CAPM",
    category_filter='theory',
    source_filter='auto-populated',
    top_k=2
)
```

### Détection d'Intention

```python
query = "Qu'est-ce que le VaR?"

# Détecter l'intention
intent = rag.analyze_query_intent(query)

print(intent)
# {
#     'intent': 'definition',
#     'confidence': 0.9,
#     'entities': ['var'],
#     'suggested_filters': {'category': 'theory'}
# }

# Utiliser les filtres suggérés
results = rag.search(query, **intent['suggested_filters'])
```

---

## 📚 Ajouter Documents Texte

### Fichier Unique

```python
# Texte simple
rag.add_document(
    "Le ratio de Sharpe mesure le rendement ajusté au risque...",
    metadata={
        'title': 'Ratio de Sharpe - Introduction',
        'category': 'metrics',
        'source': 'manuel',
        'author': 'Prof. Dupont'
    }
)
```

### Depuis Dossier

```python
# Charger .txt, .md, .json
rag.add_documents_from_folder("knowledge/documents")
```

### Batch Multiple

```python
texts = [
    "Le VaR estime la perte maximale...",
    "Le Sortino améliore le Sharpe...",
    "Markowitz a développé la MPT..."
]

metadatas = [
    {'title': 'VaR', 'category': 'risk'},
    {'title': 'Sortino', 'category': 'metrics'},
    {'title': 'Markowitz', 'category': 'theory'}
]

# Plus rapide que add_document() en boucle
rag.add_documents_batch(texts, metadatas)
```

---

## 🌐 Intégration Web Search

```python
from knowledge.rag_engine import SimpleRAG
from knowledge.web_search import WebSearchEngine

rag = SimpleRAG()
web = WebSearchEngine()

# Rechercher sur le web
topic = "Black-Litterman model"
web_results = web.search(topic)

# Ajouter au RAG automatiquement
rag.add_from_web_search(topic, web_results)

# Maintenant disponible en local
results = rag.search("Black-Litterman")
```

---

## 📊 Statistiques & Monitoring

```python
stats = rag.get_stats()

print(f"""
📊 RAG Statistics:
  Total documents: {stats['total_documents']}
  Indexed keywords: {stats['indexed_keywords']}
  
  Categories: {stats['categories']}
  Sources: {stats['sources']}
  
  Model: {stats['model']}
  Re-ranking: {stats['reranking']}
  PDF support: {stats['pdf_support']}
  
  Top keywords: {list(stats['top_keywords'].keys())[:10]}
""")
```

---

## 🔧 Configuration

### Modèles Disponibles

```python
# Léger et rapide (90 MB, recommandé pour début)
rag = SimpleRAG(model_name="all-MiniLM-L6-v2")

# Finance-optimized (130 MB, RECOMMANDÉ)
rag = SimpleRAG(model_name="BAAI/bge-small-en-v1.5")

# Haute qualité (420 MB, plus lent)
rag = SimpleRAG(model_name="all-mpnet-base-v2")
```

### Cross-Encoder Re-ranking

```python
# Avec re-ranking (meilleure qualité, +0.3s par recherche)
rag = SimpleRAG(use_cross_encoder=True)

# Sans re-ranking (plus rapide)
rag = SimpleRAG(use_cross_encoder=False)
```

---

## 💡 Best Practices

### 1. Organisation des PDFs

```
knowledge/
├── pdfs/
│   ├── courses/           # Cours de finance
│   │   ├── finance_101.pdf
│   │   └── portfolio_theory.pdf
│   ├── research/          # Articles de recherche
│   │   └── markowitz_1952.pdf
│   └── guides/            # Guides pratiques
│       └── var_guide.pdf
```

```python
# Charger avec catégories
rag.add_pdfs_from_folder("knowledge/pdfs/courses", category='course')
rag.add_pdfs_from_folder("knowledge/pdfs/research", category='research')
rag.add_pdfs_from_folder("knowledge/pdfs/guides", category='guide')
```

### 2. Nommage des Documents

```python
# ✅ Bon
metadata = {
    'title': 'Ratio de Sharpe - Chapitre 3',
    'category': 'metrics',
    'source': 'cours_finance.pdf',
    'author': 'Prof. Martin',
    'date': '2024-01-15'
}

# ❌ Mauvais
metadata = {
    'title': 'doc1',
    'category': 'stuff'
}
```

### 3. Chunking

```python
# Pour PDFs techniques (formules, équations)
rag.add_pdf(
    "math_finance.pdf",
    category='technical'
)
# Chunking intelligent préserve contexte avec overlap

# Pour documents courts
rag.add_document(text, metadata)  # Pas de chunking
```

### 4. Requêtes Optimales

```python
# ✅ Requêtes spécifiques
"Comment calculer le ratio de Sharpe?"
"Différence entre VaR et CVaR"
"Formule du CAPM"

# ❌ Requêtes trop vagues
"finance"
"risque"
"optimisation"
```

---

## 🧪 Tests

```python
# Test complet
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # 1. Ajouter PDFs
    rag.add_pdfs_from_folder("knowledge/pdfs")
    
    # 2. Stats
    stats = rag.get_stats()
    print(f"Loaded {stats['total_documents']} documents")
    
    # 3. Recherche test
    queries = [
        "Qu'est-ce que le Sharpe?",
        "Comment calculer la VaR?",
        "Différence Markowitz et Black-Litterman"
    ]
    
    for q in queries:
        results = rag.search(q, top_k=2)
        print(f"\nQuery: {q}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['metadata']['title']} (score: {r['score']:.3f})")
```

---

## 🐛 Troubleshooting

### Problème: "PDF support unavailable"

```bash
pip install PyPDF2 pdfminer.six
```

### Problème: "Out of memory"

```python
# Utiliser modèle plus léger
rag = SimpleRAG(model_name="all-MiniLM-L6-v2")

# Désactiver cross-encoder
rag = SimpleRAG(use_cross_encoder=False)
```

### Problème: "Search returns no results"

```python
# Vérifier index
stats = rag.get_stats()
print(stats['total_documents'])  # Doit être > 0

# Baisser min_score
results = rag.search(query, min_score=0.1)  # Au lieu de 0.3

# Utiliser hybrid search
results = rag.search(query, hybrid=True)
```

### Problème: "Slow search"

```python
# 1. Utiliser modèle plus rapide
rag = SimpleRAG(model_name="all-MiniLM-L6-v2")

# 2. Désactiver cross-encoder
rag = SimpleRAG(use_cross_encoder=False)

# 3. Réduire top_k
results = rag.search(query, top_k=3)  # Au lieu de 10
```

---

## 📈 Performance

### Benchmarks (CPU Intel Core i5, 8GB RAM)

| Opération | all-MiniLM-L6-v2 | bge-small-en-v1.5 | Avec Cross-Encoder |
|-----------|------------------|-------------------|---------------------|
| Index 100 docs | ~5s | ~8s | ~8s |
| Search (hybrid) | ~0.2s | ~0.3s | ~0.5s |
| Add 1 PDF (20p) | ~3s | ~5s | ~5s |

### Espace Disque

- **Modèle léger**: 90 MB (all-MiniLM-L6-v2)
- **Modèle recommandé**: 130 MB (bge-small-en-v1.5)
- **Cross-encoder**: +80 MB
- **Index** (1000 docs): ~50 MB

**Total**: ~210 MB pour configuration optimale

---

## ✅ Checklist Déploiement

- [ ] Installation dépendances
- [ ] PDFs ajoutés (courses, research, guides)
- [ ] Tests recherche OK
- [ ] Statistiques vérifiées
- [ ] Performance acceptable (<1s/recherche)
- [ ] Documentation lue

---

## 🎓 Exemples d'Usage Réels

### Dans AI Assistant

```python
# pagess/ai_assistant.py
from knowledge.rag_engine import SimpleRAG

rag = SimpleRAG()

def handle_education_query(query: str) -> str:
    # Détecter intention
    intent = rag.analyze_query_intent(query)
    
    # Rechercher
    results = rag.search(
        query,
        top_k=3,
        **intent['suggested_filters']
    )
    
    if not results:
        return "❌ Aucune information trouvée"
    
    # Formater réponse
    response = f"📚 **{query}**\n\n"
    
    for r in results:
        response += f"**{r['metadata']['title']}**\n"
        response += f"{r['text'][:300]}...\n\n"
    
    return response
```

---

## 🚀 Prochaines Étapes

1. ✅ **Ajoutez vos PDFs de cours** dans `knowledge/pdfs/`
2. ✅ **Testez la recherche** avec vos questions
3. ✅ **Intégrez dans AI Assistant** pour auto-réponses
4. ✅ **Ajoutez web search** pour expansion automatique

**Bon développement ! 🎉**
