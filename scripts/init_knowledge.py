# scripts/init_knowledge.py
"""
Script d'initialisation de la Knowledge Base
- Crée la structure de dossiers
- Télécharge des documents de référence
- Initialise le RAG
- Teste les APIs
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

print("""
╔═══════════════════════════════════════════════════╗
║     PyManager Knowledge Base Initialization       ║
╚═══════════════════════════════════════════════════╝
""")

# =============================================================================
# 1. Create Directory Structure
# =============================================================================

print("📁 Creating directory structure...")

directories = [
    "knowledge",
    "knowledge/documents",
    "knowledge/embeddings",
    "knowledge/cache"
]

for directory in directories:
    dir_path = project_root / directory
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {directory}")

# =============================================================================
# 2. Create Sample Documents
# =============================================================================

print("\n📝 Creating sample documents...")

sample_docs = {
    "sharpe_ratio.md": """# Ratio de Sharpe

## Définition
Le ratio de Sharpe est une mesure du rendement ajusté au risque d'un investissement.

## Formule
```
Ratio de Sharpe = (Rendement Portfolio - Taux Sans Risque) / Écart-type Portfolio
```

## Interprétation
- **> 2.0**: Excellent (performance exceptionnelle)
- **1.0-2.0**: Très bon (bonne performance ajustée au risque)
- **0.5-1.0**: Acceptable (performance correcte)
- **< 0.5**: Faible (rendement insuffisant pour le risque pris)
- **< 0**: Négatif (perte, sous-performance du taux sans risque)

## Utilisation
Le ratio de Sharpe aide les investisseurs à:
1. Comparer différents portfolios
2. Évaluer si le rendement compense le risque
3. Optimiser l'allocation d'actifs

## Limites
- Suppose une distribution normale des rendements
- Ne capture pas tous les types de risques
- Sensible aux données extrêmes (outliers)

## Exemple
Portfolio A: 15% rendement, 10% volatilité, taux sans risque 2%
Ratio de Sharpe = (15% - 2%) / 10% = 1.3 (Très bon)

Portfolio B: 20% rendement, 25% volatilité, taux sans risque 2%
Ratio de Sharpe = (20% - 2%) / 25% = 0.72 (Acceptable)

Conclusion: Portfolio A offre un meilleur rendement ajusté au risque.
""",
    
    "diversification.md": """# Diversification de Portfolio

## Principe Fondamental
"Ne mettez pas tous vos œufs dans le même panier"

La diversification consiste à répartir les investissements sur différents actifs
pour réduire le risque global du portfolio.

## Pourquoi Diversifier?

### 1. Réduction du Risque
- Le risque spécifique peut être éliminé
- Seul le risque systématique (marché) reste
- Lissage des rendements dans le temps

### 2. Protection
- Contre les chocs sectoriels
- Contre les événements spécifiques
- Contre la volatilité excessive

### 3. Optimisation
- Meilleur ratio risque/rendement
- Frontière efficiente de Markowitz
- Portefeuille optimal

## Comment Diversifier?

### Par Classes d'Actifs
- **Actions**: 60% (croissance)
- **Obligations**: 30% (stabilité)
- **Liquidités**: 10% (opportunités)

### Par Secteurs
- Technologie: 20%
- Santé: 15%
- Finance: 15%
- Consommation: 15%
- Industrie: 15%
- Autres: 20%

**Règle**: Aucun secteur > 25%

### Par Géographie
- Domestique: 60%
- International développé: 30%
- Marchés émergents: 10%

### Par Capitalisation
- Large cap: 70% (stabilité)
- Mid cap: 20% (équilibre)
- Small cap: 10% (croissance)

## Niveau de Diversification Optimal

**Recherche académique:**
- 15-30 actions = ~90% des bénéfices de diversification
- Au-delà de 30 = rendements marginaux décroissants
- Trop de diversification = dilution des gains

## Sur-Diversification

**Attention aux pièges:**
- Trop d'actifs → gestion complexe
- Coûts de transaction élevés
- Dilution des meilleures opportunités
- "Diworsification"

## Rééquilibrage

Fréquence recommandée: tous les 6-12 mois
- Vendre les gagnants surévalués
- Acheter les perdants sous-évalués
- Maintenir l'allocation cible
""",

    "markowitz.md": """# Théorie Moderne du Portfolio (Markowitz)

## Histoire
Développée par Harry Markowitz en 1952, cette théorie révolutionnaire
lui a valu le Prix Nobel d'Économie en 1990.

## Concept Clé
Optimiser le ratio risque/rendement en construisant un portfolio efficient.

## Hypothèses Fondamentales

1. **Investisseurs Rationnels**
   - Averses au risque
   - Préfèrent plus de rendement à moins
   - Maximisent l'utilité espérée

2. **Mesure du Risque**
   - Variance (écart-type) des rendements
   - Corrélation entre actifs
   - Matrice de covariance

3. **Efficience des Marchés**
   - Tous les investisseurs ont les mêmes informations
   - Pas de coûts de transaction
   - Actifs divisibles

## Frontière Efficiente

Ensemble des portfolios offrant:
- Le rendement maximal pour un niveau de risque donné
- Le risque minimal pour un rendement donné

## Formule Mathématique

**Rendement du Portfolio:**
```
E(Rp) = Σ wi * E(Ri)
```

**Risque du Portfolio:**
```
σp² = Σ Σ wi * wj * σij
```

Où:
- wi, wj = poids des actifs i et j
- E(Ri) = rendement espéré de l'actif i
- σij = covariance entre actifs i et j

## Portfolio Optimal

Trois stratégies principales:

### 1. Maximum Sharpe Ratio
- Meilleur rendement ajusté au risque
- Point tangent sur la frontière efficiente
- Recommandé pour la plupart des investisseurs

### 2. Minimum Variance
- Risque minimal
- Convient aux investisseurs très conservateurs
- Peut avoir un rendement faible

### 3. Maximum Return
- Rendement maximal
- Risque élevé
- Pour investisseurs agressifs

## Limites du Modèle

1. **Estimation des Paramètres**
   - Difficile d'estimer rendements futurs
   - Matrices de covariance instables
   - Sensible aux données historiques

2. **Hypothèses Irréalistes**
   - Marchés pas toujours efficients
   - Coûts de transaction existent
   - Distribution non normale

3. **Solutions Extrêmes**
   - Poids parfois irréalistes
   - Positions concentrées
   - Instabilité des allocations

## Améliorations Modernes

- **Black-Litterman**: Intègre les vues d'experts
- **Robust Optimization**: Gère l'incertitude
- **Risk Parity**: Équilibre les contributions au risque

## Dans PyManager

Utilisez le Portfolio Builder:
1. Sélectionnez vos actifs
2. Choisissez "Markowitz"
3. Sélectionnez la stratégie (Sharpe/Risk/Return)
4. Le système calcule les poids optimaux
""",

    "black_litterman.md": """# Modèle Black-Litterman

## Introduction
Développé par Fischer Black et Robert Litterman chez Goldman Sachs en 1990,
ce modèle résout les problèmes pratiques du modèle de Markowitz.

## Problèmes Résolus

### Issues du Modèle de Markowitz:
1. **Allocations Extrêmes**: Poids trop concentrés
2. **Instabilité**: Petits changements → grandes variations
3. **Pas de Point de Départ**: Commencer avec une page blanche

### Solutions de Black-Litterman:
1. **Équilibre du Marché**: Point de départ cohérent
2. **Intégration de Vues**: Incorpore vos convictions
3. **Allocations Stables**: Résultats plus réalistes

## Principe de Fonctionnement

### 1. Équilibre du Marché (Prior)
Commence avec les capitalisations boursières comme poids initiaux.

**Formule:**
```
Π = λ * Σ * w_market
```
Où:
- Π = rendements implicites d'équilibre
- λ = coefficient d'aversion au risque
- Σ = matrice de covariance
- w_market = poids de marché

### 2. Vues de l'Investisseur

**Types de vues:**
- **Absolues**: "L'action A aura 12% de rendement"
- **Relatives**: "A surperformera B de 3%"

**Formule des vues:**
```
P * μ = Q + ε
```
Où:
- P = matrice pick (définit les actifs)
- μ = rendements attendus
- Q = vues quantifiées
- ε = incertitude des vues

### 3. Combinaison (Posterior)

Combine l'équilibre et les vues via théorème de Bayes:

```
E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ * [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

Où:
- τ = paramètre de scaling (ex: 0.05)
- Ω = matrice d'incertitude des vues

## Exemple Pratique

### Données:
- Actions: AAPL, MSFT, GOOGL
- Poids de marché: 40%, 35%, 25%

### Vues:
1. "AAPL va surperformer de 5%" (confiance: 80%)
2. "MSFT va mieux faire que GOOGL de 3%" (confiance: 60%)

### Résultat:
Black-Litterman ajuste les poids pour refléter ces vues
tout en restant proche de l'équilibre de marché.

## Avantages

1. **Point de Départ Solide**: Équilibre de marché
2. **Flexibilité**: Ajoutez vos insights
3. **Stabilité**: Allocations plus réalistes
4. **Interprétabilité**: Compréhension intuitive

## Paramètres Clés

### τ (Tau)
- Mesure l'incertitude sur l'équilibre
- Typiquement: 0.01 à 0.05
- Plus grand τ = plus d'importance aux vues

### Confiance des Vues
- Détermine le poids des vues
- Haute confiance = plus d'impact
- Faible confiance = reste proche équilibre

### Taux Sans Risque
- Base pour calculer les rendements excédentaires
- Utiliser taux des bons du Trésor

## Dans PyManager

1. Portfolio Builder → "Black-Litterman"
2. Entrez vos vues (optionnel)
3. Ajustez les paramètres (τ, confiance)
4. Le modèle calcule les poids optimaux

**Sans vues:** Retourne l'équilibre de marché pur.
**Avec vues:** Ajuste selon vos convictions.
"""
}

for filename, content in sample_docs.items():
    file_path = project_root / "knowledge" / "documents" / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   ✓ {filename}")

# =============================================================================
# 3. Initialize RAG
# =============================================================================

print("\n🤖 Initializing RAG engine...")

try:
    from knowledge.rag_engine import SimpleRAG
    
    rag = SimpleRAG()
    rag.clear_index()  # Clear any existing index
    
    print("   📚 Indexing documents...")
    rag.add_documents_from_folder()
    
    stats = rag.get_stats()
    print(f"   ✓ Indexed {stats['total_documents']} documents")
    print(f"   ✓ Categories: {stats['categories']}")
    
except Exception as e:
    print(f"   ❌ Error initializing RAG: {e}")
    print("   💡 Install: pip install sentence-transformers")

# =============================================================================
# 4. Test Web Search
# =============================================================================

print("\n🌐 Testing Web Search...")

try:
    from knowledge.web_search import WebSearchEngine
    
    search = WebSearchEngine()
    
    # Test DuckDuckGo
    print("   Testing DuckDuckGo...")
    results = search._search_duckduckgo("investissement portfolio", max_results=2)
    if results.get('results'):
        print(f"   ✓ DuckDuckGo: {len(results['results'])} results")
    else:
        print("   ⚠️ DuckDuckGo: No results (might need internet)")
    
    # Test Wikipedia
    print("   Testing Wikipedia...")
    wiki_result = search._search_wikipedia("diversification")
    if wiki_result.get('found'):
        print(f"   ✓ Wikipedia: Found '{wiki_result['title']}'")
    else:
        print("   ⚠️ Wikipedia: Not found")
    
except Exception as e:
    print(f"   ❌ Error testing web search: {e}")
    print("   💡 Install: pip install duckduckgo-search wikipedia-api")

# =============================================================================
# 5. Test FRED API
# =============================================================================

print("\n📊 Testing FRED API...")

fred_key = os.getenv('FRED_API_KEY') or input("Enter FRED API key (or press Enter to skip): ").strip()

if fred_key:
    try:
        from knowledge.fed_data import FREDDataProvider
        
        fred = FREDDataProvider(fred_key)
        
        if fred.fred:
            # Test basic query
            fed_rate = fred.get_latest_value('DFF')
            if fed_rate:
                print(f"   ✓ FRED Connected: Fed Funds Rate = {fed_rate:.2f}%")
            
            # Test economic summary
            summary = fred.get_economic_summary()
            if 'indicators' in summary:
                print(f"   ✓ Retrieved {len(summary['indicators'])} indicators")
        else:
            print("   ❌ FRED API not initialized")
    
    except Exception as e:
        print(f"   ❌ Error testing FRED: {e}")
        print("   💡 Install: pip install fredapi")
else:
    print("   ⏭️  Skipped (no API key)")
    print("   💡 Get free key at: https://fred.stlouisfed.org/")

# =============================================================================
# 6. Test Integration
# =============================================================================

print("\n🧪 Testing Integration...")

try:
    print("   Testing RAG search...")
    if 'rag' in locals():
        results = rag.search("Qu'est-ce que le ratio de Sharpe?", top_k=2)
        if results:
            print(f"   ✓ RAG returned {len(results)} results")
            print(f"   ✓ Top result: {results[0]['metadata'].get('title', 'Unknown')} (score: {results[0]['score']:.3f})")
        else:
            print("   ⚠️ No RAG results")
    
except Exception as e:
    print(f"   ❌ Integration test failed: {e}")

# =============================================================================
# 7. Summary & Next Steps
# =============================================================================

print("\n" + "="*60)
print("✅ Knowledge Base Initialization Complete!")
print("="*60)

print("\n📋 Summary:")
print("   - Directory structure created")
print("   - Sample documents created and indexed")
print("   - RAG engine initialized" if 'rag' in locals() else "   - RAG engine: ⚠️ Not available")
print("   - Web search tested" if 'search' in locals() else "   - Web search: ⚠️ Not available")
print("   - FRED API tested" if fred_key else "   - FRED API: ⏭️ Skipped")

print("\n🚀 Next Steps:")
print("   1. Add your own documents to: knowledge/documents/")
print("   2. Run: python -c 'from knowledge.rag_engine import SimpleRAG; rag = SimpleRAG(); rag.add_documents_from_folder()'")
print("   3. Configure FRED_API_KEY in .streamlit/secrets.toml")
print("   4. Test AI Assistant with: streamlit run app3.py")

print("\n💡 Tips:")
print("   - Add PDF/MD/TXT files to knowledge/documents/")
print("   - RAG will automatically index them")
print("   - Web search cached for 24h")
print("   - FRED data for economic context")

print("\n📚 Documentation:")
print("   - RAG: knowledge/rag_engine.py")
print("   - Web Search: knowledge/web_search.py")
print("   - FRED: knowledge/fed_data.py")
print("   - AI Assistant: pagess/ai_assistant.py")

print("\n" + "="*60)
