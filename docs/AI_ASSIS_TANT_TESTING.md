# 🧪 Guide de Test - AI Assistant v4.0

## Tests Rapides (5 min)

### 1️⃣ Test Connexion MCP

```python
# Dans Streamlit, sidebar devrait afficher:
# 🟢 MCP v4.0
# 📊 15 tools disponibles
```

**Test manuel:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

---

### 2️⃣ Test Portfolio Queries

#### Test A: Liste portfolios
**Prompt:** `Mes portfolios`

**Résultat attendu:**
```
📊 Vos Portfolios (X total)

Vue d'ensemble:
- 💰 Valeur totale: $XX,XXX.XX
- 📈 P&L total: $X,XXX.XX (X.XX%)
...
```

#### Test B: Analyse détaillée
**Prompt:** `Analyse mon portfolio Growth`

**Résultat attendu:**
```
📊 Analyse: Growth

Holdings:
🟢 AAPL: 30.0% ($X,XXX.XX)
...

📉 Métriques de Risque:
- Sharpe Ratio: X.XX (bon/excellent)
- Max Drawdown: -X.XX%
...
```

---

### 3️⃣ Test Market Intelligence (NEW)

#### Test A: Market Overview
**Prompt:** `Vue marché US`

**Résultat attendu:**
```
📊 Market Overview - US

Sentiment global: BULLISH (X.XX%)

Indices principaux:
🟢 S&P 500: $X,XXX.XX (+X.XX%) - bullish
...

Performance par secteur:
🟢 Technology: +X.XX%
...
```

#### Test B: Sector Analysis
**Prompt:** `Analyse secteur semiconductors`

**Résultat attendu:**
```
🔬 Analyse Secteur: Semiconductors

Performance moyenne (3 mois): X.XX%
Sentiment: BULLISH

Top Performers:
🟢 NVDA: +XX.XX%
🟢 AMD: +XX.XX%
...

Diversification: MEDIUM
```

#### Test C: Quantum Computing (NEW)
**Prompt:** `Analyse secteur quantum`

**Résultat attendu:**
```
🔬 Analyse Secteur: Quantum

Performance moyenne: X.XX%

Top Performers:
🟢 IONQ: +XX.XX%
🟢 RGTI: +XX.XX%
🟢 QUBT: +XX.XX%
...
```

---

### 4️⃣ Test Sentiment Analysis (NEW)

**Prompt:** `Sentiment NVDA`

**Résultat attendu:**
```
🚀 Sentiment Analysis: NVDA

Sentiment: VERY BULLISH
- Score: X.XX
- Confiance: X.XX

Indicateurs techniques:
- RSI (14): XX.X
- Au-dessus SMA 20: ✅
- Au-dessus SMA 50: ✅
...
```

---

### 5️⃣ Test Comparison (NEW)

**Prompt:** `Compare AAPL et MSFT`

**Résultat attendu:**
```
📊 Comparaison: AAPL, MSFT

Rankings:
- 🏆 Meilleure performance: AAPL
- 💎 Meilleur Sharpe: MSFT
- 🛡️ Moins volatile: MSFT

Détails:
🟢 AAPL
   - Performance: +XX.XX%
   - Volatilité: XX.XX%
   - Sharpe: X.XX

🟢 MSFT
   - Performance: +XX.XX%
   ...
```

---

### 6️⃣ Test Backtesting (NEW)

**Prompt:** `Backtest mon portfolio Growth`

**Résultat attendu:**
```
🟢 Backtest Results: Growth

Période: YYYY-MM-DD → YYYY-MM-DD

Performance:
- Capital initial: $10,000.00
- Valeur finale: $12,XXX.XX
- Rendement total: +XX.XX%
- Rendement annualisé: +XX.XX%

Métriques de risque:
- Sharpe Ratio: X.XX
- Max Drawdown: -XX.XX%
...
```

---

### 7️⃣ Test Predictions (NEW)

**Prompt:** `Prédis performance de Growth sur 3 mois`

**Résultat attendu:**
```
🟢 Prédiction: Growth

Horizon: 3mo

Rendement attendu: +X.XX%

Intervalle de confiance (95%):
- Borne inférieure: +X.XX%
- Borne supérieure: +XX.XX%

⚠️ Les performances passées ne garantissent pas...
```

---

### 8️⃣ Test Education

**Prompt:** `Explique le Sharpe`

**Résultat attendu:**
```
📚 Ratio de Sharpe

Mesure le rendement ajusté au risque...

Formule: (R - Rf) / σ
...

Interprétation:
- > 2.0 : Excellent
...
```

---

### 9️⃣ Test Research

**Prompt:** `Recherche Apple`

**Résultat attendu:**
```
🟢 AAPL - Apple Inc.

Prix actuel: $XXX.XX
Performance 1 an: +XX.XX%

Informations:
- Secteur: Technology
- Capitalisation: $X,XXX.XXB
- P/E Ratio: XX.XX

💡 Utilisez "Analyse le sentiment sur AAPL"...
```

---

### 🔟 Test Claude AI Fallback

**Prompt:** `Quelle est la stratégie d'investissement la plus prudente ?`

**Résultat attendu:**
```
🤖 [Réponse générée par Claude AI]

Une stratégie prudente consiste à...
- Diversification
- Allocation d'actifs
- Rééquilibrage régulier
...
```

---

## Tests d'Erreur

### Test 1: Portfolio inexistant
**Prompt:** `Analyse portfolio NonExistant`

**Attendu:** `❌ Portfolio 'NonExistant' non trouvé`

### Test 2: Ticker invalide
**Prompt:** `Recherche INVALIDTICKER`

**Attendu:** `❌ Données indisponibles pour INVALIDTICKER`

### Test 3: MCP Offline
**Action:** Arrêter MCP Server

**Attendu:** Sidebar montre `🔴 MCP v4.0`

**Test prompt:** `Vue marché US`

**Attendu:** Utilise Yahoo Finance directement ou message d'erreur

---

## Tests UI/UX

### Test Conversations
1. Créer nouvelle conversation → ✅ Titre "Nouvelle conversation"
2. Envoyer 2 messages → ✅ Auto-titre avec premier message
3. Créer 2ème conversation → ✅ Bascule automatique
4. Revenir à 1ère → ✅ Historique préservé
5. Supprimer conversation → ✅ Bascule vers suivante

### Test Feedback (si disponible)
1. Envoyer message → ✅ Boutons 👍 👎 apparaissent
2. Cliquer 👍 → ✅ Feedback enregistré
3. Ouvrir "Feedback Stats" → ✅ Statistiques affichées

### Test Sidebar
1. Status MCP → ✅ Affiche 🟢/🔴
2. Status Claude → ✅ Affiche 🟢/🔴  
3. Compteur messages → ✅ S'incrémente

---

## Tests Performance

### Test 1: Caching
```bash
# Premier appel (cold)
time: Envoyer "Vue marché US"
# Attendu: 2-5 secondes

# Deuxième appel (même prompt dans 30s)
time: Envoyer "Vue marché US"
# Attendu: <1 seconde (cache MCP)
```

### Test 2: Concurrent Requests
```bash
# Terminal 1
curl -X POST http://localhost:8000/api/market/overview -d '{"region":"US"}'

# Terminal 2 (simultané)
curl -X POST http://localhost:8000/api/market/sector -d '{"sector":"technology"}'

# Attendu: Les deux répondent en <5s
```

---

## Checklist Complète

### Fonctionnalités Core
- [ ] Liste portfolios fonctionne
- [ ] Analyse portfolio détaillée
- [ ] Calcul métriques de risque

### Market Intelligence (NEW)
- [ ] Market overview (US, EU, ASIA, GLOBAL)
- [ ] Sector analysis (tech, semiconductors, quantum, etc.)
- [ ] Sentiment analysis
- [ ] Market comparison

### Advanced Features (NEW)
- [ ] Backtesting
- [ ] Predictions ML
- [ ] Monte Carlo (si implémenté)

### Education & Research
- [ ] Knowledge base accessible
- [ ] Recherche ticker fonctionne
- [ ] Claude AI fallback

### UI/UX
- [ ] Conversations management
- [ ] Feedback system (optionnel)
- [ ] Sidebar status correct
- [ ] Suggestions fonctionnent

### Performance
- [ ] MCP répond en <5s
- [ ] Cache fonctionne
- [ ] Pas de crashes

---

## Rapport de Test

```
Date: _______________
Testeur: _______________

✅ Tests réussis: ___ / 30
⚠️ Tests avec warnings: ___
❌ Tests échoués: ___

Notes:
_________________________________
_________________________________
_________________________________

Bugs trouvés:
1. _____________________________
2. _____________________________

Améliorations suggérées:
1. _____________________________
2. _____________________________
```

---

## Commandes de Debug

### Vérifier logs MCP
```bash
# Dans terminal MCP
# Devrait afficher:
# Executing tool: get_portfolios | Params: {...}
```

### Vérifier état Streamlit
```python
# Dans Streamlit console (debugging)
st.write("MCP Connected:", check_mcp_connection())
st.write("Tools available:", len(get_mcp_tools()))
st.write("Chat history:", len(st.session_state.chat_history))
```

### Vérifier MongoDB
```python
from database import get_portfolios
portfolios = list(get_portfolios(user_id))
print(f"Found {len(portfolios)} portfolios")
```

---

## 🎉 Test Réussi Si...

1. ✅ Tous les prompts suggérés fonctionnent
2. ✅ Status sidebar correct (🟢 MCP v4.0)
3. ✅ Conversations sauvegardées
4. ✅ Pas d'erreurs dans console
5. ✅ Réponses en <5 secondes
6. ✅ Claude AI fallback fonctionne
7. ✅ Market Intelligence répond correctement
8. ✅ Backtesting/Predictions fonctionnent

**Si 7/8 ✅ → Migration réussie ! 🚀**
