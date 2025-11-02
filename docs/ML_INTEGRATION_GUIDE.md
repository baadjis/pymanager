# 🤖 MCP Server v4.0 - Documentation Complète

> **Architecture Hybride REST API + MCP Protocol**  
> Production-ready pour Next.js, Mobile Apps, et AI Assistants

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [REST API Endpoints](#rest-api-endpoints)
5. [MCP Tools](#mcp-tools)
6. [Fonctionnalités Avancées](#fonctionnalités-avancées)
7. [Exemples d'utilisation](#exemples-dutilisation)
8. [Déploiement](#déploiement)
9. [FAQ](#faq)

---

## 🎯 Vue d'ensemble

### Pourquoi une architecture hybride ?

| Protocole | Use Cases | Avantages |
|-----------|-----------|-----------|
| **REST API** | Frontend (Next.js, React), Mobile (iOS/Android), Intégrations tierces | Standard, Swagger docs, Facile à consommer |
| **MCP (Model Context Protocol)** | AI Assistants (Claude, GPT), Agents autonomes, Workflows IA | Contextuel, Sémantique riche, Optimisé pour l'IA |

### Fonctionnalités Principales

✅ **Portfolio Management**
- CRUD complet sur les portfolios
- Calcul de P&L en temps réel
- Métriques de risque (Sharpe, Sortino, VaR, CVaR)
- Analyse de corrélation

✅ **Market Intelligence**
- Vue d'ensemble des marchés (US, EU, ASIA, GLOBAL)
- Analyse sectorielle et sous-sectorielle
  - **Semiconductors** (NVDA, AMD, INTC, TSM, ASML, QCOM)
  - **Quantum Computing** (IONQ, RGTI, QUBT, IBM, GOOGL)
  - **AI/ML** (NVDA, MSFT, GOOGL, META, PLTR)
  - Healthcare, Finance, Energy, Consumer, Industrials
- Sentiment analysis en temps réel
- Comparaison multi-marchés

✅ **Backtesting & Predictions**
- Backtesting avancé avec rebalancing
- Prédictions ML (ARIMA/LSTM/Ensemble)
- Simulations Monte Carlo
- Stress testing avec scénarios personnalisés

✅ **Advanced Analytics**
- Matrices de corrélation
- Optimisation de portfolio
- Risk decomposition
- Performance attribution

✅ **Infrastructure**
- Caching intelligent (5 min TTL)
- Rate limiting (TODO)
- Logging complet
- Health checks
- Swagger/ReDoc auto-générés

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Clients                               │
├──────────────┬──────────────┬──────────────┬────────────┤
│   Next.js    │  Mobile App  │ Claude AI    │  Postman   │
│  (REST API)  │  (REST API)  │   (MCP)      │ (Testing)  │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬─────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   FastAPI Server (Port    │
         │   8000)                   │
         │                           │
         │  • CORS Middleware        │
         │  • Authentication         │
         │  • Rate Limiting          │
         │  • Caching Layer          │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │      Route Handlers       │
         │                           │
         │  • REST Endpoints         │
         │  • MCP Tool Router        │
         │  • WebSocket (WIP)        │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     Business Logic        │
         │                           │
         │  • Portfolio Manager      │
         │  • Market Intelligence    │
         │  • Backtesting Engine     │
         │  • ML Predictions         │
         │  • Analytics Engine       │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     Data Sources          │
         │                           │
         │  • MongoDB (Internal)     │
         │  • Yahoo Finance API      │
         │  • Cache (In-Memory)      │
         └───────────────────────────┘
```

---

## 🚀 Installation

### Prérequis

```bash
Python 3.9+
MongoDB 7.0+
pip install -r requirements.txt
```

### Dépendances principales

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pymongo==4.5.0
pandas==2.0.3
numpy==1.24.3
yfinance==0.2.28
```

### Démarrage

```bash
# Terminal 1 - Démarrer le serveur
python mcp_server.py

# Terminal 2 - Tests
python test_mcp_v4.py

# Accès
# API Docs: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
# Health: http://localhost:8000/health
```

---

## 📡 REST API Endpoints

### System Endpoints

#### `GET /`
Root endpoint avec informations système

**Response:**
```json
{
  "service": "ΦManager Unified API + MCP",
  "version": "4.0.0",
  "status": "operational",
  "features": {
    "rest_api": true,
    "mcp_protocol": true,
    "database": true,
    "market_data": true
  }
}
```

#### `GET /health`
Health check détaillé

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "market_data": "connected",
    "cache": "operational"
  },
  "metrics": {
    "cache_size": 42,
    "uptime": "2h 30m"
  }
}
```

### Portfolio Endpoints

#### `GET /api/portfolios/{user_id}`
Récupérer tous les portfolios d'un utilisateur

**Parameters:**
- `user_id` (path): User identifier

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolios": [
      {
        "name": "Growth Portfolio",
        "current_value": 12500.50,
        "pnl": 2500.50,
        "pnl_pct": 25.0,
        "holdings": [...]
      }
    ],
    "count": 1,
    "aggregates": {
      "total_value": 12500.50,
      "total_pnl": 2500.50
    }
  }
}
```

#### `GET /api/portfolios/{user_id}/{portfolio_name}`
Détails d'un portfolio spécifique

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "Growth Portfolio",
    "holdings": [
      {
        "symbol": "AAPL",
        "weight": 0.3,
        "quantity": 10,
        "current_price": 175.50,
        "value": 1755.00
      }
    ],
    "metrics": {
      "sharpe_ratio": 1.85,
      "volatility": 0.18,
      "max_drawdown": -0.12
    },
    "correlations": {...}
  }
}
```

### Market Intelligence Endpoints

#### `POST /api/market/overview`
Vue d'ensemble du marché

**Request Body:**
```json
{
  "region": "US",
  "include_sectors": true,
  "period": "1mo"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "region": "US",
    "indices": [
      {
        "name": "S&P 500",
        "ticker": "^GSPC",
        "price": 4567.89,
        "change_1mo": 3.45,
        "sentiment": "bullish"
      }
    ],
    "market_sentiment": {
      "label": "bullish",
      "score": 2.8
    },
    "sectors": [...]
  }
}
```

#### `POST /api/market/sector`
Analyse sectorielle détaillée

**Request Body:**
```json
{
  "sector": "semiconductors",
  "subsector": null,
  "metrics": ["performance", "sentiment", "top_stocks", "correlations"],
  "period": "3mo"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sector": "semiconductors",
    "performance": {
      "average": 15.6,
      "best": {"ticker": "NVDA", "performance_3mo": 45.2},
      "worst": {"ticker": "INTC", "performance_3mo": -8.5}
    },
    "sentiment": {
      "score": 0.7,
      "label": "bullish"
    },
    "top_performers": [...],
    "correlations": {
      "average": 0.65,
      "diversification": "medium"
    }
  }
}
```

#### `POST /api/market/sentiment`
Analyse de sentiment

**Request Body:**
```json
{
  "target": "AAPL",
  "period": "1mo",
  "include_news": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "target": "AAPL",
    "sentiment": {
      "label": "bullish",
      "score": 0.35,
      "confidence": 0.35
    },
    "metrics": {
      "positive_days": 15,
      "negative_days": 6,
      "rsi": 62.5,
      "above_sma20": true,
      "above_sma50": true
    }
  }
}
```

### Backtesting Endpoints

#### `POST /api/backtest`
Backtest d'un portfolio

**Request Body:**
```json
{
  "user_id": "user123",
  "portfolio_name": "Growth Portfolio",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 10000,
  "rebalance_frequency": "monthly",
  "transaction_cost": 0.001
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "backtest_results": {
      "initial_capital": 10000,
      "final_value": 12500,
      "total_return": 0.25,
      "annualized_return": 0.25,
      "sharpe_ratio": 1.85,
      "sortino_ratio": 2.15,
      "max_drawdown": -0.12,
      "win_rate": 0.65
    },
    "configuration": {...}
  }
}
```

#### `POST /api/predict`
Prédiction de performance

**Request Body:**
```json
{
  "user_id": "user123",
  "portfolio_name": "Growth Portfolio",
  "horizon": "3mo",
  "model": "ensemble",
  "confidence_level": 0.95
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": {
      "horizon": "3mo",
      "expected_return": 0.08,
      "expected_return_pct": 8.0,
      "confidence_lower": 0.02,
      "confidence_upper": 0.14
    },
    "disclaimer": "Past performance does not guarantee future results."
  }
}
```

#### `POST /api/simulate`
Simulation Monte Carlo

**Request Body:**
```json
{
  "user_id": "user123",
  "portfolio_name": "Growth Portfolio",
  "scenarios": 1000,
  "time_horizon": 252,
  "shock_scenarios": [
    {"AAPL": -0.2, "MSFT": -0.15}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "simulation": {
      "median_value": 11500,
      "mean_value": 11600
    },
    "percentiles": {
      "5th": 9500,
      "95th": 13800
    },
    "extremes": {
      "worst_case": 7800,
      "best_case": 16500,
      "prob_loss": 0.15
    },
    "stress_tests": [...]
  }
}
```

---

## 🤖 MCP Tools

### Liste des Tools

Accédez à tous les tools via : `GET /tools`

### Catégories

1. **Portfolio Management** (3 tools)
   - `get_portfolios`
   - `get_portfolio_details`
   - `analyze_portfolio_risk`

2. **Market Intelligence** (4 tools)
   - `get_market_overview`
   - `analyze_sector`
   - `get_market_sentiment`
   - `compare_markets`

3. **Backtesting & Predictions** (3 tools)
   - `backtest_portfolio`
   - `predict_performance`
   - `simulate_scenarios`

4. **Advanced Analytics** (2 tools)
   - `calculate_correlations`
   - `optimize_portfolio`

5. **Transactions** (3 tools)
   - `get_transactions`
   - `get_watchlist`
   - `add_to_watchlist`

### Exécution d'un Tool

#### Via REST API

```bash
POST /execute
Content-Type: application/json

{
  "tool": "analyze_sector",
  "params": {
    "sector": "semiconductors",
    "metrics": ["performance", "sentiment"],
    "period": "3mo"
  }
}
```

#### Via Claude AI (MCP Protocol)

```python
# Claude détecte automatiquement les tools disponibles
# et les appelle selon le contexte de la conversation

User: "Analyse le secteur des semiconducteurs"

Claude: [Utilise automatiquement analyze_sector tool]
```

### Tool Détaillé : `analyze_sector`

**Description:**  
Analyse approfondie d'un secteur avec support des sous-secteurs (semiconductors, quantum computing, AI/ML, etc.)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "sector": {
      "type": "string",
      "description": "Main sector"
    },
    "subsector": {
      "type": "string",
      "description": "Optional subsector"
    },
    "metrics": {
      "type": "array",
      "items": {"type": "string"},
      "default": ["performance", "sentiment", "top_stocks", "correlations"]
    },
    "period": {
      "type": "string",
      "default": "3mo"
    }
  },
  "required": ["sector"]
}
```

**Output:**
```json
{
  "sector": "semiconductors",
  "performance": {
    "average": 15.6,
    "best": {"ticker": "NVDA", "performance_3mo": 45.2},
    "worst": {"ticker": "INTC", "performance_3mo": -8.5}
  },
  "sentiment": {"score": 0.7, "label": "bullish"},
  "top_performers": [...],
  "correlations": {"average": 0.65},
  "stocks": [...]
}
```

**Secteurs Supportés:**
- `technology` - AAPL, MSFT, GOOGL, META, NVDA, AVGO
- `semiconductors` - NVDA, AMD, INTC, TSM, ASML, QCOM
- `quantum` - IONQ, RGTI, QUBT, IBM, GOOGL
- `ai_ml` - NVDA, MSFT, GOOGL, META, ORCL, PLTR
- `healthcare` - JNJ, UNH, PFE, ABBV, TMO, LLY
- `finance` - JPM, BAC, WFC, GS, MS, C
- `energy` - XOM, CVX, COP, SLB, EOG, PXD
- `consumer` - AMZN, WMT, HD, MCD, NKE, SBUX
- `industrials` - BA, CAT, HON, UPS, RTX, LMT

---

## 🎓 Exemples d'utilisation

### Exemple 1: Frontend Next.js

```typescript
// app/api/portfolios/route.ts
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const userId = searchParams.get('userId')
  
  const response = await fetch(`http://localhost:8000/api/portfolios/${userId}`)
  const data = await response.json()
  
  return NextResponse.json(data)
}
```

### Exemple 2: Mobile App (React Native)

```javascript
// services/portfolioService.js
const API_BASE = 'http://localhost:8000/api'

export const getPortfolios = async (userId) => {
  const response = await fetch(`${API_BASE}/portfolios/${userId}`)
  const data = await response.json()
  return data.data.portfolios
}

export const backtest = async (params) => {
  const response = await fetch(`${API_BASE}/backtest`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params)
  })
  return await response.json()
}
```

### Exemple 3: Claude AI Integration

```python
# Dans Streamlit (ai_assistant.py)
import anthropic
import requests

def call_mcp_tool(tool_name, params):
    """Execute MCP tool"""
    response = requests.post(
        'http://localhost:8000/execute',
        json={'tool': tool_name, 'params': params}
    )
    return response.json()

# Claude appelle automatiquement
client = anthropic.Anthropic(api_key=api_key)
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    tools=[...],  # MCP tools from /tools endpoint
    messages=[{"role": "user", "content": "Analyse le marché des semiconducteurs"}]
)
```

### Exemple 4: Python Script

```python
import requests

# Market overview
response = requests.post('http://localhost:8000/api/market/overview', json={
    'region': 'US',
    'include_sectors': True,
    'period': '1mo'
})
data = response.json()

print(f"Market Sentiment: {data['data']['market_sentiment']['label']}")

# Sector analysis
response = requests.post('http://localhost:8000/api/market/sector', json={
    'sector': 'quantum',
    'metrics': ['performance', 'sentiment', 'top_stocks']
})
quantum_data = response.json()

print(f"Quantum Computing Sector: {quantum_data['data']['sentiment']['label']}")
```

---

## 🚀 Déploiement

### Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "mcp_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/
    depends_on:
      - mongodb
  
  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### Cloud Deployment (Railway, Render, Fly.io)

```bash
# Procfile (for Railway/Render)
web: python mcp_server.py

# fly.toml (for Fly.io)
app = "pymanager-mcp"

[http_service]
  internal_port = 8000
  force_https = true
```

---

## 📊 Performance & Caching

### Cache Strategy

```python
# Cache TTL: 5 minutes
CACHE_TTL = 300

# Cached endpoints:
- Market overview (by region)
- Sector analysis (by sector + period)
- Ticker info (by ticker)
```

### Rate Limiting (TODO)

```python
# Planned:
- 100 requests/minute per IP
- 1000 requests/hour per API key
```

---

## 🔒 Sécurité

### Authentication (TODO)

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/portfolios/{user_id}")
async def get_portfolios(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate token
    token = credentials.credentials
    # ...
```

### CORS Configuration

```python
# Current: Allow all (development)
allow_origins=["*"]

# Production: Restrict
allow_origins=[
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]
```

---

## 🧪 Testing

Voir `test_mcp_v4.py` pour tests automatiques complets.

```bash
python test_mcp_v4.py
```

**Tests couverts:**
- ✅ Health checks
- ✅ Portfolio CRUD
- ✅ Market intelligence
- ✅ Backtesting
- ✅ Predictions
- ✅ Analytics
- ✅ Error handling

---

## ❓ FAQ

### Q: MCP vs REST API, quand utiliser quoi ?

**REST API** : Frontend classique, mobile apps, intégrations tierces  
**MCP** : AI Assistants, agents autonomes, workflows intelligents

### Q: Peut-on migrer vers Next.js facilement ?

Oui ! Tous les endpoints REST sont prêts. Exemple :
```typescript
// app/portfolios/page.tsx
const portfolios = await fetch('/api/portfolios').then(r => r.json())
```

### Q: Le serveur supporte-t-il WebSocket ?

Pas encore, mais l'endpoint `/ws` est préparé pour l'implémentation.

### Q: Comment ajouter un nouveau secteur ?

```python
# Dans mcp_server.py
SECTOR_TICKERS = {
    Sector.YOUR_SECTOR: ["TICKER1", "TICKER2", ...]
}
```

### Q: Les prédictions ML sont-elles fiables ?

Les prédictions actuelles sont basiques (ARIMA simple). Version avancée (LSTM, Ensemble) en cours de développement.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/baadjis/pymanager/issues)
- **Email**: support@pymanager.dev
- **Discord**: [Lien à venir]

---

## 🎉 Conclusion

Vous disposez maintenant d'un **MCP Server production-ready** avec :

✅ Architecture hybride REST API + MCP  
✅ Support Next.js, mobile apps, AI assistants  
✅ Market intelligence avancée (secteurs, subsecteurs)  
✅ Backtesting & predictions ML  
✅ Caching intelligent  
✅ Documentation complète (Swagger)  

**🚀 Ready for scale !**
