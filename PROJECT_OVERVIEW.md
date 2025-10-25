# 🚀 PyManager - AI-Powered Portfolio Management Platform

Complete multi-agent AI system with Model Context Protocol (MCP) integration for intelligent portfolio management.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Agent System](#agent-system)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

PyManager is an intelligent portfolio management platform that combines:

- **Multi-agent AI orchestration** for complex financial queries
- **Model Context Protocol (MCP)** for internal data access
- **Claude AI integration** for natural language understanding
- **Real-time market data** from Yahoo Finance
- **Interactive Streamlit interface** for ease of use

### Key Benefits

✅ **Intelligent Analysis** - AI-powered insights on portfolios and stocks  
✅ **Natural Language** - Ask questions in plain English  
✅ **Real-time Data** - Live market data and quotes  
✅ **Multi-agent System** - Specialized agents for different tasks  
✅ **Extensible** - Easy to add new tools and agents  

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Φ AI Assistant                          │
│                   (Main Orchestrator)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼────────┐         ┌────────▼────────┐
│  MCP Server    │         │  Anthropic API  │
│  (FastAPI)     │         │  (Claude AI)    │
└───────┬────────┘         └────────┬────────┘
        │                           │
┌───────▼────────┐         ┌────────▼────────┐
│   Database     │         │  Yahoo Finance  │
│  (SQLite)      │         │   (Market Data) │
└────────────────┘         └─────────────────┘
```

### Component Breakdown

#### 1. **AI Assistant (Orchestrator)**
- Routes queries to appropriate agents
- Combines results from multiple sources
- Manages conversation context
- File: `ai_assistant_enhanced.py`

#### 2. **MCP Server (FastAPI)**
- Provides tools for internal data access
- Handles portfolio, watchlist, transactions
- RESTful API with JSON responses
- File: `mcp_server.py`

#### 3. **Agent System**
- **MCP Agent**: Internal data retrieval
- **Research Agent**: Company analysis
- **Screening Agent**: Stock filtering
- **Report Agent**: Document generation
- **Education Agent**: Financial concepts
- **General Agent**: Fallback handler

#### 4. **Data Layer**
- SQLite database for persistence
- Yahoo Finance for market data
- Caching for performance

---

## ✨ Features

### Portfolio Management
- Create and manage multiple portfolios
- Track holdings and positions
- Calculate returns and metrics
- Asset allocation analysis

### AI-Powered Analysis
- Natural language queries
- Intelligent recommendations
- Multi-source data synthesis
- Context-aware responses

### Company Research
- Fundamental analysis
- Valuation metrics
- Competitive positioning
- Investment thesis generation

### Stock Screening
- Custom filter criteria
- Sector-based screening
- Valuation-based filtering
- Growth/Value strategies

### Financial Education
- Concept explanations
- Investment strategies
- Risk metrics explained
- Best practices

### Reporting
- Portfolio performance reports
- Asset allocation breakdown
- Transaction history
- Recommendations

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Internet connection (for API access)

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/pymanager.git
cd pymanager
```

#### 2. Install Dependencies
```bash
pip install -r requirements_mcp.txt
```

Or install individually:
```bash
pip install streamlit fastapi uvicorn anthropic pydantic requests pandas yfinance
```

#### 3. Configure API Keys

Create `.streamlit/secrets.toml`:
```toml
# Anthropic API Key (Required for AI features)
ANTHROPIC_API_KEY = "sk-ant-your-api-key-here"

# MCP Server URL (Default: localhost)
MCP_SERVER_URL = "http://localhost:8000"
```

**Get your Anthropic API key:** https://console.anthropic.com/

#### 4. Verify Installation

Run quick test:
```bash
python test_mcp_integration.py --quick
```

---

## 🚀 Usage

### Starting the Application

#### Terminal 1: Start MCP Server
```bash
python mcp_server.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Terminal 2: Start Streamlit App
```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

### Basic Usage Examples

#### Portfolio Analysis
```
User: "Analyze my portfolio performance"
AI: [Retrieves portfolio data via MCP]
    [Calculates metrics]
    [Provides insights and recommendations]
```

#### Company Research
```
User: "Research Apple stock (AAPL)"
AI: [Fetches market data from Yahoo Finance]
    [Uses Claude for analysis]
    [Generates investment thesis]
```

#### Stock Screening
```
User: "Find high-dividend technology stocks"
AI: [Extracts screening criteria]
    [Provides guidance on using Screener]
    [Suggests additional filters]
```

#### Financial Education
```
User: "Explain the Sharpe Ratio"
AI: [Retrieves from knowledge base]
    [Provides clear explanation]
    [Includes examples and formulas]
```

---

## 📚 API Documentation

### MCP Server Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "service": "PyManager MCP Server",
  "status": "running",
  "version": "1.0.0",
  "tools_count": 6
}
```

#### List Tools
```http
GET /tools
```

**Response:**
```json
{
  "tools": [
    {
      "name": "get_portfolios",
      "description": "Retrieve all user portfolios",
      "input_schema": {...}
    },
    ...
  ]
}
```

#### Execute Tool
```http
POST /execute
Content-Type: application/json

{
  "tool": "get_portfolios",
  "params": {}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolios": [...],
    "count": 3,
    "total_value": 150000.00
  }
}
```

### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_portfolios` | Get all portfolios | None |
| `get_portfolio_by_id` | Get specific portfolio | `portfolio_id` |
| `get_watchlist` | Get watchlist | None |
| `get_transactions` | Get transaction history | `portfolio_id`, `start_date`, `end_date` (optional) |
| `calculate_portfolio_metrics` | Calculate metrics | `portfolio_id` |
| `get_allocation_breakdown` | Get allocation | `portfolio_id` (optional) |

---

## 🤖 Agent System

### Agent Orchestration Flow

```
User Query
    ↓
Orchestrator analyzes intent
    ↓
┌────────────────────────────────┐
│  Route to appropriate agent(s) │
├────────────────────────────────┤
│  • Portfolio query? → MCP      │
│  • Research query? → Research  │
│  • Screening? → Screening      │
│  • Report? → Report + MCP      │
│  • Education? → Knowledge Base │
│  • General? → Claude           │
└────────────────────────────────┘
    ↓
Execute tool calls
    ↓
Synthesize results
    ↓
Return to user
```

### Agent Descriptions

#### 1. MCP Agent
**Purpose:** Access internal data  
**Tools:** All MCP server tools  
**Use Cases:** Portfolio queries, watchlist, transactions

#### 2. Research Agent
**Purpose:** Company analysis  
**Tools:** Yahoo Finance, Claude AI  
**Use Cases:** Stock research, valuation, fundamentals

#### 3. Screening Agent
**Purpose:** Stock discovery  
**Tools:** Criteria extraction  
**Use Cases:** Find stocks matching criteria

#### 4. Report Agent
**Purpose:** Document generation  
**Tools:** MCP + formatting  
**Use Cases:** Performance reports, summaries

#### 5. Education Agent
**Purpose:** Financial education  
**Tools:** Knowledge base  
**Use Cases:** Concept explanations, tutorials

#### 6. General Agent
**Purpose:** Fallback handler  
**Tools:** Claude AI  
**Use Cases:** General financial questions

---

## 🧪 Testing

### Quick Test
```bash
python test_mcp_integration.py --quick
```

### Full Test Suite
```bash
python test_mcp_integration.py --full
```

### Test Coverage

The test suite covers:

1. ✅ Server health and connectivity
2. ✅ Tool listing and availability
3. ✅ Portfolio data retrieval
4. ✅ Watchlist functionality
5. ✅ Transaction history
6. ✅ Invalid input handling
7. ✅ Allocation calculations
8. ✅ Performance (response time)
9. ✅ AI integration readiness
10. ✅ Error handling

### Manual Testing

#### Test MCP Server
```bash
# Health check
curl http://localhost:8000/

# List tools
curl http://localhost:8000/tools

# Execute tool
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_portfolios", "params": {}}'
```

#### Test AI Assistant

1. Open Streamlit app
2. Navigate to AI Assistant
3. Try example queries:
   - "Show me my portfolios"
   - "Research Microsoft"
   - "Explain diversification"

---

## 🌐 Deployment

### Local Development (Recommended for Start)

```bash
# Terminal 1
python mcp_server.py

# Terminal 2
streamlit run app.py
```

### Docker Deployment

#### Build Images
```bash
# MCP Server
docker build -f Dockerfile.mcp -t pymanager-mcp .

# Streamlit App
docker build -f Dockerfile.streamlit -t pymanager-app .
```

#### Run with Docker Compose
```bash
docker-compose up -d
```

### Production Deployment

#### Option 1: Separate Services

**MCP Server:** Deploy on Railway, Render, or Fly.io
```bash
# Update MCP_SERVER_URL in secrets.toml
MCP_SERVER_URL = "https://your-mcp-server.com"
```

**Streamlit App:** Deploy on Streamlit Cloud
- Connect GitHub repository
- Add secrets in dashboard
- Deploy automatically

#### Option 2: Single Server

Use reverse proxy (nginx) to serve both:
```nginx
location /api/ {
    proxy_pass http://localhost:8000/;
}

location / {
    proxy_pass http://localhost:8501/;
}
```

### Environment Variables

Production secrets:
```bash
ANTHROPIC_API_KEY=sk-ant-xxx
MCP_SERVER_URL=https://mcp.yourdomain.com
DATABASE_URL=sqlite:///production.db
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. MCP Server Won't Start

**Error:** `Address already in use`

**Solution:**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn mcp_server:app --port 8001
```

#### 2. Cannot Connect to MCP Server

**Symptoms:** Red status in AI Assistant

**Solutions:**
- Verify server is running: `curl http://localhost:8000/`
- Check `MCP_SERVER_URL` in secrets.toml
- Check firewall settings
- Review server logs

#### 3. Claude API Errors

**Error:** `Invalid API key` or `Rate limit exceeded`

**Solutions:**
- Verify API key in secrets.toml
- Check API quota at console.anthropic.com
- Ensure model name is correct: `claude-sonnet-4-5-20250929`
- Wait if rate limited (retry after delay)

#### 4. Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements_mcp.txt

# Verify installations
pip list | grep anthropic
pip list | grep fastapi
pip list | grep streamlit
```

#### 5. Database Errors

**Error:** `Database locked` or `Table not found`

**Solutions:**
```bash
# Check database file permissions
ls -la *.db

# Recreate database (WARNING: deletes data)
rm pymanager.db
python database.py  # If you have init script
```

#### 6. Slow Performance

**Symptoms:** Long response times

**Solutions:**
- Check internet connection
- Verify API rate limits not hit
- Review server resource usage
- Consider caching responses
- Optimize database queries

### Debug Mode

Enable debug logging:

```python
# In mcp_server.py
import logging
logging.basicConfig(level=logging.DEBUG)

# In ai_assistant_enhanced.py
import streamlit as st
st.set_option('client.showErrorDetails', True)
```

### Getting Help

1. **Check logs:** Review terminal output for errors
2. **Run tests:** `python test_mcp_integration.py`
3. **Verify setup:** Follow QUICKSTART.md step-by-step
4. **Check docs:** Review MCP_SETUP.md
5. **GitHub Issues:** Open issue with error details

---

## 📈 Performance Tips

### Optimization Strategies

1. **Caching:** Cache Yahoo Finance requests
2. **Database:** Add indexes for frequent queries
3. **API Calls:** Batch requests when possible
4. **Frontend:** Use Streamlit caching decorators
5. **MCP Server:** Deploy close to database

### Monitoring

Track these metrics:
- MCP server response time
- Claude API usage
- Database query time
- Memory usage
- Error rates

---

## 🔒 Security Best Practices

1. **API Keys:** Never commit to version control
2. **Environment:** Use secrets.toml for local, env vars for production
3. **MCP Server:** Add authentication in production
4. **CORS:** Configure properly for your domain
5. **Rate Limiting:** Implement on MCP server
6. **Input Validation:** Sanitize all user inputs
7. **HTTPS:** Use TLS in production

---

## 📝 License

MIT License - See LICENSE file

---

## 🤝 Contributing

We welcome contributions!

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

### Code Style

- Python: Follow PEP 8
- Type hints: Use where appropriate
- Documentation: Update docs for new features
- Tests: Add tests for new functionality

---

## 🎓 Learning Resources

### Documentation
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [MCP Protocol](https://modelcontextprotocol.io/)

### Tutorials
- Check QUICKSTART.md for getting started
- See MCP_SETUP.md for detailed setup
- Review test_mcp_integration.py for examples

---

## 📞 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** support@pymanager.dev (if available)
- **Docs:** This README and linked documentation

---

## 🗺️ Roadmap

### Planned Features

- [ ] Real-time price alerts
- [ ] Advanced portfolio optimization
- [ ] Backtesting engine
- [ ] Social features (share strategies)
- [ ] Mobile app
- [ ] More data sources
- [ ] Advanced charting
- [ ] Options/derivatives support

---

## 🙏 Acknowledgments

- **Anthropic** - Claude AI
- **FastAPI** - Web framework
- **Streamlit** - UI framework
- **Yahoo Finance** - Market data
- **Community** - Contributors and users

---

**Built with ❤️ by the PyManager Team**

Φ (Phi) - Your Intelligent Portfolio Advisor

*Last updated: October 2025*
