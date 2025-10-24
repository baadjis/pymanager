# 🤖 PyManager - AI-Powered Portfolio Management

> Intelligent portfolio management with multi-agent AI orchestration and Model Context Protocol (MCP) integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Claude AI](https://img.shields.io/badge/Claude-Sonnet%204.5-purple.svg)](https://www.anthropic.com/claude)

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements_mcp.txt
```

### 2. Configure
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
MCP_SERVER_URL = "http://localhost:8000"
```

### 3. Run
```bash
# Linux/Mac
./start.sh

# Windows
start.bat
```

### 4. Use
Open http://localhost:8501 and navigate to **🤖 AI Assistant**

---

## ✨ Features

🧠 **AI-Powered Analysis** - Natural language queries with Claude AI  
📊 **Portfolio Management** - Track multiple portfolios and positions  
🔍 **Company Research** - Fundamental analysis and valuation  
📈 **Stock Screening** - Find stocks matching your criteria  
📝 **Smart Reports** - Automated performance reporting  
🎓 **Financial Education** - Learn investment concepts  
🔧 **MCP Integration** - Seamless internal data access  

---

## 🏗️ Architecture

```
User Query → Φ AI Orchestrator → Specialized Agents
                    ↓
        ┌───────────┴────────────┐
        ↓                        ↓
   MCP Server              Claude API
   (Internal Data)         (Analysis)
        ↓                        ↓
   PostgreSQL/SQLite      Yahoo Finance
```

**6 Specialized Agents:**
- 🔧 MCP Agent - Internal data
- 🔍 Research Agent - Company analysis
- 📈 Screening Agent - Stock discovery
- 📝 Report Agent - Documentation
- 🎓 Education Agent - Concepts
- 💬 General Agent - Conversations

---

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Full Setup](MCP_SETUP.md)** - Complete installation and configuration
- **[Project Overview](PROJECT_OVERVIEW.md)** - Architecture and features
- **[API Documentation](PROJECT_OVERVIEW.md#api-documentation)** - MCP endpoints

---

## 💬 Example Queries

### Portfolio Analysis
```
"Analyze my portfolio performance"
"What's my asset allocation?"
"Calculate my Sharpe ratio"
```

### Company Research
```
"Research Apple stock (AAPL)"
"Compare Microsoft vs Google"
"Is Tesla overvalued?"
```

### Stock Screening
```
"Find high-dividend technology stocks"
"Show me undervalued growth stocks"
"Screen for P/E under 15"
```

### Education
```
"Explain diversification"
"What is Beta?"
"Teach me about Modern Portfolio Theory"
```

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

### Manual Testing
```bash
# Test MCP Server
curl http://localhost:8000/

# Test tool execution
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_portfolios", "params": {}}'
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| AI Engine | Anthropic Claude Sonnet 4.5 |
| Backend | FastAPI (MCP Server) |
| Frontend | Streamlit |
| Database | SQLite/PostgreSQL |
| Market Data | Yahoo Finance (yfinance) |
| Protocol | Model Context Protocol (MCP) |

---

## 📦 Project Structure

```
pymanager/
├── app.py                          # Main Streamlit app
├── pages/
│   ├── ai_assistant_enhanced.py   # AI Assistant with MCP
│   └── ...
├── mcp_server.py                   # FastAPI MCP server
├── database.py                     # Database layer
├── dataprovider/
│   └── yahoo.py                    # Market data
├── test_mcp_integration.py         # Test suite
├── requirements_mcp.txt            # Dependencies
├── start.sh                        # Linux/Mac startup
├── start.bat                       # Windows startup
└── docs/
    ├── QUICKSTART.md
    ├── MCP_SETUP.md
    └── PROJECT_OVERVIEW.md
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your-key

# Optional
MCP_SERVER_URL=http://localhost:8000
DATABASE_URL=sqlite:///pymanager.db
```

### Get API Key

1. Visit https://console.anthropic.com/
2. Sign up/login
3. Go to API Keys section
4. Create new key
5. Copy to `.streamlit/secrets.toml`

---

## 🌐 Deployment

### Docker
```bash
docker-compose up -d
```

### Production
- **MCP Server**: Railway, Render, Fly.io
- **Streamlit App**: Streamlit Cloud
- See [MCP_SETUP.md](MCP_SETUP.md#deployment) for details

---

## 🐛 Troubleshooting

### MCP Server Issues
```bash
# Check if running
curl http://localhost:8000/

# View logs
tail -f mcp_server.log

# Restart
./start.sh
```

### API Key Issues
- Verify key in `.streamlit/secrets.toml`
- Check format: `sk-ant-...`
- Ensure quotes are present

### Port Conflicts
```bash
# Linux/Mac
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr ":8000"
taskkill /F /PID <PID>
```

See [PROJECT_OVERVIEW.md#troubleshooting](PROJECT_OVERVIEW.md#troubleshooting) for more.

---

## 📈 Roadmap

- [x] Multi-agent AI orchestration
- [x] MCP server integration
- [x] Portfolio management
- [x] Company research
- [x] Stock screening
- [x] Financial education
- [ ] Real-time alerts
- [ ] Portfolio optimization
- [ ] Backtesting engine
- [ ] Mobile app
- [ ] Options trading
- [ ] Social features

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

### Development Setup
```bash
# Clone repo
git clone https://github.com/yourusername/pymanager.git
cd pymanager

# Install dev dependencies
pip install -r requirements_mcp.txt
pip install pytest black flake8

# Run tests
python test_mcp_integration.py --full

# Format code
black .
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Anthropic** - Claude AI platform
- **FastAPI** - Modern web framework
- **Streamlit** - Beautiful UI framework
- **Yahoo Finance** - Market data
- **Community** - Contributors and users

---

## 📞 Support

- **Documentation**: See `/docs` folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/pymanager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pymanager/discussions)

---

## 🌟 Star History

If you find PyManager useful, please consider starring the repository!

---

## 📊 Status

- ✅ Core Features: Stable
- ✅ MCP Integration: Beta
- ✅ AI Agents: Production Ready
- 🚧 Advanced Features: In Development

---

## 💡 Tips

**Get Better Results:**
- Be specific in queries
- Provide context when needed
- Use follow-up questions
- Explore different agents

**Optimize Performance:**
- Cache frequent requests
- Use batch operations
- Monitor API usage
- Review logs regularly

---

## 🎓 Learning Resources

### Tutorials
- [Quick Start](QUICKSTART.md) - 5 minute setup
- [Full Guide](MCP_SETUP.md) - Complete documentation
- [Examples](PROJECT_OVERVIEW.md#example-conversations) - Query examples

### External Links
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 🚦 System Status

| Service | Status |
|---------|--------|
| MCP Server | ✅ Operational |
| AI Assistant | ✅ Operational |
| Market Data | ✅ Operational |
| Documentation | ✅ Complete |

---

<div align="center">

**Built with ❤️ using Claude, FastAPI, and Streamlit**

Φ (Phi) - Your Intelligent Portfolio Advisor

[Get Started](QUICKSTART.md) • [Documentation](PROJECT_OVERVIEW.md) • [Report Bug](https://github.com/yourusername/pymanager/issues)

</div>
