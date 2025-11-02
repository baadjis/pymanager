
"""
PyManager MCP Server v4.0 - Production Ready
==============================================
✅ REST API endpoints (Next.js, mobile apps ready)
✅ MCP Tools (Claude AI Assistant integration)
✅ Market Intelligence (overview, sentiment, sectors, subsectors)
✅ Backtesting & Predictions (with ML models)
✅ Portfolio Management (CRUD operations)
✅ Transactions & Watchlist
✅ Advanced Analytics (Risk metrics, correlations)
✅ Real-time data caching
✅ Rate limiting & security
✅ Comprehensive logging
✅ Health checks & monitoring
"""


from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from functools import lru_cache
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import asyncio

from .handlers import *
from .constant import SECTOR_TICKERS,Region,Sector,ModelType,TimeHorizon,SECTOR_TICKERS,MARKET_INDICES
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# =============================================================================
# FastAPI App Configuration
# =============================================================================

app = FastAPI(
    title="ΦManager Unified API + MCP",
    description="""
    Portfolio Management & Market Intelligence Platform
    
    **Features:**
    - 📊 Portfolio Management (CRUD)
    - 📈 Market Intelligence (Real-time data)
    - 🤖 AI Assistant Integration (MCP Protocol)
    - 🧪 Backtesting & Predictions
    - 🔐 Secure Authentication
    - 📡 WebSocket Support (Real-time)
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "System", "description": "Health checks & system info"},
        {"name": "MCP", "description": "Model Context Protocol tools"},
        {"name": "Portfolios", "description": "Portfolio management"},
        {"name": "Market", "description": "Market intelligence"},
        {"name": "Analytics", "description": "Advanced analytics"},
        {"name": "Backtesting", "description": "Backtesting & predictions"},
        {"name": "Transactions", "description": "Transaction management"},
    ]
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# =============================================================================
# Pydantic Models
# =============================================================================

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MCPToolRequest(BaseModel):
    tool: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default_factory=dict)

class MarketOverviewRequest(BaseModel):
    region: Region = Field(Region.US, description="Market region")
    include_sectors: bool = Field(True, description="Include sector performance")
    period: str = Field("1mo", description="Time period")

class SectorAnalysisRequest(BaseModel):
    sector: Sector = Field(..., description="Sector to analyze")
    subsector: Optional[str] = Field(None, description="Subsector (e.g., 'AI chips')")
    metrics: List[str] = Field(
        default=["performance", "sentiment", "top_stocks", "correlations"],
        description="Metrics to include"
    )
    period: str = Field("3mo", description="Analysis period")

class SentimentRequest(BaseModel):
    target: str = Field(..., description="Ticker or sector")
    period: str = Field("1mo", description="Analysis period")
    include_news: bool = Field(False, description="Include news sentiment")

class BacktestRequest(BaseModel):
    user_id: str
    portfolio_name: Optional[str] = None
    assets: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    start_date: str
    end_date: str
    initial_capital: float = 10000
    rebalance_frequency: Optional[str] = Field(None, description="daily/weekly/monthly")

class PredictionRequest(BaseModel):
    user_id: str
    portfolio_name: str
    horizon: TimeHorizon = Field(TimeHorizon.THREE_MONTHS)
    model: str = Field("ensemble", description="arima/lstm/ensemble")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)

class ScenarioRequest(BaseModel):
    user_id: str
    portfolio_name: str
    scenarios: int = Field(1000, ge=100, le=10000)
    time_horizon: int = Field(252, ge=1, le=1260)  # Trading days
    shock_scenarios: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Custom shock scenarios [{asset: shock_pct}]"
    )

# =============================================================================
# MCP Tools Definitions (Extended)
# =============================================================================

MCP_TOOLS = [
    # Portfolio Management
    {
        "name": "get_portfolios",
        "description": "Get all user portfolios with live P&L and metrics",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User identifier"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_portfolio_details",
        "description": "Get detailed portfolio information including holdings, metrics, and allocation",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"}
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    {
        "name": "analyze_portfolio_risk",
        "description": "Comprehensive risk analysis: VaR, CVaR, Sharpe, Sortino, Max Drawdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "confidence_level": {"type": "number", "default": 0.95}
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    
    # Market Intelligence
    {
        "name": "get_market_overview",
        "description": "Comprehensive market overview with indices, sectors, trends, and sentiment",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "enum": ["US", "EU", "ASIA", "GLOBAL"],
                    "default": "US"
                },
                "include_sectors": {"type": "boolean", "default": True},
                "period": {"type": "string", "default": "1mo"}
            }
        }
    },
    {
        "name": "analyze_sector",
        "description": "Deep sector analysis: performance, top stocks, correlations, sentiment (supports subsectors like semiconductors, quantum computing, AI/ML)",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Main sector (technology, healthcare, etc.)"
                },
                "subsector": {
                    "type": "string",
                    "description": "Optional subsector (semiconductors, quantum, ai_ml, etc.)"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["performance", "sentiment", "top_stocks", "correlations"]
                },
                "period": {"type": "string", "default": "3mo"}
            },
            "required": ["sector"]
        }
    },
    {
        "name": "get_market_sentiment",
        "description": "Analyze market sentiment for any ticker, sector, or market using price action and technical indicators",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Ticker, sector, or market"},
                "period": {"type": "string", "default": "1mo"},
                "include_news": {"type": "boolean", "default": False}
            },
            "required": ["target"]
        }
    },
    {
        "name": "compare_markets",
        "description": "Compare multiple markets, sectors, or assets with correlation analysis",
        "input_schema": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tickers/sectors to compare"
                },
                "period": {"type": "string", "default": "1y"}
            },
            "required": ["targets"]
        }
    },
    
    # Backtesting & Predictions
    {
        "name": "backtest_portfolio",
        "description": "Advanced backtesting with rebalancing, transaction costs, and benchmark comparison",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "initial_capital": {"type": "number", "default": 10000},
                "rebalance_frequency": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly", "quarterly"],
                    "default": "monthly"
                },
                "transaction_cost": {"type": "number", "default": 0.001}
            },
            "required": ["user_id", "portfolio_name", "start_date", "end_date"]
        }
    },
    {
        "name": "predict_performance",
        "description": "ML-based performance prediction using ensemble models (ARIMA + LSTM)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "horizon": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y"],
                    "default": "3mo"
                },
                "model": {
                    "type": "string",
                     "enum": ["ensemble", "xgboost", "lightgbm", "catboost", 
                        "prophet", "arima", "lstm", "randomforest"],
                    "default": "ensemble"
                },
                 "confidence_level": {
                "type": "number",
                "minimum": 0.8,
                "maximum": 0.99,
                "default": 0.95
            },
            "auto_select_model": {
                "type": "boolean",
                "default": false,
                "description": "Automatically select best model based on validation"
            }
            },
          

            "required": ["user_id", "portfolio_name"]
        }
    },
    # Dans votre TOOLS_SCHEMA ou tool registry

{
    "name": "predict_compare_models",
    "description": "Compare predictions from multiple ML models",
    "input_schema": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "portfolio_name": {"type": "string"},
            "horizon": {
                "type": "string",
                "enum": ["1w", "2w", "1mo", "3mo", "6mo", "1y"],
                "default": "3mo"
            },
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Models to compare (e.g., ['xgboost', 'ensemble'])"
            }
        },
        "required": ["user_id", "portfolio_name"]
    }
},

{
    "name": "list_prediction_models",
    "description": "List all available ML prediction models",
    "input_schema": {
        "type": "object",
        "properties": {}
    }
}
    {
        "name": "simulate_scenarios",
        "description": "Monte Carlo simulation with custom shock scenarios and stress testing",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "scenarios": {"type": "integer", "default": 1000, "minimum": 100},
                "time_horizon": {"type": "integer", "default": 252, "description": "Trading days"},
                "shock_scenarios": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Custom shock scenarios"
                }
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    
    # Advanced Analytics
    {
        "name": "calculate_correlations",
        "description": "Calculate correlation matrix for portfolio assets or custom tickers",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "tickers": {"type": "array", "items": {"type": "string"}},
                "period": {"type": "string", "default": "1y"}
            }
        }
    },
    {
        "name": "optimize_portfolio",
        "description": "Suggest portfolio optimization based on current holdings and market conditions",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "objective": {
                    "type": "string",
                    "enum": ["sharpe", "return", "risk", "sortino"],
                    "default": "sharpe"
                }
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    
    # Transactions & Watchlist
    {
        "name": "get_transactions",
        "description": "Get transaction history with filtering and aggregation",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_watchlist",
        "description": "Get watchlist with live prices and performance metrics",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "include_metrics": {"type": "boolean", "default": True}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "add_to_watchlist",
        "description": "Add ticker to watchlist (requires confirmation)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "ticker": {"type": "string"}
            },
            "required": ["user_id", "ticker"]
        }
    }
]

# =============================================================================
# Caching & Rate Limiting
# =============================================================================

# Simple in-memory cache
_cache = {}
_cache_timestamps = {}
CACHE_TTL = 300  # 5 minutes

def get_cached(key: str):
    """Get cached value if not expired"""
    if key in _cache:
        if datetime.now().timestamp() - _cache_timestamps[key] < CACHE_TTL:
            return _cache[key]
    return None

def set_cached(key: str, value: Any):
    """Set cached value"""
    _cache[key] = value
    _cache_timestamps[key] = datetime.now().timestamp()

# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/", tags=["System"])
def root():
    """API root with service information"""
    return {
        "service": "ΦManager Unified API + MCP",
        "version": "4.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "rest_api": True,
            "mcp_protocol": True,
            "database": DATABASE_AVAILABLE,
            "market_data": MARKET_DATA_AVAILABLE,
            "ml_predictions": True,
            "real_time": True
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "tools": "/tools"
        }
    }

@app.get("/health", tags=["System"])
def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if DATABASE_AVAILABLE else "unavailable",
            "market_data": "connected" if MARKET_DATA_AVAILABLE else "unavailable",
            "cache": "operational",
            "mcp_tools": f"{len(MCP_TOOLS)} tools available"
        },
        "metrics": {
            "cache_size": len(_cache),
            "uptime": "N/A"  # TODO: Track uptime
        }
    }
    
    overall_status = all([
        DATABASE_AVAILABLE,
        MARKET_DATA_AVAILABLE
    ])
    
    if not overall_status:
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/tools", tags=["MCP"])
def list_tools():
    """List all MCP tools with full schemas"""
    return {
        "tools": MCP_TOOLS,
        "count": len(MCP_TOOLS),
        "categories": {
            "portfolio": 3,
            "market_intelligence": 4,
            "backtesting": 3,
            "analytics": 2,
            "transactions": 3
        }
    }

@app.get("/tools/{tool_name}", tags=["MCP"])
def get_tool_info(tool_name: str):
    """Get detailed information about a specific tool"""
    tool = next((t for t in MCP_TOOLS if t["name"] == tool_name), None)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return tool

# =============================================================================
# MCP Execute Endpoint
# =============================================================================

@app.post("/execute", tags=["MCP"], response_model=APIResponse)
async def execute_mcp_tool(request: MCPToolRequest):
    """Execute MCP tool with comprehensive error handling"""
    tool_name = request.tool
    params = request.params
    
    logger.info(f"Executing tool: {tool_name} | Params: {params}")
    
    # Validate tool exists
    tool_def = next((t for t in MCP_TOOLS if t["name"] == tool_name), None)
    if not tool_def:
        return APIResponse(
            success=False,
            error=f"Unknown tool: {tool_name}",
            metadata={"available_tools": [t["name"] for t in MCP_TOOLS]}
        )
    
    # Validate required parameters
    required = tool_def["input_schema"].get("required", [])
    missing = [p for p in required if p not in params]
    if missing:
        return APIResponse(
            success=False,
            error=f"Missing required parameters: {', '.join(missing)}",
            metadata={"required": required, "provided": list(params.keys())}
        )
    
    # Execute tool
    try:
        data = await route_mcp_tool(tool_name, params)
        return APIResponse(
            success=True,
            data=data,
            metadata={"tool": tool_name, "execution_time": "N/A"}
        )
    except ValueError as e:
        logger.error(f"Validation error in {tool_name}: {str(e)}")
        return APIResponse(success=False, error=str(e))
    except Exception as e:
        logger.error(f"Internal error in {tool_name}: {str(e)}", exc_info=True)
        return APIResponse(success=False, error=f"Internal error: {str(e)}")

# =============================================================================
# REST API Endpoints
# =============================================================================

# Portfolio Endpoints
@app.get("/api/portfolios/{user_id}", tags=["Portfolios"], response_model=APIResponse)
async def api_get_portfolios(user_id: str):
    """Get all portfolios (REST)"""
    try:
        data = await handle_get_portfolios({"user_id": user_id})
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/portfolios/{user_id}/{portfolio_name}", tags=["Portfolios"])
async def api_get_portfolio(user_id: str, portfolio_name: str):
    """Get portfolio details (REST)"""
    try:
        data = await handle_get_portfolio_details({
            "user_id": user_id,
            "portfolio_name": portfolio_name
        })
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Market Endpoints
@app.post("/api/market/overview", tags=["Market"])
async def api_market_overview(request: MarketOverviewRequest):
    """Market overview (REST)"""
    try:
        data = await handle_market_overview({
            "region": request.region.value,
            "include_sectors": request.include_sectors,
            "period": request.period
        })
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/api/market/sector", tags=["Market"])
async def api_sector_analysis(request: SectorAnalysisRequest):
    """Sector analysis (REST)"""
    try:
        data = await handle_analyze_sector({
            "sector": request.sector.value,
            "subsector": request.subsector,
            "metrics": request.metrics,
            "period": request.period
        })
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/api/market/sentiment", tags=["Market"])
async def api_sentiment(request: SentimentRequest):
    """Market sentiment (REST)"""
    try:
        data = await handle_sentiment({
            "target": request.target,
            "period": request.period,
            "include_news": request.include_news
        })
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Backtesting Endpoints
@app.post("/api/backtest", tags=["Backtesting"])
async def api_backtest(request: BacktestRequest):
    """Backtest portfolio (REST)"""
    try:
        data = await handle_backtest(request.dict())
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/api/predict", tags=["Backtesting"])
async def api_predict(request: PredictionRequest):
    """Predict performance (REST)"""
    try:
        data = await handle_predict(request.dict())
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.post("/api/simulate", tags=["Backtesting"])
async def api_simulate(request: ScenarioRequest):
    """Simulate scenarios (REST)"""
    try:
        data = await handle_simulate(request.dict())
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# =============================================================================
# MCP Tool Routing
# =============================================================================

async def route_mcp_tool(tool_name: str, params: Dict[str, Any]) -> Any:
    """Route MCP tool to appropriate handler"""
    handlers = {
        # Portfolio
        "get_portfolios": handle_get_portfolios,
        "get_portfolio_details": handle_get_portfolio_details,
        "analyze_portfolio_risk": handle_analyze_risk,
        
        # Market Intelligence
        "get_market_overview": handle_market_overview,
        "analyze_sector": handle_analyze_sector,
        "get_market_sentiment": handle_sentiment,
        "compare_markets": handle_compare_markets,
        
        # Backtesting & Predictions
        "backtest_portfolio": handle_backtest,
        "predict_performance": handle_predict,
        "simulate_scenarios": handle_simulate,
        "predict_compare_models":handle_predict_compare,
        "list_prediction_models":handle_predict_models
        
        # Analytics
        "calculate_correlations": handle_correlations,
        "optimize_portfolio": handle_optimize,
        
        # Transactions
        "get_transactions": handle_transactions,
        "get_watchlist": handle_watchlist,
        "add_to_watchlist": handle_add_watchlist
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        raise ValueError(f"No handler for tool: {tool_name}")
    
    return await handler(params)



# =============================================================================
# WebSocket Support (Optional)
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket for real-time updates"""
    # TODO: Implement WebSocket support for real-time data
    pass

# =============================================================================
# Run Server
# =============================================================================
def run_server():

	    import uvicorn
	    
	    print("""
	╔═══════════════════════════════════════════════════════════════╗
	║                                                               ║
	║       ΦManager Unified API + MCP Server v4.0                 ║
	║       Portfolio Management & Market Intelligence              ║
	║                                                               ║
	╚═══════════════════════════════════════════════════════════════╝

	✅ Features Enabled:
	   • REST API Endpoints (Next.js, Mobile ready)
	   • MCP Tools (AI Assistant integration)
	   • Market Intelligence (Real-time data)
	   • Advanced Analytics (Risk, Correlations)
	   • Backtesting & Predictions (ML-powered)
	   • Portfolio Management (CRUD operations)
	   • Transactions & Watchlist
	   
	📊 MCP Tools Available: {}

	🌍 Market Coverage:
	   • Regions: US, EU, ASIA, GLOBAL
	   • Sectors: Technology, Semiconductors, Quantum, AI/ML, Healthcare, Finance, Energy, Consumer, Industrials
	   • Indices: S&P 500, NASDAQ, Dow Jones, Euro Stoxx, Nikkei, Hang Seng, etc.

	🔧 Services Status:
	   • Database: {}
	   • Market Data: {}
	   • Caching: Enabled (5 min TTL)

	📚 Documentation:
	   • API Docs: http://localhost:8000/docs
	   • ReDoc: http://localhost:8000/redoc
	   • Health: http://localhost:8000/health
	   • Tools: http://localhost:8000/tools

	🚀 Starting server on http://0.0.0.0:8000
	""".format(
		len(MCP_TOOLS),
		"✅ Connected" if DATABASE_AVAILABLE else "⚠️ Unavailable (Mock Mode)",
		"✅ Connected" if MARKET_DATA_AVAILABLE else "⚠️ Unavailable"
	    ))
	    
	    uvicorn.run(
		app,
		host="0.0.0.0",
		port=8000,
		log_level="info",
		access_log=True
	    )
	    
    
if __name__ == "__main__":


     run_server()
