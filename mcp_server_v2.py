"""
PyManager MCP Server v3.0 - Unified API + MCP Tools
✅ REST API endpoints (for Next.js, mobile apps)
✅ MCP Tools (for Claude AI Assistant)
✅ Market Intelligence (overview, sentiment, sectors)
✅ Backtesting & Predictions
✅ Portfolio Management
✅ Transactions & Watchlist
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database imports
try:
    from database import (
        get_portfolios, get_single_portfolio, get_transactions,
        get_watchlist, save_portfolio, add_to_watchlist, remove_from_watchlist
    )
    from utils import calculate_portfolio_current_value, get_portfolio_summary
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("⚠️ Database unavailable - mock mode")
    def get_portfolios(user_id): return []
    def get_single_portfolio(user_id, name): return None
    def get_transactions(user_id): return []
    def get_watchlist(user_id): return []
    def save_portfolio(*args, **kwargs): return {"_id": "mock"}
    def add_to_watchlist(user_id, ticker): return True
    def remove_from_watchlist(user_id, ticker): return True
    def calculate_portfolio_current_value(pf): return (0, 0, 0, [])
    def get_portfolio_summary(pf): return {}

# Market data provider
try:
    from dataprovider import yahoo
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False
    logger.warning("⚠️ Market data unavailable")
    class yahoo:
        @staticmethod
        def get_ticker_data(ticker, period='1y'): return pd.DataFrame()
        @staticmethod
        def get_ticker_info(ticker): return {}
        @staticmethod
        def retrieve_data(tickers, period='1y'): return pd.DataFrame()

app = FastAPI(
    title="PyManager Unified API + MCP",
    description="Portfolio Management, Market Intelligence & AI Tools",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class MCPToolRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MarketSentimentRequest(BaseModel):
    market: str = Field(..., description="Market/ticker to analyze")
    period: str = Field("1mo", description="Time period")

class BacktestRequest(BaseModel):
    user_id: str
    portfolio_name: Optional[str] = None
    assets: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    start_date: str
    end_date: str
    initial_capital: float = 10000

# =============================================================================
# MCP TOOLS DEFINITIONS (Extended)
# =============================================================================

MCP_TOOLS = [
    # Portfolio Management
    {
        "name": "get_portfolios",
        "description": "Get all user portfolios with live P&L",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"]
        }
    },
    {
        "name": "get_portfolio_details",
        "description": "Get detailed portfolio info with holdings",
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
        "description": "Analyze portfolio risk metrics (VaR, Sharpe, etc)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"}
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    
    # Market Intelligence
    {
        "name": "get_market_overview",
        "description": "Get market overview (indices, sectors, trends)",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["US", "EU", "ASIA", "GLOBAL"], "default": "US"},
                "include_sectors": {"type": "boolean", "default": True}
            }
        }
    },
    {
        "name": "analyze_sector",
        "description": "Deep dive into a specific sector/subsector",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {"type": "string", "description": "e.g., Technology, Healthcare, Semiconductors"},
                "metrics": {"type": "array", "items": {"type": "string"}, "default": ["performance", "sentiment", "top_stocks"]}
            },
            "required": ["sector"]
        }
    },
    {
        "name": "get_market_sentiment",
        "description": "Analyze market sentiment for ticker/sector",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Ticker or sector"},
                "period": {"type": "string", "default": "1mo"}
            },
            "required": ["target"]
        }
    },
    {
        "name": "compare_markets",
        "description": "Compare multiple markets/sectors",
        "input_schema": {
            "type": "object",
            "properties": {
                "markets": {"type": "array", "items": {"type": "string"}},
                "period": {"type": "string", "default": "1y"}
            },
            "required": ["markets"]
        }
    },
    
    # Backtesting & Predictions
    {
        "name": "backtest_portfolio",
        "description": "Backtest portfolio strategy",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "initial_capital": {"type": "number", "default": 10000}
            },
            "required": ["user_id", "portfolio_name", "start_date", "end_date"]
        }
    },
    {
        "name": "predict_performance",
        "description": "Predict portfolio performance (ML-based)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "horizon": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y"], "default": "3mo"}
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    {
        "name": "simulate_scenarios",
        "description": "Run Monte Carlo scenarios",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"},
                "scenarios": {"type": "integer", "default": 1000},
                "time_horizon": {"type": "integer", "default": 252}
            },
            "required": ["user_id", "portfolio_name"]
        }
    },
    
    # Transactions & Watchlist
    {
        "name": "get_transactions",
        "description": "Get transaction history",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_name": {"type": "string"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "get_watchlist",
        "description": "Get user watchlist with live prices",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"]
        }
    },
    {
        "name": "add_to_watchlist",
        "description": "Add ticker to watchlist",
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
# REST API ENDPOINTS
# =============================================================================

@app.get("/", tags=["System"])
def root():
    return {
        "service": "PyManager Unified API + MCP",
        "version": "3.0.0",
        "status": "running",
        "features": {
            "database": DATABASE_AVAILABLE,
            "market_data": MARKET_DATA_AVAILABLE,
            "mcp_tools": len(MCP_TOOLS),
            "rest_api": True
        }
    }

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if DATABASE_AVAILABLE else "unavailable",
        "market_data": "connected" if MARKET_DATA_AVAILABLE else "unavailable"
    }

@app.get("/tools", tags=["MCP"])
def list_tools():
    """List all MCP tools"""
    return {"tools": MCP_TOOLS, "count": len(MCP_TOOLS)}

@app.post("/execute", tags=["MCP"], response_model=APIResponse)
async def execute_mcp_tool(request: MCPToolRequest):
    """Execute MCP tool"""
    try:
        data = await route_mcp_tool(request.tool, request.params)
        return APIResponse(success=True, data=data)
    except Exception as e:
        logger.error(f"MCP error: {e}")
        return APIResponse(success=False, error=str(e))

# Portfolio Endpoints
@app.get("/api/portfolios/{user_id}", tags=["Portfolios"])
def api_get_portfolios(user_id: str):
    """Get all portfolios (REST API)"""
    try:
        portfolios = list(get_portfolios(user_id))
        summaries = [get_portfolio_summary(pf) for pf in portfolios]
        total_value = sum(s['current_value'] for s in summaries)
        total_pnl = sum(s['pnl'] for s in summaries)
        
        return APIResponse(
            success=True,
            data={
                "portfolios": summaries,
                "count": len(summaries),
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_pct": (total_pnl / sum(s['initial_amount'] for s in summaries) * 100) if summaries else 0
            }
        )
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/portfolios/{user_id}/{portfolio_name}", tags=["Portfolios"])
def api_get_portfolio(user_id: str, portfolio_name: str):
    """Get portfolio details (REST API)"""
    try:
        pf = get_single_portfolio(user_id, portfolio_name)
        if not pf:
            raise HTTPException(404, "Portfolio not found")
        
        summary = get_portfolio_summary(pf)
        return APIResponse(success=True, data=summary)
    except HTTPException:
        raise
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Market Endpoints
@app.get("/api/market/overview", tags=["Market"])
def api_market_overview(region: str = "US"):
    """Market overview (REST API)"""
    try:
        data = get_market_overview_data(region)
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/market/sector/{sector}", tags=["Market"])
def api_sector_analysis(sector: str):
    """Sector analysis (REST API)"""
    try:
        data = analyze_sector_data(sector)
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/market/sentiment/{target}", tags=["Market"])
def api_market_sentiment(target: str, period: str = "1mo"):
    """Market sentiment (REST API)"""
    try:
        data = analyze_sentiment(target, period)
        return APIResponse(success=True, data=data)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# Backtesting Endpoints
@app.post("/api/backtest", tags=["Backtesting"])
def api_backtest(request: BacktestRequest):
    """Backtest portfolio (REST API)"""
    try:
        results = run_backtest(
            request.user_id,
            request.portfolio_name,
            request.assets,
            request.weights,
            request.start_date,
            request.end_date,
            request.initial_capital
        )
        return APIResponse(success=True, data=results)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

@app.get("/api/predict/{user_id}/{portfolio_name}", tags=["Predictions"])
def api_predict(user_id: str, portfolio_name: str, horizon: str = "3mo"):
    """Predict performance (REST API)"""
    try:
        prediction = predict_portfolio_performance(user_id, portfolio_name, horizon)
        return APIResponse(success=True, data=prediction)
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# =============================================================================
# MCP TOOL ROUTING
# =============================================================================

async def route_mcp_tool(tool_name: str, params: Dict[str, Any]) -> Any:
    """Route MCP tool to handler"""
    handlers = {
        "get_portfolios": handle_get_portfolios,
        "get_portfolio_details": handle_get_portfolio_details,
        "analyze_portfolio_risk": handle_analyze_risk,
        "get_market_overview": handle_market_overview,
        "analyze_sector": handle_analyze_sector,
        "get_market_sentiment": handle_sentiment,
        "compare_markets": handle_compare_markets,
        "backtest_portfolio": handle_backtest,
        "predict_performance": handle_predict,
        "simulate_scenarios": handle_simulate,
        "get_transactions": handle_transactions,
        "get_watchlist": handle_watchlist,
        "add_to_watchlist": handle_add_watchlist
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return await handler(params)

# =============================================================================
# PORTFOLIO HANDLERS
# =============================================================================

async def handle_get_portfolios(params):
    user_id = params.get("user_id")
    portfolios = list(get_portfolios(user_id))
    summaries = [get_portfolio_summary(pf) for pf in portfolios]
    
    total_value = sum(s['current_value'] for s in summaries)
    total_pnl = sum(s['pnl'] for s in summaries)
    
    return {
        "portfolios": summaries,
        "count": len(summaries),
        "total_value": total_value,
        "total_pnl": total_pnl
    }

async def handle_get_portfolio_details(params):
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError("Portfolio not found")
    
    return get_portfolio_summary(pf)

async def handle_analyze_risk(params):
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError("Portfolio not found")
    
    summary = get_portfolio_summary(pf)
    holdings = summary['holdings']
    
    # Calculate risk metrics
    returns = []
    for h in holdings:
        try:
            data = yahoo.get_ticker_data(h['symbol'], period='1y')
            if data is not None and not data.empty:
                ret = data['Close'].pct_change().dropna()
                returns.append(ret.values)
        except:
            continue
    
    if returns:
        returns_matrix = np.array(returns).T
        portfolio_returns = returns_matrix @ np.array([h['weight'] for h in holdings])
        
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (np.mean(portfolio_returns) * 252) / volatility if volatility > 0 else 0
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return {
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(calculate_max_drawdown(portfolio_returns))
        }
    
    return {"error": "Insufficient data"}

def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# =============================================================================
# MARKET INTELLIGENCE HANDLERS
# =============================================================================

async def handle_market_overview(params):
    region = params.get("region", "US")
    return get_market_overview_data(region)

def get_market_overview_data(region):
    indices = {
        "US": [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ")],
        "EU": [("^STOXX50E", "Euro Stoxx 50"), ("^FTSE", "FTSE 100")],
        "ASIA": [("^N225", "Nikkei 225"), ("^HSI", "Hang Seng")],
        "GLOBAL": [("^GSPC", "S&P 500"), ("^STOXX50E", "Euro Stoxx"), ("^N225", "Nikkei")]
    }
    
    results = []
    for ticker, name in indices.get(region, indices["US"]):
        try:
            data = yahoo.get_ticker_data(ticker, period='1mo')
            if data is not None and not data.empty:
                current = float(data['Close'].iloc[-1])
                prev = float(data['Close'].iloc[0])
                change_pct = (current - prev) / prev * 100
                
                results.append({
                    "name": name,
                    "ticker": ticker,
                    "price": current,
                    "change_1mo": change_pct,
                    "sentiment": "bullish" if change_pct > 2 else "bearish" if change_pct < -2 else "neutral"
                })
        except:
            continue
    
    return {"region": region, "indices": results, "timestamp": datetime.now().isoformat()}

async def handle_analyze_sector(params):
    sector = params.get("sector")
    return analyze_sector_data(sector)

def analyze_sector_data(sector):
    # Mapping sectors to representative tickers
    sector_map = {
        "technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "semiconductors": ["NVDA", "AMD", "INTC", "TSM"],
        "healthcare": ["JNJ", "UNH", "PFE", "ABBV"],
        "finance": ["JPM", "BAC", "GS", "MS"],
        "energy": ["XOM", "CVX", "COP", "SLB"],
        "consumer": ["AMZN", "WMT", "HD", "NKE"]
    }
    
    tickers = sector_map.get(sector.lower(), sector_map["technology"])
    
    stocks = []
    for ticker in tickers:
        try:
            data = yahoo.get_ticker_data(ticker, period='3mo')
            info = yahoo.get_ticker_info(ticker)
            
            if data is not None and not data.empty:
                current = float(data['Close'].iloc[-1])
                start = float(data['Close'].iloc[0])
                perf = (current - start) / start * 100
                
                stocks.append({
                    "ticker": ticker,
                    "name": info.get('longName', ticker),
                    "price": current,
                    "performance_3mo": perf,
                    "market_cap": info.get('marketCap', 0),
                    "pe_ratio": info.get('trailingPE', 0)
                })
        except:
            continue
    
    avg_perf = np.mean([s['performance_3mo'] for s in stocks]) if stocks else 0
    
    return {
        "sector": sector,
        "stocks": stocks,
        "average_performance": avg_perf,
        "sentiment": "strong" if avg_perf > 10 else "weak" if avg_perf < -5 else "neutral"
    }

async def handle_sentiment(params):
    target = params.get("target")
    period = params.get("period", "1mo")
    return analyze_sentiment(target, period)

def analyze_sentiment(target, period):
    try:
        data = yahoo.get_ticker_data(target, period=period)
        if data is None or data.empty:
            return {"error": "No data"}
        
        returns = data['Close'].pct_change().dropna()
        
        # Simple sentiment based on price action
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        
        avg_return = returns.mean()
        volatility = returns.std()
        
        sentiment_score = (positive_days - negative_days) / len(returns)
        
        if sentiment_score > 0.2:
            sentiment = "very bullish"
        elif sentiment_score > 0.05:
            sentiment = "bullish"
        elif sentiment_score < -0.2:
            sentiment = "very bearish"
        elif sentiment_score < -0.05:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            "target": target,
            "period": period,
            "sentiment": sentiment,
            "sentiment_score": float(sentiment_score),
            "positive_days": int(positive_days),
            "negative_days": int(negative_days),
            "average_return": float(avg_return),
            "volatility": float(volatility)
        }
    except Exception as e:
        return {"error": str(e)}

async def handle_compare_markets(params):
    markets = params.get("markets", [])
    period = params.get("period", "1y")
    
    results = []
    for market in markets:
        try:
            data = yahoo.get_ticker_data(market, period=period)
            if data is not None and not data.empty:
                current = float(data['Close'].iloc[-1])
                start = float(data['Close'].iloc[0])
                perf = (current - start) / start * 100
                vol = float(data['Close'].pct_change().std() * np.sqrt(252))
                
                results.append({
                    "market": market,
                    "performance": perf,
                    "volatility": vol,
                    "sharpe": perf / vol if vol > 0 else 0
                })
        except:
            continue
    
    return {"comparison": results, "period": period}

# =============================================================================
# BACKTESTING & PREDICTIONS
# =============================================================================

async def handle_backtest(params):
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    start = params.get("start_date")
    end = params.get("end_date")
    capital = params.get("initial_capital", 10000)
    
    return run_backtest(user_id, name, None, None, start, end, capital)

def run_backtest(user_id, portfolio_name, assets, weights, start_date, end_date, initial_capital):
    if portfolio_name:
        pf = get_single_portfolio(user_id, portfolio_name)
        if not pf:
            raise ValueError("Portfolio not found")
        summary = get_portfolio_summary(pf)
        assets = [h['symbol'] for h in summary['holdings']]
        weights = [h['weight'] for h in summary['holdings']]
    
    if not assets or not weights:
        raise ValueError("No assets/weights")
    
    try:
        data = yahoo.retrieve_data(tuple(assets), period='max')
        data = data.loc[start_date:end_date]
        
        if data.empty:
            raise ValueError("No data for period")
        
        returns = data.pct_change().fillna(0)
        
        if isinstance(returns.columns, pd.MultiIndex):
            portfolio_returns = sum(returns[('Adj Close', asset)] * weight for asset, weight in zip(assets, weights))
        else:
            portfolio_returns = returns @ np.array(weights)
        
        portfolio_values = initial_capital * (1 + portfolio_returns).cumprod()
        
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        max_dd = calculate_max_drawdown(portfolio_returns.values)
        
        return {
            "initial_capital": initial_capital,
            "final_value": float(portfolio_values.iloc[-1]),
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "start_date": start_date,
            "end_date": end_date
        }
    except Exception as e:
        raise ValueError(f"Backtest failed: {e}")

async def handle_predict(params):
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    horizon = params.get("horizon", "3mo")
    
    return predict_portfolio_performance(user_id, name, horizon)

def predict_portfolio_performance(user_id, portfolio_name, horizon):
    pf = get_single_portfolio(user_id, portfolio_name)
    if not pf:
        raise ValueError("Portfolio not found")
    
    summary = get_portfolio_summary(pf)
    assets = [h['symbol'] for h in summary['holdings']]
    weights = [h['weight'] for h in summary['holdings']]
    
    # Simple prediction based on historical trends
    horizon_days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}[horizon]
    
    try:
        data = yahoo.retrieve_data(tuple(assets), period='1y')
        returns = data.pct_change().fillna(0)
        
        if isinstance(returns.columns, pd.MultiIndex):
            portfolio_returns = sum(returns[('Adj Close', asset)] * weight for asset, weight in zip(assets, weights))
        else:
            portfolio_returns = returns @ np.array(weights)
        
        avg_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        predicted_return = avg_return * horizon_days
        confidence_interval = 1.96 * volatility * np.sqrt(horizon_days)
        
        return {
            "horizon": horizon,
            "predicted_return": float(predicted_return),
            "confidence_lower": float(predicted_return - confidence_interval),
            "confidence_upper": float(predicted_return + confidence_interval),
            "disclaimer": "Prediction based on historical data. Not financial advice."
        }
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

async def handle_simulate(params):
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    scenarios = params.get("scenarios", 1000)
    horizon = params.get("time_horizon", 252)
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError("Portfolio not found")
    
    summary = get_portfolio_summary(pf)
    current_value = summary['current_value']
    
    # Monte Carlo simulation
    returns = np.random.normal(0.0008, 0.015, (horizon, scenarios))
    paths = current_value * (1 + returns).cumprod(axis=0)
    
    final_values = paths[-1, :]
    
    return {
        "scenarios": scenarios,
        "horizon_days": horizon,
        "median_value": float(np.median(final_values)),
        "mean_value": float(np.mean(final_values)),
        "percentile_5": float(np.percentile(final_values, 5)),
        "percentile_95": float(np.percentile(final_values, 95)),
        "worst_case": float(np.min(final_values)),
        "best_case": float(np.max(final_values))
    }

# =============================================================================
# TRANSACTIONS & WATCHLIST
# =============================================================================

async def handle_transactions(params):
    user_id = params.get("user_id")
    transactions = list(get_transactions(user_id))
    return {"transactions": transactions, "count": len(transactions)}

async def handle_watchlist(params):
    user_id = params.get("user_id")
    watchlist = list(get_watchlist(user_id))
    
    # Enrich with live prices
    enriched = []
    for item in watchlist:
        ticker = item.get('ticker')
        try:
            data = yahoo.get_ticker_data(ticker, period='1d')
            if data is not None and not data.empty:
                price = float(data['Close'].iloc[-1])
                enriched.append({**item, "price": price})
        except:
            enriched.append(item)
    
    return {"watchlist": enriched, "count": len
