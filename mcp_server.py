# mcp_server.py
"""
FastAPI MCP Server for PyManager Internal Data
Provides tools for AI to access portfolios, watchlist, and transactions
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

try:
    from database import get_portfolios, get_transactions, get_watchlist
except:
    def get_portfolios():
        return []
    def get_transactions():
        return []
    def get_watchlist():
        return []

app = FastAPI(
    title="PyManager MCP Server",
    description="Model Context Protocol server for PyManager internal data",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Models
# =============================================================================

class MCPToolRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = {}

class MCPToolResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

# =============================================================================
# MCP Tool Definitions
# =============================================================================

MCP_TOOLS = [
    {
        "name": "get_portfolios",
        "description": "Retrieve all user portfolios with their current values and allocations",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_portfolio_by_id",
        "description": "Get detailed information about a specific portfolio",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "The ID of the portfolio to retrieve"
                }
            },
            "required": ["portfolio_id"]
        }
    },
    {
        "name": "get_watchlist",
        "description": "Retrieve user's watchlist stocks",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_transactions",
        "description": "Retrieve transaction history with optional filters",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Filter by portfolio ID (optional)"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date for transactions (YYYY-MM-DD)"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date for transactions (YYYY-MM-DD)"
                }
            },
            "required": []
        }
    },
    {
        "name": "calculate_portfolio_metrics",
        "description": "Calculate performance metrics for a portfolio",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Portfolio ID to analyze"
                }
            },
            "required": ["portfolio_id"]
        }
    },
    {
        "name": "get_allocation_breakdown",
        "description": "Get asset allocation breakdown across all portfolios or a specific one",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Specific portfolio ID (optional, defaults to all)"
                }
            },
            "required": []
        }
    }
]

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    """Health check"""
    return {
        "service": "PyManager MCP Server",
        "status": "running",
        "version": "1.0.0",
        "tools_count": len(MCP_TOOLS)
    }

@app.get("/tools")
def list_tools():
    """List all available MCP tools"""
    return {"tools": MCP_TOOLS}

@app.post("/execute", response_model=MCPToolResponse)
async def execute_tool(request: MCPToolRequest):
    """Execute an MCP tool"""
    
    tool_name = request.tool
    params = request.params
    
    try:
        # Route to appropriate handler
        if tool_name == "get_portfolios":
            data = handle_get_portfolios()
        elif tool_name == "get_portfolio_by_id":
            data = handle_get_portfolio_by_id(params.get("portfolio_id"))
        elif tool_name == "get_watchlist":
            data = handle_get_watchlist()
        elif tool_name == "get_transactions":
            data = handle_get_transactions(
                params.get("portfolio_id"),
                params.get("start_date"),
                params.get("end_date")
            )
        elif tool_name == "calculate_portfolio_metrics":
            data = handle_calculate_metrics(params.get("portfolio_id"))
        elif tool_name == "get_allocation_breakdown":
            data = handle_allocation_breakdown(params.get("portfolio_id"))
        else:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return MCPToolResponse(success=True, data=data)
    
    except Exception as e:
        return MCPToolResponse(success=False, error=str(e))

# =============================================================================
# Tool Handlers
# =============================================================================

def handle_get_portfolios():
    """Get all portfolios"""
    portfolios = list(get_portfolios())
    
    total_value = sum([p.get('amount', 0) for p in portfolios])
    
    return {
        "portfolios": portfolios,
        "count": len(portfolios),
        "total_value": total_value,
        "summary": {
            "total_portfolios": len(portfolios),
            "total_assets": total_value,
            "average_portfolio_size": total_value / len(portfolios) if portfolios else 0
        }
    }

def handle_get_portfolio_by_id(portfolio_id: str):
    """Get specific portfolio"""
    if not portfolio_id:
        raise ValueError("portfolio_id is required")
    
    portfolios = list(get_portfolios())
    portfolio = next((p for p in portfolios if p.get('id') == portfolio_id), None)
    
    if not portfolio:
        raise ValueError(f"Portfolio {portfolio_id} not found")
    
    return {
        "portfolio": portfolio,
        "holdings": portfolio.get('holdings', []),
        "value": portfolio.get('amount', 0),
        "model": portfolio.get('model', 'unknown')
    }

def handle_get_watchlist():
    """Get watchlist"""
    watchlist = get_watchlist()
    
    return {
        "watchlist": watchlist,
        "count": len(watchlist),
        "tickers": [item.get('ticker') for item in watchlist]
    }

def handle_get_transactions(portfolio_id=None, start_date=None, end_date=None):
    """Get transactions with filters"""
    transactions = list(get_transactions())
    
    # Filter by portfolio
    if portfolio_id:
        transactions = [t for t in transactions if t.get('portfolio_id') == portfolio_id]
    
    # Filter by date range
    if start_date:
        start = datetime.fromisoformat(start_date)
        transactions = [t for t in transactions if datetime.fromisoformat(t.get('date', '')) >= start]
    
    if end_date:
        end = datetime.fromisoformat(end_date)
        transactions = [t for t in transactions if datetime.fromisoformat(t.get('date', '')) <= end]
    
    # Calculate aggregates
    total_buy = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'buy'])
    total_sell = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'sell'])
    
    return {
        "transactions": transactions,
        "count": len(transactions),
        "summary": {
            "total_transactions": len(transactions),
            "total_buy_amount": total_buy,
            "total_sell_amount": total_sell,
            "net_flow": total_buy - total_sell
        }
    }

def handle_calculate_metrics(portfolio_id: str):
    """Calculate portfolio metrics"""
    if not portfolio_id:
        raise ValueError("portfolio_id is required")
    
    portfolio = handle_get_portfolio_by_id(portfolio_id)['portfolio']
    transactions = handle_get_transactions(portfolio_id)['transactions']
    
    # Calculate basic metrics
    current_value = portfolio.get('amount', 0)
    invested = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'buy'])
    withdrawn = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'sell'])
    
    total_return = current_value - (invested - withdrawn)
    total_return_pct = (total_return / invested * 100) if invested > 0 else 0
    
    return {
        "portfolio_id": portfolio_id,
        "current_value": current_value,
        "total_invested": invested,
        "total_withdrawn": withdrawn,
        "total_return": total_return,
        "total_return_percentage": total_return_pct,
        "metrics": {
            "value": current_value,
            "cost_basis": invested - withdrawn,
            "gain_loss": total_return,
            "return_pct": total_return_pct
        }
    }

def handle_allocation_breakdown(portfolio_id=None):
    """Get allocation breakdown"""
    if portfolio_id:
        portfolio = handle_get_portfolio_by_id(portfolio_id)['portfolio']
        holdings = portfolio.get('holdings', [])
    else:
        portfolios = list(get_portfolios())
        holdings = []
        for p in portfolios:
            holdings.extend(p.get('holdings', []))
    
    # Group by asset type
    allocation = {}
    total_value = 0
    
    for holding in holdings:
        asset_type = holding.get('type', 'unknown')
        value = holding.get('value', 0)
        allocation[asset_type] = allocation.get(asset_type, 0) + value
        total_value += value
    
    # Calculate percentages
    allocation_pct = {
        asset: (value / total_value * 100) if total_value > 0 else 0
        for asset, value in allocation.items()
    }
    
    return {
        "allocation": allocation,
        "allocation_percentage": allocation_pct,
        "total_value": total_value,
        "diversification_score": len(allocation)
    }

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
