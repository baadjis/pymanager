# mcp_server.py
"""
MCP Server pour PyManager - Version Finale
Gère uniquement les données internes (portfolios, transactions, watchlist)
GET operations: pas de confirmation
WRITE operations: confirmation côté client (Streamlit)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database imports
try:
    from database import (
        get_portfolios,
        get_single_portfolio,
        get_transactions,
        get_watchlist,
        save_portfolio,
        add_to_watchlist,
        remove_from_watchlist
    )
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Database module not available, using mock data")
    DATABASE_AVAILABLE = False
    
    # Mock functions
    def get_portfolios(user_id):
        return []
    def get_single_portfolio(user_id, portfolio_name):
        return None
    def get_transactions(user_id):
        return []
    def get_watchlist(user_id):
        return []
    def save_portfolio(user_id, portfolio, name, model, amount, **kwargs):
        return {"_id": "mock_id"}
    def add_to_watchlist(user_id, ticker):
        return True
    def remove_from_watchlist(user_id, ticker):
        return True

app = FastAPI(
    title="PyManager MCP Server",
    description="Model Context Protocol - Internal Data Only",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
    tool: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters")
    require_approval: bool = Field(default=False, description="Require approval")
    
    @validator('tool')
    def validate_tool(cls, v):
        valid_tools = [
            "get_portfolios",
            "get_portfolio_by_name",
            "get_transactions",
            "get_watchlist",
            "calculate_portfolio_metrics",
            "get_allocation_breakdown",
            "save_portfolio",
            "add_to_watchlist",
            "remove_from_watchlist"
        ]
        if v not in valid_tools:
            raise ValueError(f"Invalid tool: {v}")
        return v

class MCPToolResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# =============================================================================
# Tool Definitions
# =============================================================================

MCP_TOOLS = [
    {
        "name": "get_portfolios",
        "description": "Retrieve all user portfolios",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"}
            },
            "required": ["user_id"]
        },
        "requires_approval": False
    },
    {
        "name": "get_portfolio_by_name",
        "description": "Get specific portfolio by name",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
                "portfolio_name": {"type": "string", "description": "Portfolio name"}
            },
            "required": ["user_id", "portfolio_name"]
        },
        "requires_approval": False
    },
    {
        "name": "get_transactions",
        "description": "Retrieve transaction history",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
                "portfolio_name": {"type": "string", "description": "Filter by portfolio (optional)"}
            },
            "required": ["user_id"]
        },
        "requires_approval": False
    },
    {
        "name": "get_watchlist",
        "description": "Retrieve user watchlist",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"}
            },
            "required": ["user_id"]
        },
        "requires_approval": False
    },
    {
        "name": "calculate_portfolio_metrics",
        "description": "Calculate portfolio performance metrics",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
                "portfolio_name": {"type": "string", "description": "Portfolio name"}
            },
            "required": ["user_id", "portfolio_name"]
        },
        "requires_approval": False
    },
    {
        "name": "get_allocation_breakdown",
        "description": "Get asset allocation breakdown",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID"},
                "portfolio_name": {"type": "string", "description": "Portfolio name (optional)"}
            },
            "required": ["user_id"]
        },
        "requires_approval": False
    },
    {
        "name": "save_portfolio",
        "description": "Save a new portfolio (requires confirmation)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "portfolio_data": {"type": "object"},
                "name": {"type": "string"},
                "model": {"type": "string"},
                "amount": {"type": "number"}
            },
            "required": ["user_id", "portfolio_data", "name", "model", "amount"]
        },
        "requires_approval": True
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
        },
        "requires_approval": True
    },
    {
        "name": "remove_from_watchlist",
        "description": "Remove ticker from watchlist (requires confirmation)",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "ticker": {"type": "string"}
            },
            "required": ["user_id", "ticker"]
        },
        "requires_approval": True
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
        "version": "2.0.0",
        "status": "running",
        "database": "connected" if DATABASE_AVAILABLE else "mock",
        "tools_count": len(MCP_TOOLS)
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected" if DATABASE_AVAILABLE else "mock",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools")
def list_tools():
    """List all available tools"""
    return {"tools": MCP_TOOLS}

@app.get("/tools/{tool_name}")
def get_tool_info(tool_name: str):
    """Get tool details"""
    tool = next((t for t in MCP_TOOLS if t["name"] == tool_name), None)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return tool

@app.post("/execute", response_model=MCPToolResponse)
async def execute_tool(request: MCPToolRequest):
    """Execute a tool"""
    tool_name = request.tool
    params = request.params
    
    logger.info(f"Executing: {tool_name} with params: {params}")
    
    # Valider les paramètres requis
    tool_def = next((t for t in MCP_TOOLS if t["name"] == tool_name), None)
    if not tool_def:
        return MCPToolResponse(
            success=False,
            error=f"Tool '{tool_name}' not found"
        )
    
    required_params = tool_def["input_schema"].get("required", [])
    missing_params = [p for p in required_params if p not in params]
    
    if missing_params:
        return MCPToolResponse(
            success=False,
            error=f"Missing required parameters: {', '.join(missing_params)}"
        )
    
    # Exécuter le tool
    try:
        data = await execute_tool_handler(tool_name, params)
        
        return MCPToolResponse(
            success=True,
            data=data,
            metadata={"tool": tool_name}
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return MCPToolResponse(
            success=False,
            error=str(e)
        )
    
    except Exception as e:
        logger.error(f"Internal error: {str(e)}", exc_info=True)
        return MCPToolResponse(
            success=False,
            error=f"Internal error: {str(e)}"
        )

# =============================================================================
# Tool Handlers
# =============================================================================

async def execute_tool_handler(tool_name: str, params: Dict[str, Any]) -> Any:
    """Route to handler"""
    handlers = {
        "get_portfolios": handle_get_portfolios,
        "get_portfolio_by_name": handle_get_portfolio_by_name,
        "get_transactions": handle_get_transactions,
        "get_watchlist": handle_get_watchlist,
        "calculate_portfolio_metrics": handle_calculate_metrics,
        "get_allocation_breakdown": handle_allocation_breakdown,
        "save_portfolio": handle_save_portfolio,
        "add_to_watchlist": handle_add_to_watchlist,
        "remove_from_watchlist": handle_remove_from_watchlist
    }
    
    handler = handlers.get(tool_name)
    if not handler:
        raise ValueError(f"No handler for tool '{tool_name}'")
    
    return await handler(params)

async def handle_get_portfolios(params: Dict[str, Any]):
    """Get all portfolios"""
    user_id = params.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    
    try:
        portfolios = list(get_portfolios(str(user_id)))
        total_value = sum([p.get('amount', 0) for p in portfolios])
        
        return {
            "portfolios": portfolios,
            "count": len(portfolios),
            "total_value": total_value,
            "summary": {
                "total_portfolios": len(portfolios),
                "total_assets": total_value,
                "average_value": total_value / len(portfolios) if portfolios else 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolios: {str(e)}")
        raise ValueError(f"Failed to retrieve portfolios: {str(e)}")

async def handle_get_portfolio_by_name(params: Dict[str, Any]):
    """Get specific portfolio"""
    user_id = params.get("user_id")
    portfolio_name = params.get("portfolio_name")
    
    if not user_id or not portfolio_name:
        raise ValueError("user_id and portfolio_name are required")
    
    try:
        portfolio = get_single_portfolio(str(user_id), portfolio_name)
        
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        
        return {
            "portfolio": portfolio,
            "name": portfolio.get('name'),
            "assets": portfolio.get('assets', []),
            "weights": portfolio.get('weights', []),
            "amount": portfolio.get('amount', 0),
            "model": portfolio.get('model', 'unknown')
        }
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        raise ValueError(f"Failed to retrieve portfolio: {str(e)}")

async def handle_get_transactions(params: Dict[str, Any]):
    """Get transactions"""
    user_id = params.get("user_id")
    portfolio_name = params.get("portfolio_name")
    
    if not user_id:
        raise ValueError("user_id is required")
    
    try:
        transactions = list(get_transactions(str(user_id)))
        
        # Filter by portfolio if specified
        if portfolio_name:
            transactions = [t for t in transactions if t.get('portfolio_name') == portfolio_name]
        
        # Calculate aggregates
        total_buy = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'buy'])
        total_sell = sum([t.get('amount', 0) for t in transactions if t.get('type') == 'sell'])
        
        return {
            "transactions": transactions,
            "count": len(transactions),
            "summary": {
                "total_buy": total_buy,
                "total_sell": total_sell,
                "net_flow": total_buy - total_sell
            }
        }
    except Exception as e:
        logger.error(f"Error getting transactions: {str(e)}")
        raise ValueError(f"Failed to retrieve transactions: {str(e)}")

async def handle_get_watchlist(params: Dict[str, Any]):
    """Get watchlist"""
    user_id = params.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    
    try:
        watchlist = list(get_watchlist(str(user_id)))
        
        return {
            "watchlist": watchlist,
            "count": len(watchlist),
            "tickers": [item.get('ticker') for item in watchlist if 'ticker' in item]
        }
    except Exception as e:
        logger.error(f"Error getting watchlist: {str(e)}")
        raise ValueError(f"Failed to retrieve watchlist: {str(e)}")

async def handle_calculate_metrics(params: Dict[str, Any]):
    """Calculate portfolio metrics"""
    user_id = params.get("user_id")
    portfolio_name = params.get("portfolio_name")
    
    if not user_id or not portfolio_name:
        raise ValueError("user_id and portfolio_name are required")
    
    try:
        portfolio = get_single_portfolio(str(user_id), portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        
        # Get transactions for this portfolio
        transactions = list(get_transactions(str(user_id)))
        portfolio_transactions = [t for t in transactions if t.get('portfolio_name') == portfolio_name]
        
        # Calculate metrics
        current_value = portfolio.get('amount', 0)
        invested = sum([t.get('amount', 0) for t in portfolio_transactions if t.get('type') == 'buy'])
        withdrawn = sum([t.get('amount', 0) for t in portfolio_transactions if t.get('type') == 'sell'])
        
        cost_basis = invested - withdrawn
        total_return = current_value - cost_basis if cost_basis > 0 else 0
        total_return_pct = (total_return / cost_basis * 100) if cost_basis > 0 else 0
        
        return {
            "portfolio_name": portfolio_name,
            "current_value": current_value,
            "cost_basis": cost_basis,
            "total_return": total_return,
            "total_return_percentage": total_return_pct,
            "transaction_count": len(portfolio_transactions)
        }
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise ValueError(f"Failed to calculate metrics: {str(e)}")

async def handle_allocation_breakdown(params: Dict[str, Any]):
    """Get allocation breakdown"""
    user_id = params.get("user_id")
    portfolio_name = params.get("portfolio_name")
    
    if not user_id:
        raise ValueError("user_id is required")
    
    try:
        if portfolio_name:
            portfolio = get_single_portfolio(str(user_id), portfolio_name)
            if not portfolio:
                raise ValueError(f"Portfolio '{portfolio_name}' not found")
            
            assets = portfolio.get('assets', [])
            weights = portfolio.get('weights', [])
            amount = portfolio.get('amount', 0)
            
            allocation = {asset: weight * amount for asset, weight in zip(assets, weights)}
        else:
            # All portfolios
            portfolios = list(get_portfolios(str(user_id)))
            allocation = {}
            
            for pf in portfolios:
                assets = pf.get('assets', [])
                weights = pf.get('weights', [])
                amount = pf.get('amount', 0)
                
                for asset, weight in zip(assets, weights):
                    allocation[asset] = allocation.get(asset, 0) + (weight * amount)
        
        total_value = sum(allocation.values())
        allocation_pct = {
            asset: (value / total_value * 100) if total_value > 0 else 0
            for asset, value in allocation.items()
        }
        
        return {
            "allocation": allocation,
            "allocation_percentage": allocation_pct,
            "total_value": total_value,
            "asset_count": len(allocation)
        }
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error calculating allocation: {str(e)}")
        raise ValueError(f"Failed to calculate allocation: {str(e)}")

async def handle_save_portfolio(params: Dict[str, Any]):
    """Save portfolio (WRITE - requires confirmation)"""
    user_id = params.get("user_id")
    portfolio_data = params.get("portfolio_data")
    name = params.get("name")
    model = params.get("model")
    amount = params.get("amount")
    
    if not all([user_id, portfolio_data, name, model, amount]):
        raise ValueError("Missing required parameters")
    
    try:
        # Note: L'objet Portfolio doit être créé côté client
        # Ici on reçoit juste les données et on sauvegarde
        result = save_portfolio(
            str(user_id),
            portfolio_data,
            name,
            model,
            amount
        )
        
        return {
            "success": True,
            "portfolio_id": str(result.get('_id', 'unknown')),
            "message": f"Portfolio '{name}' saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving portfolio: {str(e)}")
        raise ValueError(f"Failed to save portfolio: {str(e)}")

async def handle_add_to_watchlist(params: Dict[str, Any]):
    """Add to watchlist (WRITE - requires confirmation)"""
    user_id = params.get("user_id")
    ticker = params.get("ticker")
    
    if not user_id or not ticker:
        raise ValueError("user_id and ticker are required")
    
    try:
        result = add_to_watchlist(str(user_id), ticker.upper())
        
        return {
            "success": True,
            "ticker": ticker.upper(),
            "message": f"{ticker} added to watchlist"
        }
    except Exception as e:
        logger.error(f"Error adding to watchlist: {str(e)}")
        raise ValueError(f"Failed to add to watchlist: {str(e)}")

async def handle_remove_from_watchlist(params: Dict[str, Any]):
    """Remove from watchlist (WRITE - requires confirmation)"""
    user_id = params.get("user_id")
    ticker = params.get("ticker")
    
    if not user_id or not ticker:
        raise ValueError("user_id and ticker are required")
    
    try:
        result = remove_from_watchlist(str(user_id), ticker.upper())
        
        return {
            "success": True,
            "ticker": ticker.upper(),
            "message": f"{ticker} removed from watchlist"
        }
    except Exception as e:
        logger.error(f"Error removing from watchlist: {str(e)}")
        raise ValueError(f"Failed to remove from watchlist: {str(e)}")

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
╔═══════════════════════════════════════════════╗
║   PyManager MCP Server v2.0                   ║
║   Internal Data Management                     ║
╚═══════════════════════════════════════════════╝

GET Operations (no confirmation):
  ✓ get_portfolios
  ✓ get_transactions
  ✓ get_watchlist
  ✓ calculate_metrics
  ✓ get_allocation

WRITE Operations (requires confirmation):
  ⚠ save_portfolio
  ⚠ add_to_watchlist
  ⚠ remove_from_watchlist

API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health

Starting server...
""")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
