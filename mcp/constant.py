from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum

# =============================================================================
# Enums & Constants
# =============================================================================

class Region(str, Enum):
    US = "US"
    EU = "EU"
    ASIA = "ASIA"
    GLOBAL = "GLOBAL"

class Sector(str, Enum):
    TECHNOLOGY = "technology"
    SEMICONDUCTORS = "semiconductors"
    QUANTUM = "quantum"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    ENERGY = "energy"
    CONSUMER = "consumer"
    INDUSTRIALS = "industrials"
    AI_ML = "ai_ml"

class ModelType(str, Enum):
    MARKOWITZ = "markowitz"
    ML = "ml"
    RL = "rl"
    BLACK_LITTERMAN = "black_litterman"
    DISCRETIONARY = "discretionary"

class TimeHorizon(str, Enum):
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"

# Market Indices
MARKET_INDICES = {
    Region.US: [
        ("^GSPC", "S&P 500"),
        ("^DJI", "Dow Jones"),
        ("^IXIC", "NASDAQ"),
        ("^RUT", "Russell 2000")
    ],
    Region.EU: [
        ("^STOXX50E", "Euro Stoxx 50"),
        ("^FTSE", "FTSE 100"),
        ("^GDAXI", "DAX"),
        ("^FCHI", "CAC 40")
    ],
    Region.ASIA: [
        ("^N225", "Nikkei 225"),
        ("^HSI", "Hang Seng"),
        ("000001.SS", "SSE Composite"),
        ("^STI", "STI Index")
    ],
    Region.GLOBAL: [
        ("^GSPC", "S&P 500"),
        ("^STOXX50E", "Euro Stoxx"),
        ("^N225", "Nikkei 225")
    ]
}

# Sector Mappings
SECTOR_TICKERS = {
    Sector.TECHNOLOGY: ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO"],
    Sector.SEMICONDUCTORS: ["NVDA", "AMD", "INTC", "TSM", "ASML", "QCOM"],
    Sector.QUANTUM: ["IONQ", "RGTI", "QUBT", "IBM", "GOOGL"],  # Quantum computing
    Sector.AI_ML: ["NVDA", "MSFT", "GOOGL", "META", "ORCL", "PLTR"],
    Sector.HEALTHCARE: ["JNJ", "UNH", "PFE", "ABBV", "TMO", "LLY"],
    Sector.FINANCE: ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    Sector.ENERGY: ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"],
    Sector.CONSUMER: ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX"],
    Sector.INDUSTRIALS: ["BA", "CAT", "HON", "UPS", "RTX", "LMT"]
}


