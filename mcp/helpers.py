

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, Optional, List, Union

from .constant import SECTOR_TICKERS,Region,Sector,ModelType,TimeHorizon,SECTOR_TICKERS,MARKET_INDICES
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


try:
    from dataprovider import yahoo
    from utils import calculate_portfolio_current_value, get_portfolio_summary
    MARKET_DATA_AVAILABLE = True
    logger.info("✅ Market data provider connected")
except ImportError as e:
    MARKET_DATA_AVAILABLE = False
    logger.warning(f"⚠️ Market data unavailable: {e}")
    class yahoo:
        @staticmethod
        def get_ticker_data(ticker, period='1y'): return pd.DataFrame()
        @staticmethod
        def get_ticker_info(ticker): return {}
        @staticmethod
        def retrieve_data(tickers, period='1y'): return pd.DataFrame()



# =============================================================================
# Helper Functions
# =============================================================================

async def calculate_correlation_matrix(tickers: List[str], period: str = "1y") -> Dict[str, float]:
    """Calculate pairwise correlations"""
    returns_data = []
    valid_tickers = []
    
    for ticker in tickers:
        try:
            data = yahoo.get_ticker_data(ticker, period=period)
            if data is not None and not data.empty:
                returns = data['Close'].pct_change().dropna()
                returns_data.append(returns.values)
                valid_tickers.append(ticker)
        except:
            continue
    
    if len(returns_data) < 2:
        return {}
    
    # Align lengths
    min_len = min(len(r) for r in returns_data)
    aligned = [r[-min_len:] for r in returns_data]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(aligned)
    
    # Convert to dict
    correlations = {}
    for i, t1 in enumerate(valid_tickers):
        for j, t2 in enumerate(valid_tickers):
            if i < j:
                correlations[f"{t1}_vs_{t2}"] = float(corr_matrix[i, j])
    
    return correlations

def calculate_overall_sentiment(indices_data: List[Dict]) -> Dict[str, Any]:
    """Calculate overall market sentiment from indices"""
    if not indices_data:
        return {"label": "unknown", "score": 0}
    
    avg_change = np.mean([idx.get('change_1mo', 0) for idx in indices_data])
    
    if avg_change > 3:
        label = "very bullish"
    elif avg_change > 1:
        label = "bullish"
    elif avg_change < -3:
        label = "very bearish"
    elif avg_change < -1:
        label = "bearish"
    else:
        label = "neutral"
    
    return {
        "label": label,
        "score": float(avg_change),
        "bullish_indices": sum(1 for idx in indices_data if idx.get('change_1mo', 0) > 2),
        "bearish_indices": sum(1 for idx in indices_data if idx.get('change_1mo', 0) < -2)
    }

async def get_sectors_performance(period: str = "1mo") -> List[Dict[str, Any]]:
    """Get performance for all sectors"""
    sectors_perf = []
    
    for sector, tickers in SECTOR_TICKERS.items():
        try:
            # Sample 3 tickers per sector for speed
            sample_tickers = tickers[:3]
            
            performances = []
            for ticker in sample_tickers:
                try:
                    data = yahoo.get_ticker_data(ticker, period='3mo')
                    if data is not None and not data.empty:
                        if period == "1mo":
                            idx = max(0, len(data) - 21)
                        else:
                            idx = 0
                        
                        current = float(data['Close'].iloc[-1])
                        prev = float(data['Close'].iloc[idx])
                        perf = (current - prev) / prev * 100
                        performances.append(perf)
                except:
                    continue
            
            if performances:
                avg_perf = np.mean(performances)
                sectors_perf.append({
                    "sector": sector.value,
                    "performance": float(avg_perf),
                    "sentiment": "positive" if avg_perf > 2 else "negative" if avg_perf < -2 else "neutral"
                })
        except:
            continue
    
    return sorted(sectors_perf, key=lambda x: x['performance'], reverse=True)

def get_rebalance_days(dates: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
    """Get rebalancing dates based on frequency"""
    if frequency == "daily":
        return dates.tolist()
    elif frequency == "weekly":
        return [d for d in dates if d.dayofweek == 0]  # Mondays
    elif frequency == "monthly":
        return [d for d in dates if d.is_month_start or (d.day == 1)]
    elif frequency == "quarterly":
        return [d for d in dates if d.month in [1, 4, 7, 10] and d.is_month_start]
    else:
        return []


