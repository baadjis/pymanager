

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


from .helpers import  get_rebalance_days,get_sectors_performance

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Database & External Services
# =============================================================================

try:
    from database import (
        get_portfolios, get_single_portfolio, get_transactions,
        get_watchlist, save_portfolio, add_to_watchlist, 
        remove_from_watchlist, get_user
    )
    DATABASE_AVAILABLE = True
    logger.info("✅ Database connected")
except ImportError as e:
    DATABASE_AVAILABLE = False
    logger.warning(f"⚠️ Database unavailable: {e}")
    # Mock functions
    def get_portfolios(user_id): return []
    def get_single_portfolio(user_id, name): return None
    def get_transactions(user_id): return []
    def get_watchlist(user_id): return []
    def save_portfolio(*args, **kwargs): return {"_id": "mock"}
    def add_to_watchlist(user_id, ticker): return True
    def remove_from_watchlist(user_id, ticker): return True
    def get_user(user_id): return None

try:
    from dataprovider import yahoo
    from utils import calculate_portfolio_current_value, get_portfolio_summary,calculate_overall_sentiment,calculate_correlation_matrix
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
    def calculate_portfolio_current_value(pf): return (0, 0, 0, [])
    def get_portfolio_summary(pf): return {}


# =============================================================================
# HANDLERS - Portfolio Management
# =============================================================================

async def handle_get_portfolios(params):
    """Get all portfolios with enriched data"""
    user_id = params.get("user_id")
    
    try:
        portfolios = list(get_portfolios(user_id))
        summaries = []
        
        for pf in portfolios:
            summary = get_portfolio_summary(pf)
            summaries.append(summary)
        
        total_value = sum(s.get('current_value', 0) for s in summaries)
        total_pnl = sum(s.get('pnl', 0) for s in summaries)
        total_invested = sum(s.get('initial_amount', 0) for s in summaries)
        
        return {
            "portfolios": summaries,
            "count": len(summaries),
            "aggregates": {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_pct": (total_pnl / total_invested * 100) if total_invested > 0 else 0,
                "total_invested": total_invested
            }
        }
    except Exception as e:
        logger.error(f"Error getting portfolios: {e}")
        raise ValueError(f"Failed to get portfolios: {e}")

async def handle_get_portfolio_details(params):
    """Get detailed portfolio with holdings and metrics"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError(f"Portfolio '{name}' not found")
    
    summary = get_portfolio_summary(pf)
    
    # Add advanced metrics
    holdings = summary.get('holdings', [])
    if holdings:
        tickers = [h['symbol'] for h in holdings]
        weights = [h['weight'] for h in holdings]
        
        # Calculate correlations
        try:
            corr_matrix = await calculate_correlation_matrix(tickers, "1y")
            summary['correlations'] = corr_matrix
        except:
            summary['correlations'] = None
    
    return summary

async def handle_analyze_risk(params):
    """Comprehensive risk analysis"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    confidence = params.get("confidence_level", 0.95)
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError(f"Portfolio '{name}' not found")
    
    summary = get_portfolio_summary(pf)
    holdings = summary.get('holdings', [])
    
    if not holdings:
        raise ValueError("Portfolio has no holdings")
    
    # Collect returns data
    returns_data = []
    weights = []
    
    for h in holdings:
        try:
            data = yahoo.get_ticker_data(h['symbol'], period='2y')
            if data is not None and not data.empty:
                returns = data['Close'].pct_change().dropna()
                returns_data.append(returns.values)
                weights.append(h['weight'])
        except Exception as e:
            logger.warning(f"Failed to get data for {h['symbol']}: {e}")
            continue
    
    if not returns_data:
        raise ValueError("Insufficient data for risk analysis")
    
    # Calculate portfolio returns
    returns_matrix = np.array(returns_data).T
    weights_array = np.array(weights) / sum(weights)  # Normalize
    portfolio_returns = returns_matrix @ weights_array
    
    # Risk metrics
    volatility = float(np.std(portfolio_returns) * np.sqrt(252))
    sharpe = float((np.mean(portfolio_returns) * 252) / volatility if volatility > 0 else 0)
    
    # Downside metrics
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_vol = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else 0
    sortino = float((np.mean(portfolio_returns) * 252) / downside_vol if downside_vol > 0 else 0)
    
    # VaR and CVaR
    var_level = 1 - confidence
    var = float(np.percentile(portfolio_returns, var_level * 100))
    cvar = float(portfolio_returns[portfolio_returns <= var].mean())
    
    # Max Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())
    
    # Calmar Ratio
    calmar = float((np.mean(portfolio_returns) * 252) / abs(max_dd)) if max_dd != 0 else 0
    
    return {
        "portfolio_name": name,
        "risk_metrics": {
            "volatility_annual": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "var_95": var,
            "cvar_95": cvar
        },
        "interpretation": {
            "risk_level": "high" if volatility > 0.25 else "medium" if volatility > 0.15 else "low",
            "sharpe_quality": "excellent" if sharpe > 2 else "good" if sharpe > 1 else "poor",
            "drawdown_severity": "severe" if abs(max_dd) > 0.3 else "moderate" if abs(max_dd) > 0.15 else "mild"
        },
        "confidence_level": confidence
    }

# =============================================================================
# HANDLERS - Market Intelligence
# =============================================================================

async def handle_market_overview(params):
    """Comprehensive market overview"""
    region = params.get("region", "US")
    include_sectors = params.get("include_sectors", True)
    period = params.get("period", "1mo")
    
    # Check cache
    cache_key = f"market_overview_{region}_{period}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    indices = MARKET_INDICES.get(region, MARKET_INDICES[Region.US])
    
    indices_data = []
    for ticker, name in indices:
        try:
            data = yahoo.get_ticker_data(ticker, period='3mo')
            if data is not None and not data.empty:
                current = float(data['Close'].iloc[-1])
                
                # Calculate period returns
                if period == "1d":
                    prev = float(data['Close'].iloc[-2]) if len(data) > 1 else current
                elif period == "1mo":
                    prev_idx = max(0, len(data) - 21)
                    prev = float(data['Close'].iloc[prev_idx])
                else:
                    prev = float(data['Close'].iloc[0])
                
                change_pct = (current - prev) / prev * 100
                
                # Calculate volatility
                returns = data['Close'].pct_change().dropna()
                vol = float(returns.std() * np.sqrt(252))
                
                indices_data.append({
                    "name": name,
                    "ticker": ticker,
                    "price": current,
                    f"change_{period}": change_pct,
                    "volatility": vol,
                    "sentiment": "bullish" if change_pct > 2 else "bearish" if change_pct < -2 else "neutral"
                })
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            continue
    
    result = {
        "region": region,
        "period": period,
        "indices": indices_data,
        "market_sentiment": calculate_overall_sentiment(indices_data),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add sector performance if requested
    if include_sectors:
        sectors_perf = await get_sectors_performance(period)
        result["sectors"] = sectors_perf
    
    set_cached(cache_key, result)
    return result

async def handle_analyze_sector(params):
    """Deep sector analysis with subsector support"""
    sector = params.get("sector", "technology")
    subsector = params.get("subsector")
    metrics = params.get("metrics", ["performance", "sentiment", "top_stocks"])
    period = params.get("period", "3mo")
    
    # Determine tickers
    if subsector:
        # Try to find subsector mapping
        subsector_key = Sector(subsector) if subsector in [s.value for s in Sector] else None
        tickers = SECTOR_TICKERS.get(subsector_key, SECTOR_TICKERS.get(Sector(sector), []))
    else:
        tickers = SECTOR_TICKERS.get(Sector(sector), [])
    
    if not tickers:
        raise ValueError(f"No tickers found for sector '{sector}' / subsector '{subsector}'")
    
    stocks_data = []
    returns_list = []
    
    for ticker in tickers:
        try:
            data = yahoo.get_ticker_data(ticker, period='6mo')
            info = yahoo.get_ticker_info(ticker)
            
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                   close_data = data['Close'][data['Close'].columns[0]]
                else:
                   close_data = data['Close']
                current = float(close_data.iloc[-1])
                
                # Period performance
                if period == "1mo":
                    start_idx = max(0, len(data) - 21)
                elif period == "3mo":
                    start_idx = max(0, len(data) - 63)
                else:
                    start_idx = 0
                
                start = float(close_data.iloc[start_idx])
                perf = (current - start) / start * 100
                
                # Returns for correlation
                returns = close_data.pct_change().dropna()
                returns_list.append(returns.values)
                
                stocks_data.append({
                    "ticker": ticker,
                    "name": info.get('longName', ticker),
                    "price": current,
                    f"performance_{period}": perf,
                    "market_cap": info.get('marketCap', 0),
                    "pe_ratio": info.get('trailingPE'),
                    "dividend_yield": info.get('dividendYield'),
                    "sector": info.get('sector'),
                    "industry": info.get('industry')
                })
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            continue
    
    if not stocks_data:
        raise ValueError("No data available for sector analysis")
    
    # Calculate metrics
    result = {
        "sector": sector,
        "subsector": subsector,
        "period": period,
        "timestamp": datetime.now().isoformat()
    }
    
    if "performance" in metrics:
        avg_perf = np.mean([s[f'performance_{period}'] for s in stocks_data])
        result["performance"] = {
            "average": float(avg_perf),
            "best": max(stocks_data, key=lambda x: x[f'performance_{period}']),
            "worst": min(stocks_data, key=lambda x: x[f'performance_{period}'])
        }
    
    if "sentiment" in metrics:
        sentiment_score = np.mean([1 if s[f'performance_{period}'] > 5 else -1 if s[f'performance_{period}'] < -5 else 0 for s in stocks_data])
        result["sentiment"] = {
            "score": float(sentiment_score),
            "label": "bullish" if sentiment_score > 0.3 else "bearish" if sentiment_score < -0.3 else "neutral"
        }
    
    if "top_stocks" in metrics:
        top_5 = sorted(stocks_data, key=lambda x: x[f'performance_{period}'], reverse=True)[:5]
        result["top_performers"] = top_5
    
    if "correlations" in metrics and len(returns_list) > 1:
        corr_matrix = np.corrcoef(returns_list)
        avg_corr = float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        result["correlations"] = {
            "average": avg_corr,
            "diversification": "low" if avg_corr > 0.7 else "medium" if avg_corr > 0.4 else "high"
        }
    
    result["stocks"] = stocks_data
    
    return result

async def handle_sentiment(params):
    """Advanced sentiment analysis with news (FinBERT-embedding)"""
    target = params.get("target")
    period = params.get("period", "1mo")
    include_news = params.get("include_news", False)
    
    try:
        data = yahoo.get_ticker_data(target, period='3mo')
        if data is None or data.empty:
            raise ValueError(f"No data for {target}")
        
        # Gérer MultiIndex ou colonnes simples
        if isinstance(data.columns, pd.MultiIndex):
           close_data = data['Close'][data['Close'].columns[0]]
        else:
           close_data = data['Close']
        
        returns = close_data.pct_change().dropna()
        
        # Price action metrics
        positive_days = (returns > 0).sum()
        negative_days = (returns < 0).sum()
        total_days = len(returns)
        
        # Momentum indicators
        sma_20 = close_data.rolling(20).mean()
        sma_50 = close_data.rolling(50).mean()
        current_price = float(close_data.iloc[-1])
        sma_20_last = sma_20.iloc[-1]
        sma_50_last = sma_50.iloc[-1]

       # Extraire scalar si Series
        if hasattr(sma_20_last, 'iloc'):
               sma_20_last = sma_20_last.iloc[0]
        if hasattr(sma_50_last, 'iloc'):
              sma_50_last = sma_50_last.iloc[0]

        above_sma20 = current_price > float(sma_20_last) if not pd.isna(sma_20_last) else False
        above_sma50 = current_price > float(sma_50_last) if not pd.isna(sma_50_last) else False
        
        
        
        # RSI calculation
        delta = close_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_last = rsi.iloc[-1]
        current_rsi = float(rsi_last.iloc[0]) if hasattr(rsi_last, 'iloc') else float(rsi_last) if not pd.isna(rsi_last) else 50.0
        
        # Technical sentiment score (-1 to 1)
        technical_sentiment_score = (
            (positive_days - negative_days) / total_days * 0.3 +
            (1 if above_sma20 else -1) * 0.2 +
            (1 if above_sma50 else -1) * 0.2 +
            ((current_rsi - 50) / 50) * 0.3
        )
        
        # News sentiment (if requested)
        news_sentiment_score = 0.0
        news_details = None
        
        if include_news:
            try:
                news_sentiment_score, news_details = analyze_news_sentiment_finbert(target)
                
                # Combine technical + news (70% technical, 30% news)
                combined_score = technical_sentiment_score * 0.7 + news_sentiment_score * 0.3
            except Exception as e:
                logger.warning(f"News sentiment failed: {e}")
                combined_score = technical_sentiment_score
        else:
            combined_score = technical_sentiment_score
        
        # Classify sentiment
        if combined_score > 0.4:
            sentiment = "very bullish"
        elif combined_score > 0.15:
            sentiment = "bullish"
        elif combined_score < -0.4:
            sentiment = "very bearish"
        elif combined_score < -0.15:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        result = {
            "target": target,
            "period": period,
            "sentiment": {
                "label": sentiment,
                "score": float(combined_score),
                "confidence": abs(float(combined_score))
            },
            "technical_sentiment": {
                "score": float(technical_sentiment_score),
                "positive_days": int(positive_days),
                "negative_days": int(negative_days),
                "rsi": float(current_rsi),
                "above_sma20": bool(above_sma20),
                "above_sma50": bool(above_sma50)
            },
            "volatility": float(returns.std() * np.sqrt(252)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add news sentiment details
        if include_news and news_details:
            result["news_sentiment"] = news_details
        
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise ValueError(f"Failed to analyze sentiment: {e}")


# =============================================================================
# NEWS SENTIMENT ANALYSIS with finbert-embedding
# =============================================================================

# Global model cache
_finbert_model = None

def get_finbert_model():
    """Load FinBERT-embedding model (lazy loading with cache)"""
    global _finbert_model
    
    if _finbert_model is None:
        try:
            from finbert_embedding.embedding import FinbertEmbedding
            
            logger.info("Loading FinBERT-embedding model...")
            
            # Initialize FinBERT
            _finbert_model = FinbertEmbedding()
            
            logger.info("✅ FinBERT-embedding loaded")
        except Exception as e:
            logger.error(f"Failed to load FinBERT-embedding: {e}")
            return None
    
    return _finbert_model


def analyze_news_sentiment_finbert(ticker: str, max_articles: int = 10) -> tuple:
    """
    Analyze news sentiment using FinBERT-embedding
    
    Returns:
        (sentiment_score, details_dict)
        sentiment_score: float between -1 (bearish) and 1 (bullish)
        details_dict: dict with breakdown
    """
    try:
        # Get news headlines
        news_headlines = fetch_news_headlines(ticker, max_articles)
        
        if not news_headlines:
            return 0.0, {"error": "No news found", "count": 0}
        
        # Get FinBERT model
        model = get_finbert_model()
        
        if model is None:
            # Fallback to simple keyword-based sentiment
            logger.warning("FinBERT unavailable, using keyword fallback")
            return analyze_news_simple(news_headlines)
        
        # Analyze each headline with FinBERT
        sentiments = []
        labels = []
        
        for headline in news_headlines:
            try:
                # Get sentiment prediction
                # FinBERT-embedding returns: positive, negative, neutral probabilities
                result = model.sentence_vector(headline)
                
                # Extract sentiment (finbert-embedding uses different output format)
                # Assuming result is embedding vector, we need to classify
                # For simplicity, we'll use keyword-based for now with finbert context
                
                # Alternative: Use the embedding similarity approach
                sentiment_score = classify_finbert_embedding(result, headline)
                sentiments.append(sentiment_score)
                
                if sentiment_score > 0.2:
                    labels.append("positive")
                elif sentiment_score < -0.2:
                    labels.append("negative")
                else:
                    labels.append("neutral")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze headline: {e}")
                continue
        
        if not sentiments:
            return 0.0, {"error": "No headlines analyzed", "count": 0}
        
        # Calculate average sentiment
        avg_sentiment = float(np.mean(sentiments))
        
        # Count labels
        positive_count = labels.count("positive")
        negative_count = labels.count("negative")
        neutral_count = labels.count("neutral")
        
        details = {
            "score": avg_sentiment,
            "articles_analyzed": len(sentiments),
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "model": "FinBERT-embedding",
            "sample_headlines": news_headlines[:3]
        }
        
        return avg_sentiment, details
    
    except Exception as e:
        logger.error(f"FinBERT sentiment analysis failed: {e}")
        # Fallback
        return analyze_news_simple(fetch_news_headlines(ticker, max_articles))


def classify_finbert_embedding(embedding, text: str) -> float:
    """
    Classify sentiment from FinBERT embedding
    
    Since finbert-embedding gives embeddings, not direct sentiment,
    we use a hybrid approach: keywords + embedding context
    """
    # Define sentiment keywords with weights
    positive_words = {
        'surge': 1.0, 'soar': 1.0, 'rally': 0.9, 'jump': 0.8,
        'gain': 0.7, 'rise': 0.6, 'up': 0.5, 'beat': 0.9,
        'profit': 0.8, 'growth': 0.7, 'strong': 0.7, 'bullish': 1.0,
        'upgrade': 0.8, 'outperform': 0.9, 'record': 0.8, 'high': 0.6,
        'success': 0.7, 'innovation': 0.6, 'expansion': 0.6
    }
    
    negative_words = {
        'plunge': -1.0, 'crash': -1.0, 'tumble': -0.9, 'fall': -0.8,
        'drop': -0.7, 'decline': -0.6, 'down': -0.5, 'miss': -0.9,
        'loss': -0.8, 'weak': -0.7, 'bearish': -1.0, 'downgrade': -0.8,
        'underperform': -0.9, 'cut': -0.7, 'low': -0.6, 'fail': -0.8,
        'concern': -0.6, 'risk': -0.6, 'warning': -0.7, 'layoff': -0.8
    }
    
    text_lower = text.lower()
    
    # Calculate weighted sentiment
    score = 0.0
    word_count = 0
    
    for word, weight in positive_words.items():
        if word in text_lower:
            score += weight
            word_count += 1
    
    for word, weight in negative_words.items():
        if word in text_lower:
            score += weight
            word_count += 1
    
    # Normalize
    if word_count > 0:
        score = score / word_count
    
    # Clip to [-1, 1]
    score = max(-1.0, min(1.0, score))
    
    return score


def fetch_news_headlines(ticker: str, max_articles: int = 10) -> List[str]:
    """
    Fetch recent news headlines for a ticker using yfinance
    """
    try:
        import yfinance as yf
        
        # Get ticker object
        stock = yf.Ticker(ticker)
        
        # Get news
        news = stock.news
        
        if not news:
            logger.warning(f"No news found for {ticker}")
            return []
        
        # Extract headlines
        headlines = []
        for article in news[:max_articles]:
            title = article.get('title', '')
            if title:
                headlines.append(title)
        
        logger.info(f"Found {len(headlines)} headlines for {ticker}")
        return headlines
    
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return []


def analyze_news_simple(headlines: List[str]) -> tuple:
    """
    Simple keyword-based sentiment (fallback)
    Fast and CPU-friendly
    """
    if not headlines:
        return 0.0, {"error": "No headlines", "count": 0}
    
    positive_keywords = [
        'beat', 'surge', 'jump', 'rally', 'gain', 'soar', 'rise', 'up',
        'profit', 'growth', 'strong', 'bullish', 'upgrade', 'outperform',
        'record', 'high', 'success', 'innovation', 'expansion', 'boost'
    ]
    
    negative_keywords = [
        'miss', 'plunge', 'fall', 'drop', 'decline', 'loss', 'weak',
        'bearish', 'downgrade', 'underperform', 'cut', 'low', 'fail',
        'concern', 'risk', 'warning', 'layoff', 'lawsuit', 'crash'
    ]
    
    scores = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for headline in headlines:
        headline_lower = headline.lower()
        
        pos_count = sum(1 for kw in positive_keywords if kw in headline_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in headline_lower)
        
        if pos_count > neg_count:
            scores.append(0.5)
            positive_count += 1
        elif neg_count > pos_count:
            scores.append(-0.5)
            negative_count += 1
        else:
            scores.append(0.0)
            neutral_count += 1
    
    avg_score = float(np.mean(scores)) if scores else 0.0
    
    details = {
        "score": avg_score,
        "articles_analyzed": len(headlines),
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "model": "keyword-based (simple)",
        "sample_headlines": headlines[:3]
    }
    
    return avg_score, details
async def handle_compare_markets(params):
    """Compare multiple markets/assets"""
    targets = params.get("targets", [])
    period = params.get("period", "1y")
    
    if not targets or len(targets) < 2:
        raise ValueError("At least 2 targets required for comparison")
    
    comparison_data = []
    returns_list = []
    
    for target in targets:
        try:
            data = yahoo.get_ticker_data(target, period=period)
            if data is not None and not data.empty:
                current = float(data['Close'].iloc[-1])
                start = float(data['Close'].iloc[0])
                perf = (current - start) / start * 100
                
                returns = data['Close'].pct_change().dropna()
                returns_list.append(returns.values)
                
                vol = float(returns.std() * np.sqrt(252))
                sharpe = float((returns.mean() * 252) / vol) if vol > 0 else 0
                
                comparison_data.append({
                    "target": target,
                    "performance": perf,
                    "volatility": vol,
                    "sharpe_ratio": sharpe,
                    "current_price": current
                })
        except Exception as e:
            logger.warning(f"Failed to fetch {target}: {e}")
            continue
    
    if len(comparison_data) < 2:
        raise ValueError("Insufficient data for comparison")
    
    # Calculate correlations
    if len(returns_list) > 1:
        # Align lengths
        min_len = min(len(r) for r in returns_list)
        aligned_returns = [r[-min_len:] for r in returns_list]
        corr_matrix = np.corrcoef(aligned_returns)
        
        correlations = {}
        for i, t1 in enumerate(targets[:len(aligned_returns)]):
            for j, t2 in enumerate(targets[:len(aligned_returns)]):
                if i < j:
                    correlations[f"{t1}_vs_{t2}"] = float(corr_matrix[i, j])
    else:
        correlations = {}
    
    # Rankings
    best_performer = max(comparison_data, key=lambda x: x['performance'])
    best_risk_adjusted = max(comparison_data, key=lambda x: x['sharpe_ratio'])
    lowest_volatility = min(comparison_data, key=lambda x: x['volatility'])
    
    return {
        "comparison": comparison_data,
        "correlations": correlations,
        "rankings": {
            "best_performer": best_performer['target'],
            "best_risk_adjusted": best_risk_adjusted['target'],
            "lowest_volatility": lowest_volatility['target']
        },
        "period": period,
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# HANDLERS - Backtesting & Predictions
# =============================================================================

async def handle_backtest(params):
    """Advanced backtesting with rebalancing"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    start = params.get("start_date")
    end = params.get("end_date")
    capital = params.get("initial_capital", 10000)
    rebalance_freq = params.get("rebalance_frequency", "monthly")
    txn_cost = params.get("transaction_cost", 0.001)
    
    # Get portfolio
    if name:
        pf = get_single_portfolio(user_id, name)
        if not pf:
            raise ValueError(f"Portfolio '{name}' not found")
        summary = get_portfolio_summary(pf)
        assets = [h['symbol'] for h in summary['holdings']]
        weights = [h['weight'] for h in summary['holdings']]
    else:
        assets = params.get("assets", [])
        weights = params.get("weights", [])
    
    if not assets or not weights:
        raise ValueError("No assets/weights provided")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    try:
        # Fetch data
        data = yahoo.retrieve_data(tuple(assets), period='max')
        data = data.loc[start:end]
        
        if data.empty:
            raise ValueError(f"No data available for period {start} to {end}")
        
        # Calculate returns
        returns = data.pct_change().fillna(0)
        
        # Handle MultiIndex
        if isinstance(returns.columns, pd.MultiIndex):
            portfolio_returns = sum(
                returns[('Adj Close', asset)] * weight 
                for asset, weight in zip(assets, weights)
            )
        else:
            portfolio_returns = returns @ weights
        
        # Apply transaction costs
        if rebalance_freq != "none":
            # Simplified: apply cost on rebalancing days
            rebalance_days = get_rebalance_days(data.index, rebalance_freq)
            for day in rebalance_days:
                if day in portfolio_returns.index:
                    portfolio_returns.loc[day] -= txn_cost
        
        # Calculate portfolio value
        portfolio_values = capital * (1 + portfolio_returns).cumprod()
        
        # Metrics
        total_return = (portfolio_values.iloc[-1] - capital) / capital
        sharpe = float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
        
        # Max drawdown
        cumulative = portfolio_values / portfolio_values.cummax()
        drawdown = (cumulative - 1)
        max_dd = float(drawdown.min())
        
        # Sortino
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = float(portfolio_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        # Win rate
        win_rate = float((portfolio_returns > 0).sum() / len(portfolio_returns))
        
        return {
            "backtest_results": {
                "initial_capital": capital,
                "final_value": float(portfolio_values.iloc[-1]),
                "total_return": float(total_return),
                "total_return_pct": float(total_return * 100),
                "annualized_return": float(total_return / ((pd.to_datetime(end) - pd.to_datetime(start)).days / 365)),
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "volatility": float(portfolio_returns.std() * np.sqrt(252))
            },
            "configuration": {
                "start_date": start,
                "end_date": end,
                "assets": assets,
                "weights": weights.tolist(),
                "rebalance_frequency": rebalance_freq,
                "transaction_cost": txn_cost
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise ValueError(f"Backtest error: {e}")

async def handle_predict(params):
    """ML-based performance prediction"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    horizon = params.get("horizon", "3mo")
    model = params.get("model", "ensemble")
    confidence = params.get("confidence_level", 0.95)
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError(f"Portfolio '{name}' not found")
    
    summary = get_portfolio_summary(pf)
    assets = [h['symbol'] for h in summary['holdings']]
    weights = [h['weight'] for h in summary['holdings']]
    
    # Horizon mapping
    horizon_days = {
        "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252
    }
    days = horizon_days.get(horizon, 63)
    
    try:
        # Fetch historical data
        data = yahoo.retrieve_data(tuple(assets), period='2y')
        returns = data.pct_change().fillna(0)
        
        if isinstance(returns.columns, pd.MultiIndex):
            portfolio_returns = sum(
                returns[('Adj Close', asset)] * weight 
                for asset, weight in zip(assets, weights)
            )
        else:
            portfolio_returns = returns @ np.array(weights)
        
        # Simple prediction (TODO: Implement ARIMA/LSTM)
        avg_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Predicted return
        predicted_return = avg_return * days
        
        # Confidence interval
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        margin = z_score * volatility * np.sqrt(days)
        
        return {
            "prediction": {
                "horizon": horizon,
                "horizon_days": days,
                "expected_return": float(predicted_return),
                "expected_return_pct": float(predicted_return * 100),
                "confidence_level": confidence,
                "confidence_lower": float(predicted_return - margin),
                "confidence_upper": float(predicted_return + margin)
            },
            "model": model,
            "disclaimer": "Predictions based on historical data. Past performance does not guarantee future results.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError(f"Prediction error: {e}")

async def handle_simulate(params):
    """Monte Carlo simulation with stress testing"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    n_scenarios = params.get("scenarios", 1000)
    horizon = params.get("time_horizon", 252)
    shock_scenarios = params.get("shock_scenarios", [])
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError(f"Portfolio '{name}' not found")
    
    summary = get_portfolio_summary(pf)
    current_value = summary['current_value']
    
    # Historical data for parameters
    assets = [h['symbol'] for h in summary['holdings']]
    weights = [h['weight'] for h in summary['holdings']]
    
    try:
        data = yahoo.retrieve_data(tuple(assets), period='1y')
        returns = data.pct_change().dropna()
        
        if isinstance(returns.columns, pd.MultiIndex):
            portfolio_returns = sum(
                returns[('Adj Close', asset)] * weight 
                for asset, weight in zip(assets, weights)
            )
        else:
            portfolio_returns = returns @ np.array(weights)
        
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        
        # Monte Carlo simulation
        simulated_returns = np.random.normal(mu, sigma, (horizon, n_scenarios))
        paths = current_value * (1 + simulated_returns).cumprod(axis=0)
        
        final_values = paths[-1, :]
        
        # Statistics
        result = {
            "simulation": {
                "scenarios": n_scenarios,
                "horizon_days": horizon,
                "current_value": current_value,
                "median_value": float(np.median(final_values)),
                "mean_value": float(np.mean(final_values)),
                "std_dev": float(np.std(final_values))
            },
            "percentiles": {
                "5th": float(np.percentile(final_values, 5)),
                "25th": float(np.percentile(final_values, 25)),
                "50th": float(np.percentile(final_values, 50)),
                "75th": float(np.percentile(final_values, 75)),
                "95th": float(np.percentile(final_values, 95))
            },
            "extremes": {
                "worst_case": float(np.min(final_values)),
                "best_case": float(np.max(final_values)),
                "prob_loss": float((final_values < current_value).sum() / n_scenarios)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Stress scenarios
        if shock_scenarios:
            stress_results = []
            for shock in shock_scenarios:
                # Apply shock (simplified)
                shocked_value = current_value
                for asset, shock_pct in shock.items():
                    weight = next((w for a, w in zip(assets, weights) if a == asset), 0)
                    shocked_value += current_value * weight * shock_pct
                
                stress_results.append({
                    "scenario": shock,
                    "final_value": float(shocked_value),
                    "loss": float(shocked_value - current_value)
                })
            
            result["stress_tests"] = stress_results
        
        return result
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise ValueError(f"Simulation error: {e}")

# =============================================================================
# HANDLERS - Advanced Analytics
# =============================================================================

async def handle_correlations(params):
    """Calculate correlation matrix"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    tickers = params.get("tickers")
    period = params.get("period", "1y")
    
    if tickers:
        assets = tickers
    elif name:
        pf = get_single_portfolio(user_id, name)
        if not pf:
            raise ValueError(f"Portfolio '{name}' not found")
        summary = get_portfolio_summary(pf)
        assets = [h['symbol'] for h in summary['holdings']]
    else:
        raise ValueError("Either portfolio_name or tickers required")
    
    corr_matrix = await calculate_correlation_matrix(assets, period)
    
    return {
        "assets": assets,
        "correlation_matrix": corr_matrix,
        "average_correlation": float(np.mean(list(corr_matrix.values()))),
        "period": period
    }

async def handle_optimize(params):
    """Portfolio optimization suggestions"""
    user_id = params.get("user_id")
    name = params.get("portfolio_name")
    objective = params.get("objective", "sharpe")
    
    pf = get_single_portfolio(user_id, name)
    if not pf:
        raise ValueError(f"Portfolio '{name}' not found")
    
    summary = get_portfolio_summary(pf)
    
    return {
        "portfolio_name": name,
        "current_metrics": {
            "value": summary.get('current_value', 0),
            "return": summary.get('pnl_pct', 0)
        },
        "suggestions": [
            "Consider rebalancing to maintain target allocation",
            "Review underperforming assets",
            "Evaluate correlation to reduce risk"
        ],
        "note": "Full optimization coming in next version"
    }

# =============================================================================
# HANDLERS - Transactions & Watchlist
# =============================================================================

async def handle_transactions(params):
    """Get transaction history with analytics"""
    user_id = params.get("user_id")
    portfolio_name = params.get("portfolio_name")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    
    try:
        transactions = list(get_transactions(user_id))
        
        # Filter by portfolio
        if portfolio_name:
            transactions = [t for t in transactions if t.get('portfolio_name') == portfolio_name]
        
        # Filter by date
        if start_date or end_date:
            filtered = []
            for t in transactions:
                txn_date = t.get('date', t.get('timestamp', ''))
                if start_date and txn_date < start_date:
                    continue
                if end_date and txn_date > end_date:
                    continue
                filtered.append(t)
            transactions = filtered
        
        # Calculate aggregates
        total_buy = sum(t.get('total_amount', 0) for t in transactions if t.get('type') == 'buy')
        total_sell = sum(t.get('total_amount', 0) for t in transactions if t.get('type') == 'sell')
        
        # Group by asset
        by_asset = defaultdict(lambda: {'buy': 0, 'sell': 0, 'count': 0})
        for t in transactions:
            asset = t.get('ticker', t.get('asset', 'unknown'))
            txn_type = t.get('type', 'buy')
            amount = t.get('total_amount', 0)
            
            by_asset[asset][txn_type] += amount
            by_asset[asset]['count'] += 1
        
        return {
            "transactions": transactions,
            "count": len(transactions),
            "summary": {
                "total_buy": total_buy,
                "total_sell": total_sell,
                "net_flow": total_buy - total_sell,
                "by_asset": dict(by_asset)
            },
            "filters": {
                "portfolio": portfolio_name,
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise ValueError(f"Failed to get transactions: {e}")

async def handle_watchlist(params):
    """Get watchlist with live data"""
    user_id = params.get("user_id")
    include_metrics = params.get("include_metrics", True)
    
    try:
        watchlist = list(get_watchlist(user_id))
        
        enriched = []
        for item in watchlist:
            ticker = item.get('ticker', item.get('symbol'))
            
            if not ticker:
                continue
            
            entry = {"ticker": ticker, **item}
            
            if include_metrics:
                try:
                    # Get live data
                    data = yahoo.get_ticker_data(ticker, period='1mo')
                    info = yahoo.get_ticker_info(ticker)
                    
                    if data is not None and not data.empty:
                        current = float(data['Close'].iloc[-1])
                        prev = float(data['Close'].iloc[0])
                        change_pct = (current - prev) / prev * 100
                        
                        entry.update({
                            "price": current,
                            "change_1mo": change_pct,
                            "market_cap": info.get('marketCap'),
                            "pe_ratio": info.get('trailingPE'),
                            "name": info.get('longName', ticker)
                        })
                except Exception as e:
                    logger.warning(f"Failed to enrich {ticker}: {e}")
            
            enriched.append(entry)
        
        return {
            "watchlist": enriched,
            "count": len(enriched),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise ValueError(f"Failed to get watchlist: {e}")

async def handle_add_watchlist(params):
    """Add ticker to watchlist"""
    user_id = params.get("user_id")
    ticker = params.get("ticker", "").upper()
    
    if not ticker:
        raise ValueError("Ticker is required")
    
    try:
        # Validate ticker exists
        info = yahoo.get_ticker_info(ticker)
        if not info:
            raise ValueError(f"Invalid ticker: {ticker}")
        
        result = add_to_watchlist(user_id, ticker)
        
        return {
            "success": True,
            "ticker": ticker,
            "name": info.get('longName', ticker),
            "message": f"{ticker} added to watchlist"
        }
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        raise ValueError(f"Failed to add to watchlist: {e}")


