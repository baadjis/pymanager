from functools import lru_cache
import time
from typing import List, Optional
from pandas_datareader import data as pdr
import yahoo_fin.stock_info as si
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import linregress
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .browser import create_browser
except:
    pass  # browser functions are optional


@lru_cache()
def retrieve_data(tickers: tuple[str], period='5y'):
    """Retrieve data for multiple tickers"""
    logger.info(f"Retrieving data for tickers: {tickers}")
    history = yf.download(tickers=list(tickers), period=period, auto_adjust=False, progress=False)
    return history


@lru_cache()
def get_ticker_data(ticker_name: str, period='5y'):
    """Get data for a single ticker"""
    if period == "1d" or period == "5d":
        return yf.download(ticker_name, period=period, interval="1h", auto_adjust=False, progress=False)
    data = yf.download(ticker_name, period=period, auto_adjust=False, progress=False)
    return data


def get_ticker_info(ticker: str) -> dict:
    """Get ticker information from yfinance"""
    try:
        st = yf.Ticker(ticker=ticker)
        infos = st.info
        return infos
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {e}")
        return {}


def get_ticker_news(ticker: str):
    """Get news for a ticker"""
    try:
        st = yf.Ticker(ticker=ticker)
        infos = st.news
        return infos
    except Exception as e:
        logger.error(f"Error getting news for {ticker}: {e}")
        return []


def get_ticker_dividends(ticker: str):
    """Get dividend history"""
    try:
        st = yf.Ticker(ticker=ticker)
        div = st.dividends
        return div
    except Exception as e:
        logger.error(f"Error getting dividends for {ticker}: {e}")
        return pd.Dataframe()


def get_balance_sheet(ticker: str):
    """Get balance sheet"""
    try:
        st = yf.Ticker(ticker=ticker)
        div = st.balance_sheet
        return div
    except Exception as e:
        logger.error(f"Error getting balance sheet for {ticker}: {e}")
        return pd.DataFrame()


def get_cash_flow(ticker: str):
    """Get cash flow statement"""
    try:
        st = yf.Ticker(ticker=ticker)
        div = st.cashflow
        return div
    except Exception as e:
        logger.error(f"Error getting cash flow for {ticker}: {e}")
        return pd.DataFrame()


def get_ticker_financials(ticker: str):
    """Get financial statements"""
    try:
        st = yf.Ticker(ticker=ticker)
        fin = st.financials
        return fin
    except Exception as e:
        logger.error(f"Error getting financials for {ticker}: {e}")
        return pd.DataFrame()


def get_log_returns(price_history: pd.DataFrame):
    """Calculate log returns from price history"""
    prices = price_history['Adj Close']
    v = np.log(prices / prices.shift(1)).dropna()
    return v


def calculate_beta_from_returns(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate beta using linear regression
    
    Args:
        asset_returns: Returns of the asset
        benchmark_returns: Returns of the benchmark
        
    Returns:
        float: Beta value
    """
    try:
        # Align the series
        common_index = asset_returns.index.intersection(benchmark_returns.index)
        
        if len(common_index) < 30:  # Need at least 30 data points
            logger.warning("Not enough common data points for beta calculation")
            return 1.0
        
        asset_aligned = asset_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        
        # Remove any NaN values
        mask = ~(asset_aligned.isna() | benchmark_aligned.isna())
        asset_clean = asset_aligned[mask]
        benchmark_clean = benchmark_aligned[mask]
        
        if len(asset_clean) < 30:
            logger.warning("Not enough clean data points for beta calculation")
            return 1.0
        
        # Calculate beta using linear regression
        slope, intercept, r_value, p_value, std_err = linregress(benchmark_clean, asset_clean)
        
        logger.info(f"Beta calculated: {slope:.4f}, R²: {r_value**2:.4f}")
        
        return slope
        
    except Exception as e:
        logger.error(f"Error in beta calculation: {e}")
        return 1.0


def get_asset_beta_yfinance(asset: str, benchmark: str = "^GSPC", period: str = "2y") -> float:
    """
    Calculate asset beta using yfinance data
    
    Args:
        asset: Asset ticker symbol
        benchmark: Benchmark ticker (default: S&P 500)
        period: Historical period for calculation (default: 2y)
        
    Returns:
        float: Beta value
    """
    try:
        logger.info(f"Calculating beta for {asset} vs {benchmark}")
        
        # Method 1: Try to get beta directly from ticker info
        ticker = yf.Ticker(asset)
        info = ticker.info
        
        # Check various beta fields that might exist
        beta_fields = ['beta', 'beta3Year', 'betaThreeYear']
        for field in beta_fields:
            if field in info and info[field] is not None:
                try:
                    beta = float(info[field])
                    if not np.isnan(beta) and beta != 0:
                        logger.info(f"✓ Beta for {asset} from yfinance.info['{field}']: {beta:.4f}")
                        return beta
                except (ValueError, TypeError):
                    continue
        
        # Method 2: Calculate beta from historical data
        logger.info(f"Calculating beta manually for {asset}...")
        
        # Download data for both asset and benchmark
        data = yf.download([asset, benchmark], period=period, progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 30:
            logger.warning(f"Insufficient data for {asset}")
            return 1.0
        
        # Extract adjusted close prices
        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers downloaded
            asset_prices = data['Close'][asset] if 'Close' in data.columns.levels[0] else data['Adj Close'][asset]
            benchmark_prices = data['Close'][benchmark] if 'Close' in data.columns.levels[0] else data['Adj Close'][benchmark]
        else:
            # Single ticker case - need to download benchmark separately
            asset_prices = data['Close'] if 'Close' in data.columns else data['Adj Close']
            benchmark_data = yf.download(benchmark, period=period, progress=False, auto_adjust=True)
            benchmark_prices = benchmark_data['Close'] if 'Close' in benchmark_data.columns else benchmark_data['Adj Close']
        
        # Calculate daily returns
        asset_returns = asset_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Calculate beta
        beta = calculate_beta_from_returns(asset_returns, benchmark_returns)
        
        logger.info(f"✓ Calculated beta for {asset}: {beta:.4f}")
        return beta
        
    except Exception as e:
        logger.error(f"✗ Error calculating beta for {asset}: {e}")
        return 1.0


@lru_cache(maxsize=128)
def get_asset_beta(asset: str, benchmark: str = "^GSPC") -> float:
    """
    Get asset beta with caching
    
    Args:
        asset: Asset ticker symbol
        benchmark: Benchmark ticker (default: S&P 500)
        
    Returns:
        float: Beta value (default 1.0 if calculation fails)
    """
    return get_asset_beta_yfinance(asset, benchmark)


def get_assets_beta(assets: list[str], benchmark: str = "^GSPC") -> list[float]:
    """
    Get betas for multiple assets
    
    Args:
        assets: List of asset ticker symbols
        benchmark: Benchmark ticker (default: S&P 500)
        
    Returns:
        list: List of beta values
    """
    logger.info(f"Getting betas for {len(assets)} assets")
    betas = []
    
    for asset in assets:
        try:
            beta = get_asset_beta(asset, benchmark)
            betas.append(beta)
            logger.info(f"  {asset}: β = {beta:.4f}")
        except Exception as e:
            logger.error(f"  {asset}: Error - {e}, using default β = 1.0")
            betas.append(1.0)
    
    return betas


@lru_cache()
def get_sector(ticker: str) -> tuple[str, str]:
    """
    Get financial sector and industry of a ticker
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        tuple: (sector, industry)
    """
    logger.info(f"Getting sector for {ticker}")
    try:
        company_infos = get_ticker_info(ticker=ticker)
        company_sector = company_infos.get("sector", "Unknown")
        company_industry = company_infos.get("industry", "Unknown")
        return company_sector, company_industry
    except Exception as e:
        logger.error(f"Error getting sector for {ticker}: {e}")
        return "Unknown", "Unknown"


def get_sectors_weights(assets: list, weights: list) -> dict:
    """
    Get sector weights for a portfolio
    
    Args:
        assets: Portfolio assets
        weights: Asset weights
        
    Returns:
        dict: Dictionary of sectors and their weights
    """
    logger.info(f"Calculating sector weights for {len(assets)} assets")
    assets_sectors = {a: get_sector(a)[0] for a in assets}
    sectors_weights = {}
    
    for i, v in enumerate(assets):
        sect = assets_sectors[v]
        sectors_weights[sect] = sectors_weights.get(sect, 0) + weights[i]
    
    return sectors_weights


def retrieve_data_sequence(tickers: List[str], period='5y'):
    """Legacy function - use retrieve_data instead"""
    dataframes = pd.DataFrame()

    for ticker_name in tickers:
        history = pdr.get_data_yahoo(ticker_name, period=period)
        
        if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
            history = history.iloc[1:]
        
        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        dataframes = pd.concat([dataframes, history], axis=1)
    
    return dataframes


# Browser-based functions (optional, require selenium)
def get_index_components(ticker: str):
    """Get index components using browser scraping"""
    url = f"https://finance.yahoo.com/quote/{ticker}/components?p={ticker}"
    
    browser = create_browser()
    try:
        browser.get(url)
        button = browser.find_element(by="xpath", value="//button[@value='agree']")
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(2)
        browser.refresh()
        time.sleep(2)
        
        table = pd.read_html(browser.page_source)
        return table[0]
        
    except Exception as e:
        logger.error(f"Error getting index components: {e}")
        return pd.DataFrame()
            
    finally:
        if browser:
            browser.close()
            browser.quit()


def search_for(term: str):
    """Search for a ticker using browser"""
    url = f"https://finance.yahoo.com/"
    
    browser = create_browser()
    try:
        browser.get(url)
        button = browser.find_element(by="xpath", value="//button[@value='agree']")
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(4)
        browser.refresh()
        time.sleep(4)
        input_elem = browser.find_element(by="xpath", value="//input[@id='yfin-usr-qry']")
        input_elem.send_keys(term)
        
        browser.find_element(by="id", value='header-desktop-search-button').click()
        browser.refresh()
        time.sleep(4)

        ticker = browser.find_element(by="xpath", value="//h1[@class='D(ib) Fz(18px)']").text
        return ticker
        
    except Exception as e:
        logger.error(f"Error searching for term: {e}")
        return None
            
    finally:
        if browser:
            browser.close()
            browser.quit()


def get_world_indices():
    """Get world indices data"""
    url = f"https://finance.yahoo.com/world-indices"
    
    browser = create_browser()
    try:
        browser.get(url)
        button = browser.find_element(by="xpath", value="//button[@value='agree']")
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(4)
        browser.refresh()
        time.sleep(4)
        
        table = pd.read_html(browser.page_source)
        return table[0]
        
    except Exception as e:
        logger.error(f"Error getting world indices: {e}")
        return pd.DataFrame()
            
    finally:
        if browser:
            browser.close()
            browser.quit()


if __name__ == '__main__':
    # Test beta calculation
    print("\n=== Testing Beta Calculation ===\n")
    
    test_assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    betas = get_assets_beta(test_assets)
    
    print("\nResults:")
    print("-" * 40)
    for asset, beta in zip(test_assets, betas):
        print(f"{asset:6s}: β = {beta:6.4f}")
    print("-" * 40)
    
    # Test with a specific asset
    print("\n=== Detailed Test for AAPL ===\n")
    apple_beta = get_asset_beta("AAPL")
    print(f"AAPL Beta: {apple_beta:.4f}")
