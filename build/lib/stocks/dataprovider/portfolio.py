from functools import lru_cache
import pandas as pd

from pandas_datareader import data as pdr
import yahoo_fin.stock_info as si
import yfinance as yf
#import requests
import time

from selenium import webdriver 
yf.pdr_override()

def create_browser():
    """"
    create and configurate a browser
    """
    browser_options = webdriver.FirefoxOptions()
    browser_options.add_argument("--incognito")
    browser_options.add_argument("--headless")
    browser_options.add_argument("start-maximized")
    browser_options.add_argument("disable-infobars")
    browser_options.add_argument("--disable-extensions")
    browser_options.add_argument('--no-sandbox')
    browser_options.add_argument('--disable-application-cache')
    browser_options.add_argument('--disable-gpu')
    browser_options.add_argument("--disable-dev-shm-usage")
            #browser_options.add_experimental_option('useAutomationExtension', False)
    browser=webdriver.Firefox(options=browser_options)
    return browser



#data=retrieve_data_sequence(stocks)

#data=retrieve_data(stocks,back_period)
@lru_cache()
def retrieve_data(tickers: tuple[str],period='8y'):
    """retrieve financial(HLOCV) data for givens tickers

    Args:
        tickers (tuple[str]): the tickers to retrieve
        period (str, optional): the period. Defaults to '8y'.

    Returns:
        pandas dataframe: the data frame containing tickers data
    """
    history = pdr.get_data_yahoo(tickers=list(tickers),period=period)
    return history

@lru_cache()
def get_ticker_data(ticker_name:str,period='8y'):

  data= pdr.get_data_yahoo(ticker_name,period=period)
  return data

@lru_cache()
def get_sector(ticker:str)->str:
    """get financial sector of a ticker
    Args:
        ticker (str): the tciker to get sector

    Returns:
        str: the sector of the ticker
    """
    company_infos =si.get_company_info(ticker)
    company_sector= company_infos.loc["sector"].Value
    return company_sector

def get_sectors_weights(assets,weights)->dict:
    """get sectors weights given assets and weights on a portfolio

    Args:
        assets (_type_): portfolio assets
        weights (_type_): assets weight

    Returns:
        dict: dict of sectors and weights
    """
    assets_sectors={a:get_sector(a)for a in assets}  
    sectors_weights={}
    
    for i ,v in enumerate(assets):
        sect=assets_sectors[v]
        sectors_weights[sect]=sectors_weights.get(sect,0)+weights[i]
    return sectors_weights

def get_asset_beta(asset:str):
    """get asset beta

    Args:
        asset (str): the asset

    Returns:
        _type_: asset beta
    """
    quote=si.get_quote_table(asset)
    return quote['Beta (5Y Monthly)']

def get_assets_beta(assets:list[str])->list:
    """get betas given list of assets

    Args:
        assets (list[str]): list of assets

    Returns:
        list: betas of assets
    """

    return [get_asset_beta(asset) for asset in assets]

def get_index_components(ticker:str):
    
    url=f"https://finance.yahoo.com/quote/%5E{ticker}/components?p=%5E{ticker}"
    
    browser=create_browser()
    try:

        browser.get(url)
        button=browser.find_element(by="xpath",value="//button[@value='agree']")
    
    
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(2)
        browser.refresh()
        time.sleep(2)
        
        table=pd.read_html(browser.page_source)

    
        return table[0].dropna()
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            browser=None
   

    
   

