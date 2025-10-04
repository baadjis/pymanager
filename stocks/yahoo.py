
from functools import lru_cache
import time
from typing import List
from pandas_datareader import data as pdr
import yahoo_fin.stock_info as si
import pandas as pd
import yfinance as yf
import numpy as np
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



@lru_cache()
def retrieve_data(tickers: tuple[str],period='8y'):
    history = pdr.get_data_yahoo(tickers=list(tickers),period=period)
    return history
@lru_cache()
def get_ticker_data(ticker_name:str,period='8y'):
  print(ticker_name)
  if period=="1d" or period=="5d":
     return pdr.get_data_yahoo(ticker_name,period=period,interval="1h")
  data= pdr.get_data_yahoo(ticker_name,period=period)

  return data

def get_log_returns(price_history: pd.DataFrame):
  prices = price_history['Adj Close']
  v=np.log(prices/prices.shift(1)).dropna()
  return v
def retrieve_data_sequence(tickers: List[str],period='9y'):
  dataframes = pd.DataFrame()

  for ticker_name in tickers:
    
    history = pdr.get_data_yahoo(ticker_name,period=period)
    
    if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
      history = history.iloc[1:]
  
    assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
    dataframes.concat([history], axis=1)
    
    return  dataframes


def get_asset_beta(asset:str):
    """get asset beta

    Args:
        asset (str): the asset

    Returns:
        _type_: asset beta
    """
    quote=si.get_quote_table(asset)
    return quote['Beta (5Y Monthly)']

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
            company_industry=company_infos.loc["industry"].Value
            return company_sector,company_industry

def get_assets_beta(assets:list[str])->list:
    """get betas given list of assets

    Args:
        assets (list[str]): list of assets

    Returns:
        list: betas of assets
    """

    return [get_asset_beta(asset) for asset in assets]

def get_index_components(ticker:str):
    
    url=f"https://finance.yahoo.com/quote/{ticker}/components?p={ticker}"
    
    browser=create_browser()
    try:

        browser.get(url)
        button=browser.find_element(by="xpath",value="//button[@value='agree']")
    
    
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(2)
        browser.refresh()
        time.sleep(2)
        
        table=pd.read_html(browser.page_source)

        print("table",table[0])
        return table[0]
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            browser=None
   
def search_for(term):
    url=f"https://finance.yahoo.com/"
    
    browser=create_browser()
    try:

        browser.get(url)
        button=browser.find_element(by="xpath",value="//button[@value='agree']")
    
    
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(4)
        browser.refresh()
        time.sleep(4)
        input = browser.find_element(by="xpath",value="//input[@id='yfin-usr-qry']")
        print(input)
        input.send_keys(term)
        
        # Click on Search icon and wait for 2 seconds.
        browser.find_element(by="id",value='header-desktop-search-button').click()
        browser.refresh()
        time.sleep(4)
        print("here")

        ticker=browser.find_element(by="xpath",value="//h1[@class='D(ib) Fz(18px)']").text
        
        
        
        #print("table",table[0])
        return ticker
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            browser=None

    
def get_world_indices():
    url=f"https://finance.yahoo.com/world-indices"
    
    browser=create_browser()
    try:

        browser.get(url)
        button=browser.find_element(by="xpath",value="//button[@value='agree']")
    
    
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(4)
        browser.refresh()
        time.sleep(4)
        
        table=pd.read_html(browser.page_source)
        
        #print("table",table[0])
        return table[0]
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            browser=None
if __name__=='__main__':
    """major_indices=get_world_indices()
    final_data=major_indices[['Symbol','Name']]
    print(final_data)
    final_data.to_json("major_indices.json",orient="records")"""
    cac=search_for("cac 40")
    print(cac)
