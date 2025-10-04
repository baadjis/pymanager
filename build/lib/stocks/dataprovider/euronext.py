import pandas as pd
from dataprovider.browser import create_browser
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

market_list={"amsterdam":["stocks","indices","bonds","etfs",
              "structured-products","funds",
              "index-options","stock-options","dividend-derivatives",
              "stock-futures","index-futures"]
             
             ,"brussels":["stocks","indices","bonds","etfs",
              "structured-products",
              "stock-options","dividend-derivatives",
              "stock-futures","index-futures"],

             "oslo":["stocks","indices","bonds","etfs",
              "bonds-indice","funds","derivatives"],

             "paris":["stocks","indices","bonds","etfs",
              "structured-products",
              "stock-options","dividend-derivatives",
              "stock-futures","index-futures","index-options"],

             "lisbon":["stock-futures","index-futures","dividend-derivatives"],
             "dublin":["stocks","indices","listed-etfs",
                       "bonds","listed-funds",
                       "govbonds"],
             "milan":["stocks","bonds","fixed-income","etfs",
              "structured-products","funds",
              "index-options","stock-options","dividend-derivatives",
              "stock-futures","index-futures"]
             }

def get_tables(page_name:str):

    url="https://live.euronext.com/en"
    if page_name!="":
       url=f"https://live.euronext.com/en/{page_name}"

    
    
    browser=create_browser()
    try:

        browser.get(url)
        #wait = WebDriverWait(browser, 5)
        #search = wait.until(EC.visibility_of_element_located((By.NAME, 'q')))
        time.sleep(3)

        accept_button = browser.find_elements(by=By.TAG_NAME, value="button")[2]
        #print(accept_button.text)
    
        browser.execute_script("arguments[0].click();", accept_button)
    
       
        tables=pd.read_html(browser.page_source)
        
          
        while True:
            try:
              
              next_button=browser.find_element(by=By.XPATH, value="//a[contains(@id,'next')]")
              

              button_class=next_button.get_attribute("class")
              print(button_class)

              if "disabled" not in button_class:

                    browser.execute_script("arguments[0].click();",next_button)
                    time.sleep(1)
                    tables2=pd.read_html(browser.page_source)
                    tables=tables + tables2
                    
              else:
                  break
            except:
                break
            
        return tables
        
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
     

def get_home():

    tables=get_tables(page_name="")
    out={"national":tables[0],"europe":tables[1],"currency":tables[2]}
    return out
    
def get_market(market:str):

    if market in market_list:
      tables=get_tables(page_name=f"markets/{market}")
      ntables=len(tables)
      out={"indexes":tables[0],"most_active":tables[3] if ntables>=4 else None,
           "most_active_stock_derivatives":tables[4] if ntables>=5 else None,
           "most_active_indices_derivatives":tables[5] if ntables>=6 else None}
      return out
    else:
        raise Exception("unknown market") 
       
def get_market_indexes(market:str):
    if market in market_list :
      if "indices" in market_list[market]: 
          tables=get_tables(page_name=f"markets/{market}/indices/list")
          return pd.concat(tables)
      
      raise Exception(f"indices are not listed by euronex for {market} market")
    else:
        raise Exception("unknown market")
def get_market_etfs(market:str):
    if market in market_list:
      if "etfs" in market_list[market] or "listed-etfs":
        tables=get_tables(page_name=f"markets/{market}/etfs/list")
        return pd.concat(tables)
      
      raise Exception(f"etfs are not listed by euronex for {market} market")
    else:
        raise Exception("unknown market")
def get_market_ipos(market:str):
    if market in market_list:
      tables=get_tables(page_name=f"markets/{market}/ipos")
      return pd.concat(tables)
    else:
        raise Exception("unknown market")
    
def get_market_fixed_income(market:str,bond_type:str=""):
    page_name=f"markets/{market}/fixed-income/list"
    if bond_type!="":
          page_name=f"markets/{market}/fixed-income/{bond_type}/list"

    if market in market_list:
      tables=get_tables(page_name=page_name)
      return pd.concat(tables)
    else:
        raise Exception("unknown market")
    
def get_market_bonds(market:str):
    
    if market=="milan":
        return get_market_fixed_income(market=market)
    page_name=f"markets/{market}/bonds/list"


    if market in market_list:
      if "bonds" in market_list[market]:
        tables=get_tables(page_name=page_name)
        return pd.concat(tables)
      raise Exception(f"bonds are not listed for {market} market in euronext")
    else:
        raise Exception("unknown market")
    
def get_market_govbonds(market:str):
    
    
    page_name=f"markets/{market}/govbonds/list"


    if market in market_list:
      if "govbonds" in market_list[market]:
        tables=get_tables(page_name=page_name)
        return pd.concat(tables)
      raise Exception(f"bonds are not listed for {market} market in euronext")
    else:
        raise Exception("unknown market")
    
def get_market_fixed_income_mot(market:str):
    return get_market_fixed_income(market=market,bond_type="mot")

def get_market_fixed_income_eurotlx(market:str):
    return get_market_fixed_income("eurotlx")

def get_market_fixed_income_euronext_access(market:str):

    return get_market_fixed_income(market=market,bond_type=f"euronext-access")
    
def get_market_structured_products(market:str):

    if market in market_list:
      tables=get_tables(page_name=f"markets/{market}/structured-products/list")
      return pd.concat(tables)
    else:
        raise Exception("unknown market")
    
def get_market_derivatives(market:str,derivative_type:str=""):

    if market in market_list:
       page_name=f"markets/{market}/derivatives/list"

       if derivative_type!="":
          page_name=f"markets/{market}/{derivative_type}/list"

           
    
          
       tables=get_tables(page_name=page_name)
       return pd.concat(tables)
    else:
        raise Exception("unknown market")
    
def get_market_index_futures(market:str):

    return get_market_derivatives(market=market,derivative_type="index-derivatives/futures")

def get_market_index_options(market:str):
    
    return get_market_derivatives(market=market,derivative_type="index-derivatives/options")

def get_market_stock_options(market:str):
    
    return get_market_derivatives(market=market,derivative_type="stock-options")

def get_market_stock_futures(market:str):
    
    return get_market_derivatives(market=market,derivative_type="stock-futures")

def get_market_dividend_derivatives(market:str):

    return get_market_derivatives(market=market,derivative_type="dividend-derivatives")

def get_market_derivatives_vendor_code(market:str):

    if market in market_list:
      
      tables=get_tables(page_name=f"markets/{market}/derivatives/quote-vendor-codes")
      
      return pd.concat(tables)
    
    else:

        raise Exception("unknown market")
    
def get_market_funds(market:str):

    if market in market_list:
      
      tables=get_tables(page_name=f"markets/{market}/funds/list")
      
      return pd.concat(tables)
    
    else:

        raise Exception("unknown market")

    
if __name__=='__main__':
    #indexes=get_home()
    #print(indexes["national"])
    #amsterdam_indexes=get_market("oslo")["indexes"]
    #print(amsterdam_indexes)
    paris_markets_indexes=get_market_bonds("dublin")

    print(paris_markets_indexes)