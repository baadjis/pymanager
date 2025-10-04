import time
import pandas as pd
import yfinance as yf
#from requests_html import HTMLSession
from dataprovider.browser import create_browser
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def search_with_class(st:str,class_name:str):
     
    url='http://www.google.com'
    browser=create_browser()
    try:

        browser.get(url)
        #wait = WebDriverWait(browser, 5)
        #search = wait.until(EC.visibility_of_element_located((By.NAME, 'q')))
        search=browser.find_element(by=By.NAME,value='q')
        print(search)
        accept_button= browser.find_elements(by=By.TAG_NAME,value="button")[3]
        print(accept_button)
        
        browser.execute_script("arguments[0].click();", accept_button)
        time.sleep(5)
        search.send_keys(st,Keys.RETURN) # hit return after you enter search text
        time.sleep(3)

        el=browser.find_element(by=By.CLASS_NAME, value=class_name)
        print(el.text)
        return el
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            
     

def search_isin(st:str):
   
    url='http://www.google.com'
    browser=create_browser()
    try:

        browser.get(url)
        #wait = WebDriverWait(browser, 5)
        #search = wait.until(EC.visibility_of_element_located((By.NAME, 'q')))
        search=browser.find_element(by=By.NAME,value='q')
        print(search)
        accept_button= browser.find_elements(by=By.TAG_NAME,value="button")[3]
        print(accept_button)
        
        browser.execute_script("arguments[0].click();", accept_button)
        time.sleep(5)
        search.send_keys(st,Keys.RETURN) # hit return after you enter search text
        time.sleep(3)

        el=browser.find_element(by=By.CLASS_NAME, value='hgKElc')
        print(el.text)
        return el.text.split("-")[1].split(":")[1]
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()

def search_ticker(st:str):
    
    url='http://www.google.com'
    browser=create_browser()
    try:

        browser.get(url)
        #wait = WebDriverWait(browser, 5)
        #search = wait.until(EC.visibility_of_element_located((By.NAME, 'q')))
        search=browser.find_element(by=By.NAME,value='q')
        print(search)
        accept_button= browser.find_elements(by=By.TAG_NAME,value="button")[3]
        print(accept_button)
        
        browser.execute_script("arguments[0].click();", accept_button)
        time.sleep(5)
        search.send_keys(st,Keys.RETURN) # hit return after you enter search text
        time.sleep(3)

        el=browser.find_element(by=By.CLASS_NAME, value='loJjTe')
        print(el.text)
        return el.text.split(":")[1]
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()

    
   
    
def get_ticker_info(ticker):
    """session=HTMLSession()
    url=f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"

    r=session.get(url=url)
    
    html=r.html
    
    html.render(sleep=2,timeout=20)
    #print(html.text)
    b=html.xpath('//button[contains(@class,"accept-all")]')
    #b=html.find("button")
    print(b)
    #p=html.xpath('//div[@class="asset-profile-container")]')
    #print(p)
    """
    st=yf.Ticker(ticker=ticker)
    infos =st.info
    print(infos)
    return infos
    
def get_ticker_info2(ticker:str):

    url=f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
    
    browser=create_browser()
    try:

        browser.get(url)
        button=browser.find_element(by="xpath",value="//button[@name='agree']")
        
       
        browser.execute_script("arguments[0].click();", button)
        
        time.sleep(2)
        browser.refresh()

        time.sleep(2)
        asset_profile_container=browser.find_element(by=By.CLASS_NAME,value="asset-profile-container")
        #print(asset_profile_container.text)
        ps=asset_profile_container.find_elements(by=By.TAG_NAME,value="p")
        desc=ps[0].text
        print(desc)
        sector_industry=ps[1].find_elements(by=By.TAG_NAME,value="span")
        sector=sector_industry[1].text
        industry=sector_industry[3].text
        table=pd.read_html(browser.page_source)
        print(f"sector :{sector} ,industry:{industry}")
        #print("table",table)
        return table[0]
        
    except Exception as e:
        print(e.__class__)
            
    finally:
            browser.close()
            browser.quit()
            browser=None

class TickerQuote():

    def __init__(self,ticker) -> None:
         self.ticker=ticker
         self.browser=create_browser()

    def get_profile(self):
        url=f"https://finance.yahoo.com/quote/{self.ticker}/profile?p={self.ticker}"
        
        
        try:

            self.browser.get(url)
            button=self.browser.find_element(by="xpath",value="//button[@name='agree']")
            
            
            self.browser.execute_script("arguments[0].click();", button)
            
            time.sleep(2)
            self.browser.refresh()

            time.sleep(2)
            asset_profile_container=self.browser.find_element(by=By.CLASS_NAME,value="asset-profile-container")
            section=self.browser.find_elements(by=By.TAG_NAME,value="section")[1]
            
            #print(asset_profile_container.text)
            ps=asset_profile_container.find_elements(by=By.TAG_NAME,value="p")
            address=ps[0].text
            
            sector_industry=ps[1].find_elements(by=By.TAG_NAME,value="span")
            sector=sector_industry[1].text
            industry=sector_industry[3].text
            full_time_employees=sector_industry[5].text
            tables=pd.read_html(self.browser.page_source)
            #print(f"sector :{sector} ,industry:{industry}")
            #print("table",table)
            
            subsections=section.find_elements(by=By.TAG_NAME,value="section")
        
            description=subsections[1].text
            gouvernance=subsections[2].text
            return {"sector": sector,"industry":industry,"officers":tables[0],
                    "fulltimeemployees":full_time_employees,
                    "address":address,
                    "gouvernance":gouvernance,
                    "description":description

                    }
            
        except Exception as e:
            print(e.__class__)
                
        finally:
                self.browser.close()
                self.browser.quit()
                

    def _get_tables(self,page_name:str):
        url=f"https://finance.yahoo.com/quote/{self.ticker}/{page_name}"
        
        
        try:

            self.browser.get(url)
            button=self.browser.find_element(by="xpath",value="//button[@name='agree']")
            
        
            self.browser.execute_script("arguments[0].click();", button)
            
            time.sleep(2)
            self.browser.refresh()

            time.sleep(2)
            
            tables=pd.read_html(self.browser.page_source)
            
            
            return tables
            
        except Exception as e:
            print(e.__class__)
                
        finally:
                self.browser.close()
                self.browser.quit()
                
         
    def get_financials(self):
         
         return self._get_tables(page_name="financials")
        
    def get_analysis(self):
         
         return self._get_tables(page_name="analysis")
        
    def get_holders(self):
         
        return self._get_tables(page_name="holders")
    
    def get_summary(self):
         
        return self._get_tables(page_name="summary")
    
    def get_historical_data(self):

        return self._get_tables(page_name="history")
    
    def get_stats(self):

        return self._get_tables(page_name="key-statistics")
    
    
if __name__=="__main__":

    """goog=TickerQuote("GOOG")
    goog_profile=goog.get_profile()
    print(goog_profile["description"])
    """
    #print(search_with_class("ticker of apple","").text)
    #print(search_ticker("ticker of apple"))
    print(search_isin("what is isin of societe general"))
