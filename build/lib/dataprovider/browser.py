from selenium import webdriver 
from fake_useragent import UserAgent

def create_browser():
    """"
    create and configurate a browser
    """
    user_agent = UserAgent()
    firefox_driver_bin="/snap/firefox/current/usr/lib/firefox/geckodriver"
    serv=webdriver.firefox.service.Service(executable_path=firefox_driver_bin)
    browser_options = webdriver.firefox.options.Options()
    browser_options.binary=webdriver.firefox.firefox_binary.FirefoxBinary("/usr/bin/firefox")
   
  
    browser_options.add_argument(f'user-agent={user_agent.random}')
    browser_options.add_argument("--incognito")
    browser_options.add_argument("--headless")
    browser_options.add_argument("start-maximized")
    browser_options.add_argument("disable-infobars")
    browser_options.add_argument("--disable-extensions")
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument('--disable-application-cache')
    browser_options.add_argument("--disable-gpu")
    browser_options.add_argument("--disable-dev-shm-usage")
    browser_options.add_argument("--width=800")
    browser_options.add_argument("--height=800")
    

            #browser_options.add_experimental_option('useAutomationExtension', False)
    browser=webdriver.Firefox(options=browser_options,service=serv)
    return browser
