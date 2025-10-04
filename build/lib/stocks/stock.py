#!/usr/bin/env python3.11

from functools import cached_property
import sys

import pandas as pd

from dataprovider import yahoo

import yahoo_fin.stock_info as si
from yahoo_fin import options


class Stock:
    def __init__(self,ticker) -> None:
        self.ticker=ticker
        self.data=pd.DataFrame()
    def get_data(self,period="2y"):
    
        self.data=yahoo.get_ticker_data(self.ticker,period=period)
    
    @property
    def market_cap(self):
        return self.data["Open"]*self.data["Volume"]
    def get_financials(self):
        return yahoo.get_ticker_financials(self.ticker)
        
    def get_analysises(self):
        return si.get_analysts_info(self.ticker)
    @cached_property
    def infos(self):
        try:
            info=yahoo.get_ticker_info(self.ticker)
            return info
        except:
            return None
    def get_news(self):
        return yahoo.get_ticker_news(self.ticker)
          

    def get_sector(self):
       
        inf =self.infos
        if not(inf is None):
            return inf.loc["sector"].Value
        return "undefined"
    def get_industry(self):
        inf =self.infos
        if not(inf is None):
            return inf.loc["industry"].Value
        return "undefined"

    def get_balance_sheet(self):
        return yahoo.get_balance_sheet(self.ticker)

    def get_cash_flow(self):
        return yahoo.get_cash_flow(self.ticker)
    
    def get_dividends(self):
        return yahoo.get_dividends(self.ticker)

    def get_quote_table(self):
        return si.get_quote_table(self.ticker)
    
    def get_stats(self):
        return si.get_stats(self.ticker)
    def get_stats_valuation(self):
        return si.get_stats_valuation(self.ticker)

    def get_calls(self):
        return options.get_calls(self.ticker)
    def get_puts(self):
        return options.get_puts(self.ticker)
    def get_options_chain(self):
        return options.get_options_chain(self.ticker)

if __name__=="__main__":
    st=Stock("BNP.PA")
    print(st.get_balance_sheet())

    
