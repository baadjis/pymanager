import sys

import pandas as pd
from shell import Cli
from dataprovider import news,tickerquote
from ml import nlp
from stocks import stock
import viz
from stocks.stock import Stock
from printer import printer

ticker_arg = {"ticker": {"type": str, "help": "the ticker"}}

stock_cli = Cli("stock")

def print_dataframe_list(dfs:list[pd.DataFrame]|None):
     for tab in dfs:
            printer.print_dataframe(tab)
     

def stock_menu(st):
    
    fun=None
    while True:
        
        fun=input("[stock]"+st+'$')
        
        match fun:
             
          case "sector":
             
             st1=tickerquote.TickerQuote(st)
             print(st1.get_profile()["sector"])

          case "plot":
             
             viz.stock.plot_stock(st)

          case "dividends":
               
               st1=tickerquote.TickerQuote(st)
               print(st1.get_dividends())

          case "stats":
                
                st1=tickerquote.TickerQuote(st)
                stats=st1.get_stats()
                print_dataframe_list(stats)

          case "description":
                
                st1=tickerquote.TickerQuote(st)
                print(st1.get_profile()["description"])

          case "financials":
               
               st1=tickerquote.TickerQuote(st)

               financials=st1.get_financials()
               
               for tab in financials:
                     printer.print_dataframe(tab)

          case "analysis":
               
               st1=tickerquote.TickerQuote(st)

               analysis=st1.get_analysis()
               print_dataframe_list(analysis)

          case "summary":
                  
                  st1=tickerquote.TickerQuote(st)
                  summary=st1.get_summary()
                  print_dataframe_list(summary)

          case "holders":
                  
                  st1=tickerquote.TickerQuote(st)
                  holders=st1.get_holders()
                  print_dataframe_list(holders)
               


          case "news":
                
                sumar=news.get_news_summary(st)
                viz.stock.word_cloud(sumar)

          case "sentiments":
                
                sumar=news.get_news_summary(st)

                print(nlp.get_sentiments_stock(sumar).sum())

          case "quit":
                
                sys.exit()

          case _:
                  
                  print(f"unknow command: {fun}")
        
@stock_cli.command(cmd_args=ticker_arg)
def go(ticker:str):
    stock_menu(ticker)



@stock_cli.command()
def quit():
    sys.exit()



if __name__=="__main__":
    stock_cli.run()