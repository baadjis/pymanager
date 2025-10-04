import sys
import yahoo_fin.stock_info as si
from cli import shell
from portfolios.cli import portfolio_cli
from stocks.cli import stock_cli

import portfolios.ui as ui
#import warnings
pro_arg = {"pro": {"type": str, "help": "the programm to run"}}

main_cli=shell.Cli("main")

@main_cli.command(cmd_args=pro_arg)
def go(pro:str):
  assert pro in ["portfolio","stock","ui"]
  if pro=="portfolio":
       
       portfolio_cli.run()
  if pro=="stock":
      stock_cli.run()
  if pro=="ui":
    ui.run()

@main_cli.command()
def quit():
  sys.exit()

if __name__=='__main__':
   main_cli.run()
    
    
      
      
      
        
      
      

       
       
