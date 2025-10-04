
import sys


from rich import console,table
import inquirer


from database import get_portfolios, save_portfolio,get_single_portfolio
from portfolios import describe ,factory
from portfolios import Portfolio
from printer.printer import print_dict
import viz 
from dataprovider import yahoo
from shell import Cli


name_arg = {"name": {"type": str, "help": "portfolio name"}}

portfolio_cli = Cli("portfolio")


def explore_portfolio_menu(name:str,portfolio:Portfolio):
      
      while True:

        cline=input('[portfolio]'+name+'$')
        pf=get_single_portfolio(name=name)
        args=cline.split(" ")
        fun=args[0]
        match fun:
          case "quit":
            sys.exit()
          case "viz":
            questions = [
            inquirer.Checkbox('graphs',
                    message="choose your graphs",
                    choices=['correlation', 'pie'],
                    ),
            ]
            answers = inquirer.prompt(questions)  
            
            graphs=answers["graphs"]   
            if "correlation" in graphs:
            
               corr=portfolio.correlation_matrix
               viz.portfolio.corr_heatmap(corr)

            if "pie" in graphs:
               viz.portfolio.plot_pie(portfolio.assets,portfolio.weights)
            
       
          case "summary":
             summary=describe.summary(portfolio,False)
             print_dict(dct=summary,colnames=["metric","value"],title="summary")

          case "composition":
              assets,weigths=describe.composition(portfolio,False)
              comps_dict={assets[i]:f"{weigths[i]:.2%}" for i in range(len(assets))}

              print_dict(dct=comps_dict,colnames=["asset","weight"],title="composition")
        
          case "describe":

             print("model :  {} \n".format(pf["model"]))
             print("method :  {} \n".format(pf["method"]))
          case _ :
                print("unknown command")
        
def build_portfolio_menu():
    stocks= input("enter assets:").split()
    portfolio=None
    while True:

        cline=input('[portfolio]build$')
        args=cline.split(" ")
        fun=args[0]
        match fun:

           case "quit":
              sys.exit()
        
           case "markowitz":

                npoints=3000
                data=yahoo.retrieve_data(tuple(stocks),period="8y")
                if len(args)==2:
                    npoints=args[1]

                viz.portfolio.plot_markowitz_curve(stocks,npoints,data)
                
           case "corr":
                period="1y"
                if len(args)==2:
                  period=args[1]
                data=yahoo.retrieve_data(tuple(stocks),period)
                portfolio=Portfolio(stocks,data=data)
                corr=portfolio.correlation_matrix
                viz.portfolio.corr_heatmap(corr)
            
           case "plot":
                  viz.portfolio.plot(stocks,args[1],"1mo")

           case "summary":
                data=yahoo.retrieve_data(tuple(stocks))
                portfolio=factory.create_portfolio_by_name(stocks,args[1],data)
                describe.summary(portfolio)

           case "composition":
                data=yahoo.retrieve_data(tuple(stocks))
                portfolio=factory.create_portfolio_by_name(stocks,args[1],data)
                describe.composition(portfolio)
           case "save":
                data=yahoo.retrieve_data(tuple(stocks))
                portfolio=factory.create_portfolio_by_name(stocks,args[1],data)
                save_portfolio(portfolio,"first",model=args[1],)
        
           case "pie":
                viz.portfolio.plot_pie(stocks,portfolio.weights)
        
@portfolio_cli.command()
def ls():
    pfs=list(get_portfolios())
    portfolio_names=[p["name"] for p in pfs]
    print(*portfolio_names,sep="\n")

@portfolio_cli.command(cmd_args=name_arg)
def composition(name:str):
    db_portfolio=get_single_portfolio(name)
    assets=db_portfolio["assets"]
        
    data=yahoo.retrieve_data(tuple(assets))
    portfolio=Portfolio(assets=assets,data=data)
    portfolio.set_weights(db_portfolio["weights"])

    assets,weights=describe.composition(portfolio,cmline=False)
    assets=portfolio.assets
    weights=portfolio.weights
  #print(list(zip(portfolio.assets,portfolio.weights)))
    ctable = table.Table(title="composition")
    ctable.add_column("asset", justify="right", style="cyan", no_wrap=True)
    ctable.add_column("weight", style="magenta")
    for i in range(len(assets)):
        ctable.add_row(assets[i],str(weights[i]))
    cconsole = console.Console()
    cconsole.print(ctable)

@portfolio_cli.command(cmd_args=name_arg)
def go(name:str):  
        
        db_portfolio=get_single_portfolio(name)
        assets=db_portfolio["assets"]
        
        data=yahoo.retrieve_data(tuple(assets))
        portfolio=Portfolio(assets=assets,data=data)
        
        portfolio.set_weights(db_portfolio["weights"])
 
        explore_portfolio_menu(name=name,portfolio=portfolio)

@portfolio_cli.command()
def build():
    """
    run the script to build a portfolio
    """
    build_portfolio_menu()


@portfolio_cli.command()
def quit():
    """
    quit the script
    """
    sys.exit()
if __name__=="__main__":
    portfolio_cli.run()