import pandas as pd
from rich import table,console

def print_dict(dct:dict,colnames=["key","value"],title=""):
    ctable = table.Table(title=title)
    for i in range(len(colnames)):
      ctable.add_column(colnames[i], justify="right", style="cyan", no_wrap=True)
      #ctable.add_column("weight", style="magenta")
    keys=list(dct.keys())
    values=list(dct.values())
    for i in range(len(keys)):
        ctable.add_row(keys[i],str(values[i]))
    cconsole = console.Console()
    cconsole.print(ctable)
    
def print_dataframe(df:pd.DataFrame,title:str=""):
    
    colnames=df.columns.to_list()
    
    nrows=df.shape[0]
    ctable = table.Table(title=title)
    for col in colnames:

      ctable.add_column(str(col), justify="right", style="cyan", no_wrap=True)

    for i in range(nrows):

      rows=df.iloc[i].to_list()
      
      rows_string=tuple([str(r) for r in rows])
      
      ctable.add_row(*rows_string)

    cconsole = console.Console()
    cconsole.print(ctable)
    
    
if __name__=="__main__":
  
  dct={"language":"python","skill":"advanced"}
  print_dict(dct,title="skills")

  data={"name":["alice","bob","chalie"],
        "age":[25,26,27]
        }
  df =pd.DataFrame(data)
  print_dataframe(df)
 