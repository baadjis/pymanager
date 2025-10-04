from portfolio import Portfolio
from factory import back_period, create_benchmark
from rich import table,console

def print_dict(dct:dict,colnames=["key","value"],title="",):
    ctable = table.Table(title=title)
    for i in range(len(colnames)):
      ctable.add_column(colnames[i], justify="right", style="cyan", no_wrap=True)
      #ctable.add_column("weight", style="magenta")
    keys=dct.keys()
    values=dct.values()
    for i in range(len()):
        ctable.add_row(keys[i],str(values[i]))
    cconsole = console.Console()
    cconsole.print(ctable)
    



def composition(portfolio:Portfolio,cmline:bool=True):
  assets=portfolio.assets

  weights=portfolio.weights

  if cmline:
    print("=="*20)
    print("assets  |  weights")
    for i in range(len(assets)):
      print(f"{assets[i]} | {weights[i]}")
  else:
    return assets, weights



  #print(sum(portfolio.weights))

def summary(port:Portfolio,cmline=True):
  benchmark=create_benchmark("SPY",period=back_period)
  if cmline:
    print("=="*20)
    print(f"sharp ratio:{port.sharp_ratio}")
    print(f"treynor ratio:{port.treynor_ratio(benchmark)}")
    print(f"variance:{port.variance}")
    print(f"var:{port.VAR()}")
    print(f"mvar:{port.Cornish_Fisher_var()}")
    print(f"skew:{port.skewness}")
    print(f"kurtosis:{port.kurtosis}")
    print(f"expected return:{port.expected_return}")
    print(f"expected profit:{port.initial*port.expected_return}")
    print(f"alpha:{port.alpha(benchmark)}")
    print(f"beta:{port.beta(benchmark)}")
  else:
    data={"sharp ratio":port.sharp_ratio,
    "treynor ratio":port.treynor_ratio(benchmark),
    "variance":port.variance,
    "var":port.VAR(),
    "modified var":port.Cornish_Fisher_var(),
    "skew":port.skewness,
    "kurtosis":port.kurtosis,
    "expected return":port.expected_return,
    "expected profit":port.initial*port.expected_return,
    "alpha":port.alpha(benchmark),
    "beta":port.beta(benchmark)
    }
    return data

if __name__=="__main__":
  dct={"language":"python","skill":"advanced"}
  print_dict(dct)
