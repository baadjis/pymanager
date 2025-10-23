from portfolio import Portfolio
from factory import back_period, create_benchmark
from rich import table, console

def print_dict(dct: dict, colnames=["key", "value"], title=""):
    """Print dictionary as a rich table"""
    ctable = table.Table(title=title)
    for colname in colnames:
        ctable.add_column(colname, justify="right", style="cyan", no_wrap=True)
    
    for key, value in dct.items():
        ctable.add_row(str(key), str(value))
    
    cconsole = console.Console()
    cconsole.print(ctable)

    
def composition(portfolio: Portfolio, cmline: bool = True):
    """Display portfolio composition"""
    assets = portfolio.assets
    weights = portfolio.weights
    
    if cmline:
        print("==" * 20)
        print("assets  |  weights")
        for i in range(len(assets)):
            print(f"{assets[i]:8s} | {weights[i]:.4f} ({weights[i]*100:.2f}%)")
        print("==" * 20)
    else:
        return assets, weights


def summary(port: Portfolio, benchmark=None, cmline=True):
    """
    Display comprehensive portfolio summary with all risk metrics
    
    Args:
        port: Portfolio object
        benchmark: Benchmark portfolio for comparison (optional)
        cmline: If True, print to console. If False, return dict
    """
    
    # Basic metrics
    data = {
        # Returns
        "Expected Return (Annual)": f"{port.expected_return:.4%}",
        "Expected Profit": f"${port.initial * port.expected_return:.2f}",
        
        # Risk Metrics
        "Volatility (Annual)": f"{port.stdev:.4%}",
        "Variance": f"{port.variance:.6f}",
        
        # Value at Risk
        "VaR (95%)": f"${port.VAR():.2f}",
        "Modified VaR (Cornish-Fisher)": f"${port.Cornish_Fisher_var():.2f}",
        "Historical VaR (95%)": f"${port.value_at_risk_historical():.2f}",
        "CVaR/Expected Shortfall (95%)": f"${port.conditional_var():.2f}",
        
        # Distribution Metrics
        "Skewness": f"{port.skewness:.4f}",
        "Kurtosis": f"{port.kurtosis:.4f}",
        
        # Risk-Adjusted Returns
        "Sharpe Ratio": f"{port.sharp_ratio:.4f}",
        "Sortino Ratio": f"{port.sortino_ratio():.4f}",
        "Calmar Ratio": f"{port.calmar_ratio():.4f}",
        "Omega Ratio": f"{port.omega_ratio():.4f}",
        "Sterling Ratio": f"{port.sterling_ratio():.4f}",
        "Burke Ratio": f"{port.burke_ratio():.4f}",
        "Martin Ratio": f"{port.martin_ratio():.4f}",
        "Gain-to-Pain Ratio": f"{port.gain_to_pain_ratio():.4f}",
        
        # Drawdown Metrics
        "Max Drawdown": f"{port.max_drawdown:.4%}",
        "Ulcer Index": f"{port.ulcer_index():.4f}",
        
        # Downside Risk
        "Downside Deviation": f"{port.downside_deviation:.4%}",
        "Tail Ratio (95%/5%)": f"{port.tail_ratio:.4f}",
        
        # Performance Distribution
        "Positive Periods": f"{port.positive_periods:.2%}",
        "Negative Periods": f"{port.negative_periods:.2%}",
    }
    
    # Add benchmark-related metrics if benchmark is provided
    if benchmark is not None:
        benchmark_data = {
            "Alpha (Annual)": f"{port.alpha(benchmark) * 252:.4%}",
            "Beta": f"{port.beta(benchmark):.4f}",
            "Treynor Ratio": f"{port.treynor_ratio(benchmark):.4f}",
            "Information Ratio": f"{port.information_ratio(benchmark):.4f}",
            "Tracking Error": f"{port.tracking_error(benchmark):.4%}",
        }
        data.update(benchmark_data)
    
    if cmline:
        print("\n" + "==" * 40)
        print("PORTFOLIO SUMMARY")
        print("==" * 40)
        
        # Print in categories
        print("\n📊 RETURNS")
        print("-" * 60)
        print(f"  Expected Return (Annual):     {data['Expected Return (Annual)']:>12}")
        print(f"  Expected Profit:              {data['Expected Profit']:>12}")
        
        print("\n⚠️  RISK METRICS")
        print("-" * 60)
        print(f"  Volatility (Annual):          {data['Volatility (Annual)']:>12}")
        print(f"  Variance:                     {data['Variance']:>12}")
        print(f"  Downside Deviation:           {data['Downside Deviation']:>12}")
        
        print("\n💰 VALUE AT RISK")
        print("-" * 60)
        print(f"  VaR (95%):                    {data['VaR (95%)']:>12}")
        print(f"  Modified VaR:                 {data['Modified VaR (Cornish-Fisher)']:>12}")
        print(f"  Historical VaR:               {data['Historical VaR (95%)']:>12}")
        print(f"  CVaR/Expected Shortfall:      {data['CVaR/Expected Shortfall (95%)']:>12}")
        
        print("\n📈 RISK-ADJUSTED RETURNS")
        print("-" * 60)
        print(f"  Sharpe Ratio:                 {data['Sharpe Ratio']:>12}")
        print(f"  Sortino Ratio:                {data['Sortino Ratio']:>12}")
        print(f"  Calmar Ratio:                 {data['Calmar Ratio']:>12}")
        print(f"  Omega Ratio:                  {data['Omega Ratio']:>12}")
        print(f"  Sterling Ratio:               {data['Sterling Ratio']:>12}")
        print(f"  Burke Ratio:                  {data['Burke Ratio']:>12}")
        print(f"  Martin Ratio:                 {data['Martin Ratio']:>12}")
        print(f"  Gain-to-Pain Ratio:           {data['Gain-to-Pain Ratio']:>12}")
        
        print("\n📉 DRAWDOWN METRICS")
        print("-" * 60)
        print(f"  Max Drawdown:                 {data['Max Drawdown']:>12}")
        print(f"  Ulcer Index:                  {data['Ulcer Index']:>12}")
        
        print("\n📊 DISTRIBUTION")
        print("-" * 60)
        print(f"  Skewness:                     {data['Skewness']:>12}")
        print(f"  Kurtosis:                     {data['Kurtosis']:>12}")
        print(f"  Tail Ratio:                   {data['Tail Ratio (95%/5%)']:>12}")
        print(f"  Positive Periods:             {data['Positive Periods']:>12}")
        print(f"  Negative Periods:             {data['Negative Periods']:>12}")
        
        if benchmark is not None:
            print("\n🎯 BENCHMARK COMPARISON")
            print("-" * 60)
            print(f"  Alpha (Annual):               {data['Alpha (Annual)']:>12}")
            print(f"  Beta:                         {data['Beta']:>12}")
            print(f"  Treynor Ratio:                {data['Treynor Ratio']:>12}")
            print(f"  Information Ratio:            {data['Information Ratio']:>12}")
            print(f"  Tracking Error:               {data['Tracking Error']:>12}")
        
        print("\n" + "==" * 40 + "\n")
    else:
        return data


def compare_portfolios(portfolios: list[Portfolio], names: list[str], benchmark=None):
    """
    Compare multiple portfolios side by side
    
    Args:
        portfolios: List of Portfolio objects
        names: List of portfolio names
        benchmark: Optional benchmark for comparison
    """
    print("\n" + "==" * 50)
    print("PORTFOLIO COMPARISON")
    print("==" * 50 + "\n")
    
    comparison_data = []
    
    for port, name in zip(portfolios, names):
        metrics = {
            "Portfolio": name,
            "Return": f"{port.expected_return:.2%}",
            "Volatility": f"{port.stdev:.2%}",
            "Sharpe": f"{port.sharp_ratio:.3f}",
            "Sortino": f"{port.sortino_ratio():.3f}",
            "Max DD": f"{port.max_drawdown:.2%}",
            "Calmar": f"{port.calmar_ratio():.3f}",
            "VaR": f"${port.VAR():.0f}",
        }
        
        if benchmark is not None:
            metrics["Alpha"] = f"{port.alpha(benchmark) * 252:.2%}"
            metrics["Beta"] = f"{port.beta(benchmark):.3f}"
            metrics["Treynor"] = f"{port.treynor_ratio(benchmark):.3f}"
        
        comparison_data.append(metrics)
    
    # Create comparison table
    ctable = table.Table(title="Portfolio Metrics Comparison")
    
    # Add columns
    for key in comparison_data[0].keys():
        ctable.add_column(key, justify="right" if key != "Portfolio" else "left")
    
    # Add rows
    for metrics in comparison_data:
        ctable.add_row(*[str(v) for v in metrics.values()])
    
    cconsole = console.Console()
    cconsole.print(ctable)


def risk_metrics_explained():
    """
    Print explanations of all risk metrics
    """
    explanations = {
        "Sharpe Ratio": "Excess return per unit of total risk. Higher is better. >1 is good, >2 is very good.",
        "Sortino Ratio": "Like Sharpe, but only penalizes downside volatility. Better for asymmetric returns.",
        "Calmar Ratio": "Annual return divided by max drawdown. Measures return vs worst loss.",
        "Treynor Ratio": "Excess return per unit of systematic risk (beta). Good for diversified portfolios.",
        "Information Ratio": "Alpha divided by tracking error. Measures manager skill vs benchmark.",
        "Omega Ratio": "Probability-weighted gains vs losses. >1 means more upside than downside.",
        "Sterling Ratio": "Return divided by average drawdown. Like Calmar but uses average, not max.",
        "Burke Ratio": "Return divided by square root of sum of squared drawdowns. Penalizes multiple drawdowns.",
        "Martin Ratio": "Return divided by Ulcer Index. Measures pain-adjusted returns.",
        "Max Drawdown": "Largest peak-to-trough decline. Shows worst historical loss.",
        "Ulcer Index": "Measures depth and duration of drawdowns. Lower is better.",
        "VaR (95%)": "Maximum expected loss in 95% of cases. 5% chance of losing more.",
        "CVaR": "Average loss in the worst 5% of cases. More conservative than VaR.",
        "Tail Ratio": "Ratio of 95th to 5th percentile returns. >1 means positive asymmetry.",
        "Skewness": "Asymmetry of return distribution. Positive = more upside potential.",
        "Kurtosis": "Tail thickness. High values = more extreme events than normal distribution.",
    }
    
    print("\n" + "==" * 50)
    print("RISK METRICS GUIDE")
    print("==" * 50 + "\n")
    
    for metric, explanation in explanations.items():
        print(f"📌 {metric}")
        print(f"   {explanation}\n")


if __name__ == "__main__":
    # Example usage
    print("Testing portfolio description module...")
    
    # Show risk metrics guide
    risk_metrics_explained()
    
    # Test dictionary printing
    dct = {"language": "python", "skill": "advanced", "domain": "finance"}
    print_dict(dct, title="Test Dictionary")
