# utils.py
"""
Fonctions utilitaires générales
"""

import pandas as pd
import dask.dataframe as dd
from stock import Stock
from ta import moving_average, rsi, lrc
from dataprovider import yahoo


def format_pnl(value, pct):
    """Formate P&L avec couleur"""
    color_class = "positive" if value >= 0 else "negative"
    return f'<span class="{color_class}">${value:+,.2f} ({pct:+.2f}%)</span>'


def get_indicator(stock: Stock, indicator):
    """Récupère les indicateurs techniques"""
    data = stock.data
    indicators_map = {
        "MA 50": lambda: moving_average(data, 50),
        "MA 200": lambda: moving_average(data, 200),
        "RSI": lambda: rsi(data),
        "Volume": lambda: data["Volume"],
        "LRC": lambda: lrc(data)
    }
    return indicators_map.get(indicator, lambda: None)()


def dask_read_json(file):
    """Lit un fichier JSON avec Dask"""
    return dd.read_json(file, blocksize=None, orient="records", lines=False).compute()


def calculate_portfolio_current_value(portfolio_data):
    """Calcule la valeur actuelle d'un portfolio"""
    try:
        assets = portfolio_data['assets']
        data = yahoo.retrieve_data(tuple(assets), "1d")
        
        if isinstance(data.columns, pd.MultiIndex):
            latest_prices = data["Adj Close"].iloc[-1]
            print(latest_prices)
        else:
            latest_prices = data["Adj Close"].loc[-1]
        
        quantities = portfolio_data.get('quantities', [])
        if quantities:
            current_value=0.0
            for i, q in  enumerate(quantities):
               print(q,latest_prices[i],q * latest_prices[i] )
               current_value += q * latest_prices[i] 
            pnl = current_value - portfolio_data['amount']
            pnl_pct = (pnl / portfolio_data['amount']) * 100
            return current_value, pnl, pnl_pct
    except:
        pass
    
    return portfolio_data['amount'], 0, 0
