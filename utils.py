# utils.py
"""
Fonctions utilitaires V2
✅ Support structure holdings[]
✅ Calcul PnL unifié pour dashboard ET portfolio_manager
✅ get_portfolio_summary() - fonction centrale
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
    """
    Calcule la valeur actuelle d'un portfolio et son P&L
    ✅ Support NOUVELLE structure holdings[]
    ✅ Support ANCIENNE structure (legacy) pour compatibilité
    
    Args:
        portfolio_data: Dict avec soit holdings[] (V2) soit assets[] (legacy)
    
    Returns:
        tuple: (current_value, pnl, pnl_pct, holdings_details)
    """
    try:
        initial_amount = portfolio_data.get('initial_amount', 0)
        
        # ✅ NOUVELLE STRUCTURE - Holdings[]
        if 'holdings' in portfolio_data:
            return _calculate_from_holdings(portfolio_data, initial_amount)
        
        # ✅ ANCIENNE STRUCTURE - Legacy (assets[], weights[], quantities[])
        elif 'assets' in portfolio_data:
            return _calculate_from_legacy(portfolio_data, initial_amount)
        
        else:
            return initial_amount, 0.0, 0.0, []
        
    except Exception as e:
        print(f"❌ Erreur calcul portfolio: {e}")
        initial_amount = portfolio_data.get('initial_amount', 0)
        return initial_amount, 0.0, 0.0, []


def _calculate_from_holdings(portfolio_data, initial_amount):
    """
    Calcul depuis nouvelle structure holdings[]
    """
    holdings = portfolio_data.get('holdings', [])
    
    if not holdings:
        return initial_amount, 0.0, 0.0, []
    
    # Extraire les symbols
    symbols = [h['symbol'] for h in holdings]
    
    # Récupérer prix actuels
    data = yahoo.retrieve_data(tuple(symbols), "1d")
    
    if data is None or data.empty:
        return initial_amount, 0.0, 0.0, []
    
    # Gérer MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        latest_prices = data["Adj Close"].iloc[-1]
    else:
        if len(symbols) == 1:
            latest_prices = pd.Series([data["Adj Close"].iloc[-1]], index=symbols)
        else:
            latest_prices = data["Adj Close"].iloc[-1]
    
    # Calculer valeur actuelle et détails
    current_value = 0.0
    holdings_details = []
    
    for i, holding in enumerate(holdings):
        symbol = holding['symbol']
        quantity = holding['quantity']
        initial_value = holding['initial_value']
        initial_price = holding['initial_price']
        weight = holding['weight']
        
        try:
            # Récupérer prix actuel - TOUJOURS par index pour cohérence
            if isinstance(latest_prices, pd.Series):
                current_price = float(latest_prices.iloc[i])
            else:
                current_price = float(latest_prices[i])
            
            # Calculer valeur de marché actuelle
            market_value = quantity * current_price
            current_value += market_value
            
            # Calculer P&L de ce holding
            holding_pnl = market_value - initial_value
            holding_pnl_pct = (holding_pnl / initial_value * 100) if initial_value > 0 else 0
            
            holdings_details.append({
                'symbol': symbol,
                'name': holding.get('name', symbol),
                'type': holding.get('type', 'stock'),
                'weight': weight,
                'quantity': quantity,
                'initial_price': initial_price,
                'current_price': current_price,
                'initial_value': initial_value,
                'market_value': market_value,
                'pnl': holding_pnl,
                'pnl_pct': holding_pnl_pct
            })
            
        except Exception as e:
            print(f"⚠️ Erreur calcul {symbol}: {e}")
            # Fallback: utiliser valeur initiale
            current_value += initial_value
            holdings_details.append({
                'symbol': symbol,
                'name': holding.get('name', symbol),
                'type': holding.get('type', 'stock'),
                'weight': weight,
                'quantity': quantity,
                'initial_price': initial_price,
                'current_price': initial_price,
                'initial_value': initial_value,
                'market_value': initial_value,
                'pnl': 0.0,
                'pnl_pct': 0.0
            })
    
    # Calculer P&L total
    pnl = current_value - initial_amount
    pnl_pct = (pnl / initial_amount * 100) if initial_amount > 0 else 0.0
    
    return current_value, pnl, pnl_pct, holdings_details


def _calculate_from_legacy(portfolio_data, initial_amount):
    """
    Calcul depuis ancienne structure (assets[], weights[], quantities[])
    Pour compatibilité avec anciens portfolios
    """
    assets = portfolio_data.get('assets', [])
    quantities = portfolio_data.get('quantities', [])
    weights = portfolio_data.get('weights', [])
    
    if not assets or not quantities or len(assets) != len(quantities):
        return initial_amount, 0.0, 0.0, []
    
    # Récupérer les prix actuels
    data = yahoo.retrieve_data(tuple(assets), "1d")
    
    if data is None or data.empty:
        return initial_amount, 0.0, 0.0, []
    
    # Gérer MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        latest_prices = data["Adj Close"].iloc[-1]
    else:
        if len(assets) == 1:
            latest_prices = pd.Series([data["Adj Close"].iloc[-1]], index=assets)
        else:
            latest_prices = data["Adj Close"].iloc[-1]
    
    # Calculer valeur actuelle
    current_value = 0.0
    holdings_details = []
    
    for i, asset in enumerate(assets):
        try:
            # ✅ TOUJOURS par index pour cohérence avec portfolio_details
            if isinstance(latest_prices, pd.Series):
                price = float(latest_prices.iloc[i])
            else:
                price = float(latest_prices[i])
            
            quantity = float(quantities[i])
            market_value = quantity * price
            current_value += market_value
            
            # Calculer valeur initiale
            weight = weights[i] if i < len(weights) else 0
            initial_value = initial_amount * weight
            initial_price = initial_value / quantity if quantity > 0 else 0
            
            holding_pnl = market_value - initial_value
            holding_pnl_pct = (holding_pnl / initial_value * 100) if initial_value > 0 else 0
            
            holdings_details.append({
                'symbol': asset,
                'name': asset,
                'type': 'stock',
                'weight': weight,
                'quantity': quantity,
                'initial_price': initial_price,
                'current_price': price,
                'initial_value': initial_value,
                'market_value': market_value,
                'pnl': holding_pnl,
                'pnl_pct': holding_pnl_pct
            })
            
        except Exception as e:
            print(f"⚠️ Erreur calcul {asset}: {e}")
            continue
    
    # Calculer P&L total
    pnl = current_value - initial_amount
    pnl_pct = (pnl / initial_amount * 100) if initial_amount > 0 else 0.0
    
    return current_value, pnl, pnl_pct, holdings_details


def get_portfolio_summary(portfolio_data):
    """
    Retourne un résumé complet du portfolio
    ✅ FONCTION CENTRALE utilisée par dashboard ET portfolio_manager
    
    Args:
        portfolio_data: Dict portfolio (V2 ou legacy)
    
    Returns:
        dict: {
            'name': str,
            'initial_amount': float,
            'current_value': float,
            'pnl': float,
            'pnl_pct': float,
            'num_holdings': int,
            'holdings': list,
            'model': str,
            'method': str,
            'created_at': str
        }
    """
    current_value, pnl, pnl_pct, holdings_details = calculate_portfolio_current_value(portfolio_data)
    
    return {
        'name': portfolio_data.get('name', 'N/A'),
        'initial_amount': portfolio_data.get('initial_amount', 0),
        'current_value': current_value,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'num_holdings': len(holdings_details),
        'holdings': holdings_details,
        'model': portfolio_data.get('model', 'N/A'),
        'method': portfolio_data.get('method', 'N/A'),
        'created_at': portfolio_data.get('created_at', 'N/A')
    }
