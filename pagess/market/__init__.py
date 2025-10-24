# pages/__init__.py
"""
Package initialization pour les pages du Portfolio Manager
"""



try:
    from .stock_explorer import render_explorer
except ImportError as e:
    print(f"Warning: Could not import stock_explorer: {e}")
    render_stock_explorer = None

try:
    from .screener import render_screener
except ImportError as e:
    print(f"Warning: Could not import stock_screener: {e}")
    render_stock_screener = None

try:
    from .market_overview import render_market_oveview
except ImportError as e:
    print(f"Warning: Could not import render_market_overview: {e}")
    render_market_overview = None

    
try:
    from .market import render_market
except ImportError as e:
    print(f"Warning: Could not import render_market: {e}")
    render_market = None



# Définir ce qui est exporté
__all__ = [
    
    'render_explorer',
    'render_screener',
    'render_market',
    'render_market_overview'
   
]
