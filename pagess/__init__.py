# pages/__init__.py
"""
Package initialization pour les pages du Portfolio Manager
"""

# Importer les fonctions principales de chaque page
try:
    from .portfolio_manager import render_portfolio_manager
except ImportError as e:
    print(f"Warning: Could not import portfolio_manager: {e}")
    render_portfolio_manager = None

try:
    from .market import render_stock_explorer
except ImportError as e:
    print(f"Warning: Could not import stock_explorer: {e}")
    render_stock_explorer = None

try:
    from .market import render_stock_screener
except ImportError as e:
    print(f"Warning: Could not import stock_screener: {e}")
    render_stock_screener = None
try:
    from .market import render_market
except ImportError as e:
    print(f"Warning: Could not import render_market: {e}")
    render_market = None

try:
    from .ai import render_ai_assistant
except ImportError as e:
    print(f"Warning: Could not import ai_assistant: {e}")
    render_ai_assistant = None

try:
    from .dashboard import render_dashboard
except ImportError as e:
    print(f"Warning: Could not import dashboard: {e}")
    render_dashboard = None

# Définir ce qui est exporté
__all__ = [
    'render_portfolio_manager',
    'render_stock_explorer',
    'render_stock_screener',
    'render_ai_assistant',
    'render_dashboard',
    'render_market'
]
