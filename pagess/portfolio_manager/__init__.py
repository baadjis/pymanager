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


# Définir ce qui est exporté
__all__ = [
    'render_portfolio_manager',
   
]
