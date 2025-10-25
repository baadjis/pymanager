#!/usr/bin/python3
"""
PyManager Database - MongoDB Multi-User
Gestion complète: Users, Portfolios, Watchlists, Alerts, Transactions
"""

import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
import logging
from typing import Dict, List, Optional, Any
from bson import ObjectId
import hashlib
from .database import db ,logger,portfolios_collection


# Portfolios Management
# =============================================================================

def save_portfolio(user_id: str, portfolio, name: str, **kwargs) -> Optional[str]:
    """
    Sauvegarde un portfolio pour un utilisateur
    
    Args:
        user_id: ID de l'utilisateur
        portfolio: Objet portfolio avec assets, weights, data
        name: Nom du portfolio
        **kwargs: amount, method, etc.
    
    Returns:
        portfolio_id si succès, None sinon
    """
    try:
        d = dict(kwargs)
        amount = d.get("amount", 0)
        assets = portfolio.assets
        weights = portfolio.weights
        amounts = [amount * w for w in weights]
        mtms = list(portfolio.data["Adj Close"].values[-1])
        quantities = [p[0]/p[1] for p in zip(amounts, mtms)]
        
        portfolio_doc = {
            "user_id": ObjectId(user_id),
            "name": name,
            "assets": assets,
            "weights": weights,
            "quantities": quantities,
            "amounts": amounts,
            "total_amount": amount,
            "method": d.get("method", "equal"),
            "model": d.get("model", "balanced"),
            "currency": d.get("currency", "USD"),
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "is_active": True,
            "performance": {
                "total_return": 0.0,
                "ytd_return": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0
            }
        }
        
        result = portfolios_collection.insert_one(portfolio_doc)
        logger.info(f"✓ Portfolio sauvegardé: {name} (user: {user_id})")
        return str(result.inserted_id)
        
    except DuplicateKeyError:
        logger.error(f"❌ Portfolio déjà existant: {name} pour cet utilisateur")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde portfolio: {e}")
        return None

def get_portfolios(user_id: str, **kwargs) -> List[Dict]:
    """
    Récupère les portfolios d'un utilisateur
    
    Args:
        user_id: ID de l'utilisateur
        **kwargs: Filtres additionnels (method, model, etc.)
    
    Returns:
        Liste de portfolios
    """
    try:
        filter_query = {"user_id": ObjectId(user_id), **kwargs}
        portfolios = list(portfolios_collection.find(filter_query).sort("created_at", DESCENDING))
        
        # Convertir ObjectId en string
        for p in portfolios:
            p['_id'] = str(p['_id'])
            p['user_id'] = str(p['user_id'])
        
        return portfolios
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolios: {e}")
        return []

def get_single_portfolio(user_id: str, name: str) -> Optional[Dict]:
    """Récupère un portfolio spécifique d'un utilisateur"""
    try:
        portfolio = portfolios_collection.find_one({
            "user_id": ObjectId(user_id),
            "name": name
        })
        
        if portfolio:
            portfolio['_id'] = str(portfolio['_id'])
            portfolio['user_id'] = str(portfolio['user_id'])
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolio: {e}")
        return None

def get_portfolio_by_id(portfolio_id: str) -> Optional[Dict]:
    """Récupère un portfolio par son ID"""
    try:
        portfolio = portfolios_collection.find_one({"_id": ObjectId(portfolio_id)})
        
        if portfolio:
            portfolio['_id'] = str(portfolio['_id'])
            portfolio['user_id'] = str(portfolio['user_id'])
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolio: {e}")
        return None

def update_portfolio(portfolio_id: str, updates: Dict) -> bool:
    """Met à jour un portfolio"""
    try:
        updates['updated_at'] = datetime.datetime.utcnow()
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour portfolio: {e}")
        return False

def delete_portfolio(portfolio_id: str, user_id: str) -> bool:
    """Supprime un portfolio (soft delete)"""
    try:
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id), "user_id": ObjectId(user_id)},
            {"$set": {"is_active": False, "deleted_at": datetime.datetime.utcnow()}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur suppression portfolio: {e}")
        return False





if __name__ == '__main__':
    print("🚀 Initialisation de la base de données PyManager...")
    

