#!/usr/bin/python3
"""
PyManager Database - MongoDB Multi-User
Gestion complète: Users, Portfolios, Watchlists, Alerts, Transactions
"""

import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from typing import Dict, List, Optional, Any
from bson import ObjectId
import hashlib
from .database import db , watchlists_collection,logger

# =============================================================================
# Watchlist Management
# =============================================================================

def add_to_watchlist(user_id: str, ticker: str, **kwargs) -> Optional[str]:
    """
    Ajoute une action à la watchlist
    
    Args:
        user_id: ID de l'utilisateur
        ticker: Symbole boursier
        **kwargs: notes, target_price, etc.
    """
    try:
        watchlist_doc = {
            "user_id": ObjectId(user_id),
            "ticker": ticker.upper(),
            "added_at": datetime.datetime.utcnow(),
            "notes": kwargs.get("notes", ""),
            "target_price": kwargs.get("target_price"),
            "alert_price_above": kwargs.get("alert_price_above"),
            "alert_price_below": kwargs.get("alert_price_below"),
            "tags": kwargs.get("tags", [])
        }
        
        result = watchlists_collection.insert_one(watchlist_doc)
        logger.info(f"✓ Ajouté à watchlist: {ticker} (user: {user_id})")
        return str(result.inserted_id)
        
    except DuplicateKeyError:
        logger.warning(f"⚠ {ticker} déjà dans la watchlist")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur ajout watchlist: {e}")
        return None

def get_watchlist(user_id: str) -> List[Dict]:
    """Récupère la watchlist d'un utilisateur"""
    try:
        watchlist = list(watchlists_collection.find(
            {"user_id": ObjectId(user_id)}
        ).sort("added_at", DESCENDING))
        
        for item in watchlist:
            item['_id'] = str(item['_id'])
            item['user_id'] = str(item['user_id'])
        
        return watchlist
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération watchlist: {e}")
        return []

def remove_from_watchlist(user_id: str, ticker: str) -> bool:
    """Retire une action de la watchlist"""
    try:
        result = watchlists_collection.delete_one({
            "user_id": ObjectId(user_id),
            "ticker": ticker.upper()
        })
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur suppression watchlist: {e}")
        return False

def update_watchlist_item(user_id: str, ticker: str, updates: Dict) -> bool:
    """Met à jour un élément de la watchlist"""
    try:
        result = watchlists_collection.update_one(
            {"user_id": ObjectId(user_id), "ticker": ticker.upper()},
            {"$set": updates}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour watchlist: {e}")
        return False



# =============================================================================
# Main & Tests
# =============================================================================

if __name__ == '__main__':
    print("🚀 Initialisation de la base de données PyManager...")
    
 
