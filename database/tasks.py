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
from .database import db,alerts_collection 



# =============================================================================
# Alerts/Tasks Management
# =============================================================================

def create_alert(user_id: str, ticker: str, alert_type: str, **kwargs) -> Optional[str]:
    """
    Crée une alerte
    
    Args:
        user_id: ID de l'utilisateur
        ticker: Symbole boursier
        alert_type: 'price_above', 'price_below', 'volume', 'news', etc.
        **kwargs: threshold, message, etc.
    """
    try:
        alert_doc = {
            "user_id": ObjectId(user_id),
            "ticker": ticker.upper() if ticker else None,
            "alert_type": alert_type,
            "threshold": kwargs.get("threshold"),
            "condition": kwargs.get("condition", ">="),
            "message": kwargs.get("message", ""),
            "is_active": True,
            "is_triggered": False,
            "created_at": datetime.datetime.utcnow(),
            "triggered_at": None,
            "notification_method": kwargs.get("notification_method", "app"),  # app, email, sms
            "repeat": kwargs.get("repeat", False),
            "priority": kwargs.get("priority", "normal")  # low, normal, high
        }
        
        result = alerts_collection.insert_one(alert_doc)
        logger.info(f"✓ Alerte créée: {alert_type} sur {ticker}")
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"❌ Erreur création alerte: {e}")
        return None

def get_alerts(user_id: str, active_only: bool = True) -> List[Dict]:
    """Récupère les alertes d'un utilisateur"""
    try:
        filter_query = {"user_id": ObjectId(user_id)}
        if active_only:
            filter_query["is_active"] = True
        
        alerts = list(alerts_collection.find(filter_query).sort("created_at", DESCENDING))
        
        for alert in alerts:
            alert['_id'] = str(alert['_id'])
            alert['user_id'] = str(alert['user_id'])
        
        return alerts
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération alertes: {e}")
        return []

def trigger_alert(alert_id: str) -> bool:
    """Marque une alerte comme déclenchée"""
    try:
        result = alerts_collection.update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "is_triggered": True,
                    "triggered_at": datetime.datetime.utcnow(),
                    "is_active": False  # Désactive sauf si repeat=True
                }
            }
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur déclenchement alerte: {e}")
        return False

def delete_alert(alert_id: str, user_id: str) -> bool:
    """Supprime une alerte"""
    try:
        result = alerts_collection.delete_one({
            "_id": ObjectId(alert_id),
            "user_id": ObjectId(user_id)
        })
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur suppression alerte: {e}")
        return False

# =============================================================================
# Main & Tests
# =============================================================================

if __name__ == '__main__':
    print("🚀 Initialisation de la base de données PyManager...")
    
  
