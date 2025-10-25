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

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration MongoDB
MONGO_URI = 'mongodb://localhost:27017/'
DATABASE_NAME = 'pymanager_db'

# Client MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Collections
users_collection = db.users
portfolios_collection = db.portfolios
watchlists_collection = db.watchlists
alerts_collection = db.alerts
transactions_collection = db.transactions

# =============================================================================
# Initialisation de la base de données
# =============================================================================

def init_database():
    """Initialise la base de données avec indexes et contraintes"""
    logger.info("Initialisation de la base de données PyManager...")
    
    try:
        # Index Users
        users_collection.create_index([("email", ASCENDING)], unique=True)
        users_collection.create_index([("username", ASCENDING)], unique=True)
        logger.info("✓ Index users créés")
        
        # Index Portfolios
        portfolios_collection.create_index([("user_id", ASCENDING)])
        portfolios_collection.create_index([("user_id", ASCENDING), ("name", ASCENDING)], unique=True)
        portfolios_collection.create_index([("created_at", DESCENDING)])
        logger.info("✓ Index portfolios créés")
        
        # Index Watchlists
        watchlists_collection.create_index([("user_id", ASCENDING)])
        watchlists_collection.create_index([("user_id", ASCENDING), ("ticker", ASCENDING)], unique=True)
        logger.info("✓ Index watchlists créés")
        
        # Index Alerts
        alerts_collection.create_index([("user_id", ASCENDING)])
        alerts_collection.create_index([("ticker", ASCENDING)])
        alerts_collection.create_index([("is_active", ASCENDING)])
        alerts_collection.create_index([("triggered_at", DESCENDING)])
        logger.info("✓ Index alerts créés")
        
        # Index Transactions
        transactions_collection.create_index([("user_id", ASCENDING)])
        transactions_collection.create_index([("portfolio_id", ASCENDING)])
        transactions_collection.create_index([("transaction_date", DESCENDING)])
        logger.info("✓ Index transactions créés")
        
        logger.info("✅ Base de données initialisée avec succès!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False


if __name__ == '__main__':
    print("🚀 Initialisation de la base de données PyManager...")
    
    # Initialiser la base
    init_database()
    
    # Tests
    print("\n📝 Tests...")
    
    # Test création utilisateur
    user_id = create_user(
        username="testuser",
        email="test@example.com",
        password="password123",
        first_name="Test",
        last_name="User"
    )
    
    if user_id:
        print(f"✓ User créé: {user_id}")
        
        # Test authentification
        user = authenticate_user("testuser", "password123")
        if user:
            print(f"✓ Authentification réussie: {user['username']}")
        
        # Les autres tests nécessitent un objet portfolio réel
        print("\n💡 Pour tester portfolios, watchlist, alerts:")
        print("   Utilisez l'application Streamlit")
    
    print("\n✅ Tests terminés!")
