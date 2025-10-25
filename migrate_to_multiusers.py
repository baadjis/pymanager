#!/usr/bin/python3
"""
Script de migration MongoDB
Ancien schéma (single-user) → Nouveau schéma (multi-user)
"""

import datetime
from pymongo import MongoClient
from bson import ObjectId
import logging

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Connexion MongoDB
OLD_DATABASE = 'portfolios_database'
NEW_DATABASE = 'pymanager_db'

client = MongoClient('mongodb://localhost:27017/')
old_db = client[OLD_DATABASE]
new_db = client[NEW_DATABASE]

# =============================================================================
# Fonctions de migration
# =============================================================================

def create_default_user():
    """Crée un utilisateur par défaut pour les données migrées"""
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = new_db.users.find_one({"username": "admin"})
        if existing_user:
            logger.info("✓ Utilisateur 'admin' existe déjà")
            return str(existing_user['_id'])
        
        # Créer l'utilisateur
        from database import hash_password
        
        user_doc = {
            "username": "admin",
            "email": "admin@pymanager.local",
            "password": hash_password("admin123"),  # Changez ce mot de passe !
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "is_active": True,
            "first_name": "Admin",
            "last_name": "User",
            "preferences": {
                "currency": "USD",
                "theme": "dark",
                "language": "fr"
            }
        }
        
        result = new_db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        logger.info(f"✅ Utilisateur par défaut créé: admin (ID: {user_id})")
        logger.warning("⚠️  IMPORTANT: Changez le mot de passe par défaut (admin123) !")
        
        return user_id
        
    except Exception as e:
        logger.error(f"❌ Erreur création utilisateur: {e}")
        return None

def migrate_portfolios(user_id: str):
    """Migre les portfolios vers le nouveau schéma"""
    try:
        # Récupérer tous les anciens portfolios
        old_portfolios = list(old_db.portfolios.find({}))
        
        if not old_portfolios:
            logger.info("ℹ️  Aucun portfolio à migrer")
            return 0
        
        logger.info(f"📊 Migration de {len(old_portfolios)} portfolios...")
        
        migrated_count = 0
        
        for old_pf in old_portfolios:
            try:
                # Construire le nouveau document
                new_portfolio = {
                    "user_id": ObjectId(user_id),
                    "name": old_pf.get("name", "Portfolio sans nom"),
                    "assets": old_pf.get("assets", []),
                    "weights": old_pf.get("weights", []),
                    "quantities": old_pf.get("quantities", []),
                    "amounts": old_pf.get("amounts", []),
                    "total_amount": old_pf.get("amount", 0),
                    "method": old_pf.get("method", "equal"),
                    "model": old_pf.get("model", "balanced"),
                    "currency": old_pf.get("currency", "USD"),
                    "created_at": old_pf.get("created_at", datetime.datetime.utcnow()),
                    "updated_at": datetime.datetime.utcnow(),
                    "is_active": True,
                    "performance": {
                        "total_return": 0.0,
                        "ytd_return": 0.0,
                        "sharpe_ratio": 0.0,
                        "volatility": 0.0
                    },
                    # Conserver l'ancien ID pour référence
                    "migrated_from": str(old_pf['_id'])
                }
                
                # Insérer dans nouvelle collection
                result = new_db.portfolios.insert_one(new_portfolio)
                
                logger.info(f"  ✓ Migré: {new_portfolio['name']}")
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Erreur migration portfolio {old_pf.get('name')}: {e}")
                continue
        
        logger.info(f"✅ {migrated_count}/{len(old_portfolios)} portfolios migrés")
        return migrated_count
        
    except Exception as e:
        logger.error(f"❌ Erreur migration portfolios: {e}")
        return 0

def migrate_watchlist(user_id: str):
    """
    Migre la watchlist (si elle existe dans l'ancienne DB)
    """
    try:
        # Vérifier si collection watchlist existe
        if 'watchlist' not in old_db.list_collection_names():
            logger.info("ℹ️  Pas de watchlist à migrer")
            return 0
        
        old_watchlist = list(old_db.watchlist.find({}))
        
        if not old_watchlist:
            logger.info("ℹ️  Watchlist vide")
            return 0
        
        logger.info(f"⭐ Migration de {len(old_watchlist)} éléments watchlist...")
        
        migrated_count = 0
        
        for item in old_watchlist:
            try:
                new_item = {
                    "user_id": ObjectId(user_id),
                    "ticker": item.get("ticker", "").upper(),
                    "added_at": item.get("added_at", datetime.datetime.utcnow()),
                    "notes": item.get("notes", ""),
                    "target_price": item.get("target_price"),
                    "tags": item.get("tags", [])
                }
                
                new_db.watchlists.insert_one(new_item)
                logger.info(f"  ✓ Migré: {new_item['ticker']}")
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Erreur migration watchlist item: {e}")
                continue
        
        logger.info(f"✅ {migrated_count}/{len(old_watchlist)} items watchlist migrés")
        return migrated_count
        
    except Exception as e:
        logger.error(f"❌ Erreur migration watchlist: {e}")
        return 0

def migrate_transactions(user_id: str):
    """Migre les transactions (si elles existent)"""
    try:
        if 'transactions' not in old_db.list_collection_names():
            logger.info("ℹ️  Pas de transactions à migrer")
            return 0
        
        old_transactions = list(old_db.transactions.find({}))
        
        if not old_transactions:
            logger.info("ℹ️  Aucune transaction")
            return 0
        
        logger.info(f"💳 Migration de {len(old_transactions)} transactions...")
        
        # Mapper anciens portfolio IDs → nouveaux
        portfolio_mapping = {}
        for old_pf in old_db.portfolios.find({}):
            new_pf = new_db.portfolios.find_one({"migrated_from": str(old_pf['_id'])})
            if new_pf:
                portfolio_mapping[str(old_pf['_id'])] = str(new_pf['_id'])
        
        migrated_count = 0
        
        for txn in old_transactions:
            try:
                old_portfolio_id = str(txn.get("portfolio_id", ""))
                new_portfolio_id = portfolio_mapping.get(old_portfolio_id)
                
                if not new_portfolio_id:
                    logger.warning(f"  ⚠ Portfolio non trouvé pour transaction")
                    continue
                
                new_txn = {
                    "user_id": ObjectId(user_id),
                    "portfolio_id": ObjectId(new_portfolio_id),
                    "transaction_type": txn.get("type", "buy").lower(),
                    "ticker
