#!/usr/bin/python3
"""
PyManager - Script de Migration MongoDB
Migre les portfolios existants vers la nouvelle structure
"""

import datetime
from pymongo import MongoClient
from bson import ObjectId
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MONGO_URI = 'mongodb://localhost:27017/'
DATABASE_NAME = 'pymanager_db'

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

portfolios_collection = db.portfolios
users_collection = db.users

def backup_database():
    """Backup avant migration"""
    logger.info("📦 Backup de la base de données...")
    
    try:
        # Backup collections
        backup_portfolios = list(portfolios_collection.find())
        backup_users = list(users_collection.find())
        
        # Sauvegarder dans collections backup
        db.portfolios_backup.drop()
        db.users_backup.drop()
        
        if backup_portfolios:
            db.portfolios_backup.insert_many(backup_portfolios)
            logger.info(f"✓ Backup portfolios: {len(backup_portfolios)} documents")
        
        if backup_users:
            db.users_backup.insert_many(backup_users)
            logger.info(f"✓ Backup users: {len(backup_users)} documents")
        
        logger.info("✅ Backup terminé!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur backup: {e}")
        return False

def migrate_portfolios():
    """
    Migration portfolios:
    - total_amount → initial_amount
    - Ajoute current_value
    - Ajoute performance metrics
    """
    logger.info("\n🔄 Migration des portfolios...")
    
    try:
        portfolios = list(portfolios_collection.find())
        count_migrated = 0
        
        for portfolio in portfolios:
            portfolio_id = portfolio['_id']
            
            # Préparer les updates
            updates = {}
            
            # 1. Rename total_amount → initial_amount
            if 'total_amount' in portfolio and 'initial_amount' not in portfolio:
                updates['initial_amount'] = portfolio['total_amount']
                updates['$unset'] = {'total_amount': ""}
                logger.info(f"  • Portfolio {portfolio.get('name')}: total_amount → initial_amount")
            
            # 2. Ajouter current_value si absent
            if 'current_value' not in portfolio:
                updates['current_value'] = portfolio.get('total_amount', 0)
            
            # 3. Ajouter/mettre à jour performance si absent
            if 'performance' not in portfolio:
                updates['performance'] = {
                    'total_return': 0.0,
                    'total_return_pct': 0.0,
                    'ytd_return': 0.0,
                    'mtd_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'volatility': 0.0,
                    'var_95': 0.0,
                    'last_updated': datetime.datetime.utcnow()
                }
            
            # 4. Ajouter metadata si absent
            if 'metadata' not in portfolio:
                updates['metadata'] = {
                    'tags': [],
                    'description': '',
                    'risk_profile': 'moderate',
                    'rebalance_frequency': 'quarterly'
                }
            
            # 5. S'assurer que updated_at existe
            if 'updated_at' not in portfolio:
                updates['updated_at'] = datetime.datetime.utcnow()
            
            # Appliquer les updates
            if updates:
                # Séparer $unset des autres updates
                unset_fields = updates.pop('$unset', None)
                
                if updates:
                    portfolios_collection.update_one(
                        {'_id': portfolio_id},
                        {'$set': updates}
                    )
                
                if unset_fields:
                    portfolios_collection.update_one(
                        {'_id': portfolio_id},
                        {'$unset': unset_fields}
                    )
                
                count_migrated += 1
        
        logger.info(f"✅ Portfolios migrés: {count_migrated}/{len(portfolios)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur migration portfolios: {e}")
        return False

def migrate_users():
    """
    Migration users:
    - Ajoute subscription/license fields
    - Ajoute usage tracking
    """
    logger.info("\n🔄 Migration des users...")
    
    try:
        users = list(users_collection.find())
        count_migrated = 0
        
        for user in users:
            user_id = user['_id']
            updates = {}
            
            # 1. Ajouter subscription si absent
            if 'subscription' not in user:
                updates['subscription'] = {
                    'tier': 'free',  # free, individual, professional, institutional
                    'status': 'active',  # active, trial, expired, cancelled
                    'started_at': user.get('created_at', datetime.datetime.utcnow()),
                    'expires_at': None,
                    'auto_renew': False,
                    'billing_cycle': None,  # monthly, yearly, biennial
                    'payment_method': None
                }
                logger.info(f"  • User {user.get('username')}: Ajout subscription (free)")
            
            # 2. Ajouter license_type si absent
            if 'license_type' not in user:
                updates['license_type'] = 'free'  # free, student, academic, individual, professional, institutional
            
            # 3. Ajouter usage tracking si absent
            if 'usage' not in user:
                updates['usage'] = {
                    'portfolios_count': 0,
                    'ai_queries_today': 0,
                    'market_explorer_today': 0,
                    'last_ai_query': None,
                    'last_login': user.get('created_at', datetime.datetime.utcnow())
                }
            
            # 4. Ajouter verification (pour student/academic)
            if 'verification' not in user:
                updates['verification'] = {
                    'email_verified': False,
                    'student_verified': False,
                    'academic_verified': False,
                    'verification_method': None,  # email_edu, github_student, manual
                    'verified_at': None,
                    'expires_at': None  # Pour student (renew yearly)
                }
            
            # 5. Ajouter profile si absent
            if 'profile' not in user:
                updates['profile'] = {
                    'first_name': user.get('first_name', ''),
                    'last_name': user.get('last_name', ''),
                    'company': '',
                    'role': '',
                    'phone': '',
                    'timezone': 'UTC',
                    'avatar_url': None
                }
            
            # 6. Enrichir preferences
            if 'preferences' in user:
                prefs = user['preferences']
                if 'notifications' not in prefs:
                    updates['preferences.notifications'] = {
                        'email': True,
                        'push': False,
                        'sms': False,
                        'alerts': True,
                        'newsletter': True
                    }
            else:
                updates['preferences'] = {
                    'currency': 'USD',
                    'theme': 'dark',
                    'language': 'fr',
                    'notifications': {
                        'email': True,
                        'push': False,
                        'sms': False,
                        'alerts': True,
                        'newsletter': True
                    }
                }
            
            # Appliquer les updates
            if updates:
                users_collection.update_one(
                    {'_id': user_id},
                    {'$set': updates}
                )
                count_migrated += 1
        
        logger.info(f"✅ Users migrés: {count_migrated}/{len(users)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur migration users: {e}")
        return False

def verify_migration():
    """Vérifie que la migration s'est bien passée"""
    logger.info("\n🔍 Vérification de la migration...")
    
    try:
        # Check portfolios
        portfolios_with_initial = portfolios_collection.count_documents({'initial_amount': {'$exists': True}})
        portfolios_with_total = portfolios_collection.count_documents({'total_amount': {'$exists': True}})
        total_portfolios = portfolios_collection.count_documents({})
        
        logger.info(f"📊 Portfolios:")
        logger.info(f"  • Total: {total_portfolios}")
        logger.info(f"  • Avec initial_amount: {portfolios_with_initial}")
        logger.info(f"  • Avec total_amount (ancien): {portfolios_with_total}")
        
        if portfolios_with_total > 0:
            logger.warning(f"  ⚠️  {portfolios_with_total} portfolios ont encore 'total_amount'")
        
        # Check users
        users_with_subscription = users_collection.count_documents({'subscription': {'$exists': True}})
        users_with_license = users_collection.count_documents({'license_type': {'$exists': True}})
        total_users = users_collection.count_documents({})
        
        logger.info(f"\n👥 Users:")
        logger.info(f"  • Total: {total_users}")
        logger.info(f"  • Avec subscription: {users_with_subscription}")
        logger.info(f"  • Avec license_type: {users_with_license}")
        
        # Sample data
        sample_portfolio = portfolios_collection.find_one()
        if sample_portfolio:
            logger.info(f"\n📋 Exemple portfolio:")
            logger.info(f"  • Name: {sample_portfolio.get('name')}")
            logger.info(f"  • Has initial_amount: {'initial_amount' in sample_portfolio}")
            logger.info(f"  • Has current_value: {'current_value' in sample_portfolio}")
            logger.info(f"  • Has performance: {'performance' in sample_portfolio}")
        
        sample_user = users_collection.find_one()
        if sample_user:
            logger.info(f"\n👤 Exemple user:")
            logger.info(f"  • Username: {sample_user.get('username')}")
            logger.info(f"  • License: {sample_user.get('license_type', 'N/A')}")
            logger.info(f"  • Subscription tier: {sample_user.get('subscription', {}).get('tier', 'N/A')}")
        
        logger.info(f"\n✅ Vérification terminée!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification: {e}")
        return False

def rollback_migration():
    """
    Rollback en cas de problème
    Restaure depuis les backups
    """
    logger.info("\n⚠️  ROLLBACK de la migration...")
    
    try:
        response = input("Êtes-vous sûr de vouloir rollback? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Rollback annulé")
            return
        
        # Restaurer portfolios
        backup_portfolios = list(db.portfolios_backup.find())
        if backup_portfolios:
            portfolios_collection.drop()
            portfolios_collection.insert_many(backup_portfolios)
            logger.info(f"✓ Restauré {len(backup_portfolios)} portfolios")
        
        # Restaurer users
        backup_users = list(db.users_backup.find())
        if backup_users:
            users_collection.drop()
            users_collection.insert_many(backup_users)
            logger.info(f"✓ Restauré {len(backup_users)} users")
        
        logger.info("✅ Rollback terminé!")
        
    except Exception as e:
        logger.error(f"❌ Erreur rollback: {e}")

def main():
    """Script principal de migration"""
    
    print("""
╔════════════════════════════════════════════════════════╗
║          PyManager - Migration Database V2             ║
╚════════════════════════════════════════════════════════╝
    """)
    
    print("Ce script va:")
    print("  1. Backup de la base de données actuelle")
    print("  2. Migrer les portfolios (total_amount → initial_amount)")
    print("  3. Migrer les users (ajout subscription/license)")
    print("  4. Vérifier la migration")
    print()
    
    response = input("Continuer? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration annulée")
        return
    
    print()
    
    # 1. Backup
    if not backup_database():
        print("\n❌ Backup échoué. Migration annulée.")
        return
    
    # 2. Migration portfolios
    if not migrate_portfolios():
        print("\n❌ Migration portfolios échouée.")
        response = input("Rollback? (yes/no): ")
        if response.lower() == 'yes':
            rollback_migration()
        return
    
    # 3. Migration users
    if not migrate_users():
        print("\n❌ Migration users échouée.")
        response = input("Rollback? (yes/no): ")
        if response.lower() == 'yes':
            rollback_migration()
        return
    
    # 4. Vérification
    verify_migration()
    
    print("""
╔════════════════════════════════════════════════════════╗
║              ✅ Migration Terminée!                     ║
╚════════════════════════════════════════════════════════╝

💾 Backups disponibles dans:
  - portfolios_backup
  - users_backup

Pour rollback manuel:
  db.portfolios.drop()
  db.portfolios.insertMany(db.portfolios_backup.find().toArray())
    """)

if __name__ == '__main__':
    main()
