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
                    "ticker": txn.get("ticker", "").upper(),
                    "quantity": txn.get("quantity", 0),
                    "price": txn.get("price", 0),
                    "total_amount": txn.get("total_amount", txn.get("quantity", 0) * txn.get("price", 0)),
                    "fees": txn.get("fees", 0),
                    "transaction_date": txn.get("date", datetime.datetime.utcnow()),
                    "notes": txn.get("notes", ""),
                    "created_at": txn.get("created_at", datetime.datetime.utcnow())
                }
                
                new_db.transactions.insert_one(new_txn)
                logger.info(f"  ✓ Migré: {new_txn['transaction_type']} {new_txn['ticker']}")
                migrated_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Erreur migration transaction: {e}")
                continue
        
        logger.info(f"✅ {migrated_count}/{len(old_transactions)} transactions migrées")
        return migrated_count
        
    except Exception as e:
        logger.error(f"❌ Erreur migration transactions: {e}")
        return 0

def create_backup():
    """Crée une backup de l'ancienne base de données"""
    try:
        import subprocess
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{OLD_DATABASE}_{timestamp}"
        
        logger.info(f"💾 Création backup: {backup_name}")
        
        cmd = f"mongodump --db {OLD_DATABASE} --out ./backups/{backup_name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Backup créée: ./backups/{backup_name}")
            return True
        else:
            logger.error(f"❌ Erreur backup: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur création backup: {e}")
        logger.warning("⚠️  Continuez sans backup? (non recommandé)")
        return False

def verify_migration(user_id: str):
    """Vérifie que la migration s'est bien passée"""
    try:
        logger.info("\n🔍 Vérification de la migration...")
        
        # Compter les documents
        old_portfolios_count = old_db.portfolios.count_documents({})
        new_portfolios_count = new_db.portfolios.count_documents({"user_id": ObjectId(user_id)})
        
        logger.info(f"📊 Portfolios:")
        logger.info(f"   Ancien: {old_portfolios_count}")
        logger.info(f"   Nouveau: {new_portfolios_count}")
        
        if old_portfolios_count == new_portfolios_count:
            logger.info("   ✅ Correspondance parfaite")
        else:
            logger.warning(f"   ⚠️  Différence: {old_portfolios_count - new_portfolios_count}")
        
        # Vérifier les index
        logger.info("\n🔍 Vérification des index...")
        
        collections_to_check = ['users', 'portfolios', 'watchlists', 'alerts', 'transactions']
        for coll_name in collections_to_check:
            indexes = new_db[coll_name].index_information()
            logger.info(f"   {coll_name}: {len(indexes)} index")
        
        logger.info("\n✅ Vérification terminée")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification: {e}")
        return False

def print_summary(stats: dict):
    """Affiche un résumé de la migration"""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE LA MIGRATION")
    print("="*60)
    print(f"👤 Utilisateur créé: {stats.get('user_created', False)}")
    print(f"📊 Portfolios migrés: {stats.get('portfolios', 0)}")
    print(f"⭐ Watchlist migrée: {stats.get('watchlist', 0)}")
    print(f"💳 Transactions migrées: {stats.get('transactions', 0)}")
    print(f"⏱️  Durée: {stats.get('duration', 0):.2f}s")
    print("="*60)
    print("\n✅ Migration terminée avec succès!")
    print("\n📝 NOTES IMPORTANTES:")
    print("   1. Mot de passe par défaut: admin123 (CHANGEZ-LE!)")
    print("   2. Username: admin")
    print("   3. Email: admin@pymanager.local")
    print(f"   4. Backup disponible: ./backups/")
    print("\n💡 Prochaines étapes:")
    print("   1. Tester la nouvelle base de données")
    print("   2. Changer le mot de passe admin")
    print("   3. Créer d'autres utilisateurs si nécessaire")
    print("   4. Mettre à jour app3.py pour utiliser database.py")
    print("="*60)

# =============================================================================
# Script principal
# =============================================================================

def main():
    """Fonction principale de migration"""
    import time
    start_time = time.time()
    
    stats = {
        'user_created': False,
        'portfolios': 0,
        'watchlist': 0,
        'transactions': 0,
        'duration': 0
    }
    
    print("="*60)
    print("🚀 MIGRATION MONGODB - PyManager")
    print("="*60)
    print(f"Base source: {OLD_DATABASE}")
    print(f"Base destination: {NEW_DATABASE}")
    print("="*60)
    
    # Demander confirmation
    response = input("\n⚠️  Voulez-vous créer une backup avant de continuer? (O/n): ")
    if response.lower() != 'n':
        create_backup()
    
    response = input("\n▶️  Continuer avec la migration? (O/n): ")
    if response.lower() == 'n':
        print("❌ Migration annulée")
        return
    
    print("\n🚀 Début de la migration...\n")
    
    try:
        # Étape 1: Initialiser la nouvelle base
        logger.info("1️⃣  Initialisation de la nouvelle base de données...")
        from database import init_database
        init_database()
        
        # Étape 2: Créer utilisateur par défaut
        logger.info("\n2️⃣  Création de l'utilisateur par défaut...")
        user_id = create_default_user()
        if not user_id:
            logger.error("❌ Impossible de créer l'utilisateur")
            return
        stats['user_created'] = True
        
        # Étape 3: Migrer les portfolios
        logger.info("\n3️⃣  Migration des portfolios...")
        stats['portfolios'] = migrate_portfolios(user_id)
        
        # Étape 4: Migrer la watchlist
        logger.info("\n4️⃣  Migration de la watchlist...")
        stats['watchlist'] = migrate_watchlist(user_id)
        
        # Étape 5: Migrer les transactions
        logger.info("\n5️⃣  Migration des transactions...")
        stats['transactions'] = migrate_transactions(user_id)
        
        # Étape 6: Vérification
        logger.info("\n6️⃣  Vérification de la migration...")
        verify_migration(user_id)
        
        # Calculer durée
        stats['duration'] = time.time() - start_time
        
        # Afficher résumé
        print_summary(stats)
        
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}")
        logger.error("La migration a échoué. Vérifiez les logs ci-dessus.")
        return

def rollback():
    """Fonction de rollback en cas de problème"""
    print("="*60)
    print("↩️  ROLLBACK - Restauration de la backup")
    print("="*60)
    
    import os
    import subprocess
    
    # Lister les backups disponibles
    backup_dir = "./backups"
    if not os.path.exists(backup_dir):
        print("❌ Aucun dossier backup trouvé")
        return
    
    backups = sorted([d for d in os.listdir(backup_dir) if d.startswith("backup_")])
    
    if not backups:
        print("❌ Aucune backup disponible")
        return
    
    print("\n📦 Backups disponibles:")
    for i, backup in enumerate(backups, 1):
        print(f"  {i}. {backup}")
    
    choice = input(f"\n Choisissez une backup (1-{len(backups)}) ou 0 pour annuler: ")
    
    try:
        choice = int(choice)
        if choice == 0:
            print("❌ Rollback annulé")
            return
        
        if 1 <= choice <= len(backups):
            backup_name = backups[choice - 1]
            backup_path = os.path.join(backup_dir, backup_name)
            
            print(f"\n🔄 Restauration de {backup_name}...")
            
            cmd = f"mongorestore --db {OLD_DATABASE} {backup_path}/{OLD_DATABASE}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Backup restaurée avec succès")
            else:
                print(f"❌ Erreur: {result.stderr}")
        else:
            print("❌ Choix invalide")
            
    except ValueError:
        print("❌ Entrée invalide")
    except Exception as e:
        print(f"❌ Erreur: {e}")

# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         PyManager - Migration MongoDB                     ║
    ║         Single-user → Multi-user                          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'rollback':
            rollback()
        elif sys.argv[1] == 'verify':
            # Vérification sans migration
            from database import init_database
            init_database()
            user = new_db.users.find_one({"username": "admin"})
            if user:
                verify_migration(str(user['_id']))
            else:
                print("❌ Utilisateur admin non trouvé")
        else:
            print("Usage:")
            print("  python migrate_to_multiuser.py          # Migration normale")
            print("  python migrate_to_multiuser.py rollback # Restaurer backup")
            print("  python migrate_to_multiuser.py verify   # Vérifier migration")
    else:
        main()
