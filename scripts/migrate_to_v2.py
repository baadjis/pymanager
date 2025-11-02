#!/usr/bin/python3
"""
Script de migration DB: Ancienne structure → V2 (Holdings)

⚠️ IMPORTANT: Faire un backup de la DB avant d'exécuter!

Usage:
    python migrate_to_v2.py --dry-run    # Teste sans modifier
    python migrate_to_v2.py --execute    # Exécute la migration
"""

import argparse
from database.portfolios import migrate_all_portfolios_to_v2, portfolios_collection
from database.database import db
from bson import ObjectId
import datetime


def backup_database():
    """Crée un backup de la collection portfolios"""
    try:
        backup_name = f"portfolios_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Copier tous les documents dans collection backup
        portfolios = list(portfolios_collection.find({}))
        
        if portfolios:
            backup_collection = db[backup_name]
            backup_collection.insert_many(portfolios)
            print(f"✓ Backup créé: {backup_name} ({len(portfolios)} portfolios)")
            return True
        else:
            print("⚠️ Aucun portfolio à sauvegarder")
            return True
            
    except Exception as e:
        print(f"❌ Erreur backup: {e}")
        return False


def analyze_database():
    """Analyse l'état actuel de la DB"""
    try:
        total = portfolios_collection.count_documents({})
        legacy = portfolios_collection.count_documents({
            "assets": {"$exists": True},
            "holdings": {"$exists": False}
        })
        v2 = portfolios_collection.count_documents({
            "holdings": {"$exists": True}
        })
        
        print("\n" + "=" * 60)
        print("DATABASE ANALYSIS")
        print("=" * 60)
        print(f"Total portfolios: {total}")
        print(f"Legacy structure (assets[]): {legacy}")
        print(f"V2 structure (holdings[]): {v2}")
        print("=" * 60 + "\n")
        
        # Afficher quelques exemples
        if legacy > 0:
            print("📋 Exemples de portfolios à migrer:")
            examples = list(portfolios_collection.find({
                "assets": {"$exists": True},
                "holdings": {"$exists": False}
            }).limit(3))
            
            for i, pf in enumerate(examples, 1):
                print(f"\n{i}. {pf.get('name', 'N/A')}")
                print(f"   - Initial amount: ${pf.get('initial_amount', 0):,.2f}")
                print(f"   - Assets: {', '.join(pf.get('assets', []))}")
                print(f"   - Created: {pf.get('created_at', 'N/A')}")
        
        return legacy
        
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return 0


def dry_run_migration():
    """Simule la migration sans modifier la DB"""
    try:
        print("\n" + "=" * 60)
        print("DRY RUN - Simulation de migration")
        print("=" * 60 + "\n")
        
        legacy_portfolios = list(portfolios_collection.find({
            "assets": {"$exists": True},
            "holdings": {"$exists": False}
        }))
        
        if not legacy_portfolios:
            print("✓ Aucun portfolio à migrer!")
            return
        
        print(f"📊 {len(legacy_portfolios)} portfolios seront migrés:\n")
        
        for i, pf in enumerate(legacy_portfolios, 1):
            name = pf.get('name', 'N/A')
            assets = pf.get('assets', [])
            weights = pf.get('weights', [])
            quantities = pf.get('quantities', [])
            initial_amount = pf.get('initial_amount', 0)
            
            print(f"{i}. {name}")
            print(f"   Initial: ${initial_amount:,.2f}")
            
            # Simuler transformation en holdings
            holdings = []
            for j, asset in enumerate(assets):
                weight = weights[j] if j < len(weights) else 0
                quantity = quantities[j] if j < len(quantities) else 0
                initial_value = initial_amount * weight
                initial_price = initial_value / quantity if quantity > 0 else 0
                
                holding = {
                    "symbol": asset,
                    "weight": weight,
                    "quantity": quantity,
                    "initial_price": initial_price,
                    "initial_value": initial_value
                }
                holdings.append(holding)
            
            print(f"   Holdings à créer:")
            for holding in holdings:
                print(f"      - {holding['symbol']}: {holding['quantity']:.4f} @ ${holding['initial_price']:.2f}")
            print()
        
        print("=" * 60)
        print("⚠️ C'est une simulation - aucune modification effectuée")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"❌ Erreur dry run: {e}")


def execute_migration():
    """Exécute la migration réelle"""
    try:
        print("\n" + "=" * 60)
        print("MIGRATION RÉELLE")
        print("=" * 60 + "\n")
        
        confirm = input("⚠️ Êtes-vous sûr de vouloir migrer la DB? (yes/no): ")
        
        if confirm.lower() != 'yes':
            print("❌ Migration annulée")
            return
        
        print("\n1️⃣ Création du backup...")
        if not backup_database():
            print("❌ Backup échoué - migration annulée")
            return
        
        print("\n2️⃣ Migration en cours...")
        migrated, errors = migrate_all_portfolios_to_v2()
        
        print("\n" + "=" * 60)
        print("RÉSULTATS")
        print("=" * 60)
        print(f"✓ Portfolios migrés: {migrated}")
        print(f"❌ Erreurs: {errors}")
        print("=" * 60 + "\n")
        
        if migrated > 0:
            print("✓ Migration terminée avec succès!")
            print("\n📋 Vérifiez vos portfolios dans l'app")
        
    except Exception as e:
        print(f"❌ Erreur migration: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Migration DB PyManager: Legacy → V2 (Holdings)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simule la migration sans modifier la DB'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Exécute la migration réelle'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyse l\'état actuel de la DB'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PyManager - Migration DB vers V2")
    print("=" * 60)
    
    # Analyser DB
    num_legacy = analyze_database()
    
    if args.dry_run:
        dry_run_migration()
    
    elif args.execute:
        if num_legacy > 0:
            execute_migration()
        else:
            print("\n✓ Aucune migration nécessaire!")
    
    elif args.analyze:
        print("\n✓ Analyse terminée")
    
    else:
        print("\n⚠️ Utilisez --dry-run, --execute ou --analyze")
        parser.print_help()


if __name__ == "__main__":
    main()
