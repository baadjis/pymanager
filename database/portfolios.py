#!/usr/bin/python3
"""
PyManager Database - Portfolios Management V2
✅ Structure holdings[] avec toutes les infos par asset
✅ PnL calculé 100% live via utils.py
✅ Migration depuis ancienne structure
"""

import datetime
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError
from .database import db, logger, portfolios_collection
from typing import Dict, List, Optional
from bson import ObjectId

# =============================================================================
# Portfolios Management V2
# =============================================================================

def save_portfolio(user_id: str, portfolio, name: str, **kwargs) -> Optional[str]:
    """
    Sauvegarde un portfolio V2 avec structure holdings[]
    
    Args:
        user_id: ID de l'utilisateur
        portfolio: Objet portfolio avec assets, weights, data
        name: Nom du portfolio
        **kwargs: amount (initial_amount), method, model, etc.
    
    Returns:
        portfolio_id si succès, None sinon
    """
    try:
        d = dict(kwargs)
        initial_amount = d.get("amount", 0)
        assets = portfolio.assets
        weights = portfolio.weights
        
        # Récupérer les prix initiaux
        mtms = list(portfolio.data["Adj Close"].values[-1])
        
        # Construire holdings[] avec toutes les infos
        holdings = []
        for i, (asset, weight) in enumerate(zip(assets, weights)):
            initial_value = initial_amount * weight
            initial_price = float(mtms[i])
            quantity = initial_value / initial_price
            
            holding = {
                "symbol": asset,
                "name": get_asset_name(asset),  # TODO: fetch depuis yfinance
                "type": "stock",  # TODO: détecter le type (stock, etf, crypto)
                "weight": weight,
                "quantity": quantity,
                "initial_price": initial_price,
                "initial_value": initial_value,
                "purchase_date": datetime.datetime.utcnow().strftime("%Y-%m-%d")
            }
            holdings.append(holding)
        
        portfolio_doc = {
            # Identifiers
            "user_id": ObjectId(user_id),
            "name": name,
            
            # Holdings - NOUVELLE STRUCTURE
            "holdings": holdings,
            "initial_amount": initial_amount,
            
            # Strategy
            "method": d.get("method", "equal"),
            "model": d.get("model", "balanced"),
            "rebalance_frequency": d.get("rebalance_frequency", "quarterly"),
            
            # Metadata
            "currency": d.get("currency", "USD"),
            "metadata": {
                "tags": d.get("tags", []),
                "description": d.get("description", ""),
                "risk_profile": d.get("risk_profile", "moderate"),
                "investment_goal": d.get("investment_goal", ""),
                "time_horizon": d.get("time_horizon", "")
            },
            
            # Status
            "is_active": True,
            "is_paper": d.get("is_paper", True),
            
            # Timestamps
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "last_rebalanced": datetime.datetime.utcnow()
        }
        
        result = portfolios_collection.insert_one(portfolio_doc)
        logger.info(f"✓ Portfolio V2 sauvegardé: {name} (user: {user_id}, initial: ${initial_amount})")
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
    ✅ Support ancienne structure (legacy) ET nouvelle structure (V2)
    """
    try:
        filter_query = {"user_id": ObjectId(user_id)}
        
        if 'is_active' not in kwargs:
            filter_query["is_active"] = True
        else:
            filter_query["is_active"] = kwargs.pop('is_active')
        
        filter_query.update(kwargs)
        
        portfolios = list(portfolios_collection.find(filter_query).sort("created_at", DESCENDING))
        
        # Convertir ObjectId et normaliser structure
        for p in portfolios:
            p['_id'] = str(p['_id'])
            p['user_id'] = str(p['user_id'])
            
            # ✅ Normaliser: convertir ancienne structure en nouvelle
            if 'holdings' not in p and 'assets' in p:
                p = _migrate_portfolio_to_v2(p)
        
        return portfolios
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolios: {e}")
        return []


def get_single_portfolio(user_id: str, name: str) -> Optional[Dict]:
    """Récupère un portfolio spécifique"""
    try:
        portfolio = portfolios_collection.find_one({
            "user_id": ObjectId(user_id),
            "name": name,
            "is_active": True
        })
        
        if portfolio:
            portfolio['_id'] = str(portfolio['_id'])
            portfolio['user_id'] = str(portfolio['user_id'])
            
            # Normaliser structure
            if 'holdings' not in portfolio and 'assets' in portfolio:
                portfolio = _migrate_portfolio_to_v2(portfolio)
        
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
            
            # Normaliser structure
            if 'holdings' not in portfolio and 'assets' in portfolio:
                portfolio = _migrate_portfolio_to_v2(portfolio)
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolio: {e}")
        return None


# =============================================================================
# Migration Legacy → V2
# =============================================================================

def _migrate_portfolio_to_v2(portfolio: Dict) -> Dict:
    """
    Convertit un portfolio legacy (assets[], weights[], quantities[])
    en structure V2 (holdings[])
    """
    try:
        assets = portfolio.get('assets', [])
        weights = portfolio.get('weights', [])
        quantities = portfolio.get('quantities', [])
        amounts = portfolio.get('amounts', [])
        initial_amount = portfolio.get('initial_amount', 0)
        
        holdings = []
        for i, asset in enumerate(assets):
            weight = weights[i] if i < len(weights) else 0
            quantity = quantities[i] if i < len(quantities) else 0
            initial_value = amounts[i] if i < len(amounts) else (initial_amount * weight)
            initial_price = initial_value / quantity if quantity > 0 else 0
            
            holding = {
                "symbol": asset,
                "name": get_asset_name(asset),
                "type": "stock",
                "weight": weight,
                "quantity": quantity,
                "initial_price": initial_price,
                "initial_value": initial_value,
                "purchase_date": portfolio.get('created_at', datetime.datetime.utcnow()).strftime("%Y-%m-%d")
            }
            holdings.append(holding)
        
        portfolio['holdings'] = holdings
        
        # Garder anciens champs pour compatibilité temporaire
        # portfolio['_legacy'] = True
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Erreur migration portfolio: {e}")
        return portfolio


def migrate_all_portfolios_to_v2():
    """
    ⚠️ Migration complète DB: ancienne structure → V2
    À exécuter UNE FOIS
    """
    try:
        # Trouver tous les portfolios avec ancienne structure
        legacy_portfolios = list(portfolios_collection.find({
            "assets": {"$exists": True},
            "holdings": {"$exists": False}
        }))
        
        migrated = 0
        errors = 0
        
        for portfolio in legacy_portfolios:
            try:
                # Migrer structure
                migrated_pf = _migrate_portfolio_to_v2(portfolio)
                
                # Mettre à jour en DB
                portfolios_collection.update_one(
                    {"_id": portfolio['_id']},
                    {
                        "$set": {
                            "holdings": migrated_pf['holdings'],
                            "updated_at": datetime.datetime.utcnow()
                        },
                        "$unset": {
                            "assets": "",
                            "weights": "",
                            "quantities": "",
                            "amounts": ""
                        }
                    }
                )
                migrated += 1
                
            except Exception as e:
                logger.error(f"❌ Erreur migration portfolio {portfolio.get('name', 'N/A')}: {e}")
                errors += 1
        
        logger.info(f"✓ Migration terminée: {migrated} portfolios migrés, {errors} erreurs")
        return migrated, errors
        
    except Exception as e:
        logger.error(f"❌ Erreur migration globale: {e}")
        return 0, 0


# =============================================================================
# Helpers
# =============================================================================

def get_asset_name(symbol: str) -> str:
    """Récupère le nom complet d'un asset"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        return ticker.info.get('longName', symbol)
    except:
        return symbol


def update_portfolio(portfolio_id: str, updates: Dict) -> bool:
    """Met à jour un portfolio V2"""
    try:
        # Interdire modification des champs calculés
        forbidden_fields = ['current_value', 'pnl', 'pnl_pct', 'performance']
        for field in forbidden_fields:
            if field in updates:
                del updates[field]
                logger.warning(f"⚠️ Champ '{field}' ignoré (calculé live)")
        
        if not updates:
            return False
        
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
            {"$set": {
                "is_active": False,
                "deleted_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur suppression portfolio: {e}")
        return False


if __name__ == '__main__':
    print("🚀 Portfolios Management V2 (Holdings Structure)")
    print("\n⚠️ Pour migrer la base de données:")
    print("from database.portfolios import migrate_all_portfolios_to_v2")
    print("migrate_all_portfolios_to_v2()")
