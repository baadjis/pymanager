#!/usr/bin/python3
"""
PyManager Database - Portfolios Management
total_amount → initial_amount
"""

import datetime
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError
from .database import db, logger, portfolios_collection
from typing import Dict, List, Optional
from bson import ObjectId

# =============================================================================
# Portfolios Management
# =============================================================================

def save_portfolio(user_id: str, portfolio, name: str, **kwargs) -> Optional[str]:
    """
    Sauvegarde un portfolio pour un utilisateur
    
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
        initial_amount = d.get("amount", 0)  # Renommé de total_amount
        assets = portfolio.assets
        weights = portfolio.weights
        amounts = [initial_amount * w for w in weights]
        mtms = list(portfolio.data["Adj Close"].values[-1])
        quantities = [p[0]/p[1] for p in zip(amounts, mtms)]
        
        portfolio_doc = {
            # Identifiers
            "user_id": ObjectId(user_id),
            "name": name,
            
            # Assets
            "assets": assets,
            "weights": weights,
            "quantities": quantities,
            "amounts": amounts,  # Montants par asset
            
            # Values
            "initial_amount": initial_amount,  # ✅ Changé de total_amount
            "current_value": initial_amount,   # Initialement = initial_amount
            
            # Strategy
            "method": d.get("method", "equal"),
            "model": d.get("model", "balanced"),
            "rebalance_frequency": d.get("rebalance_frequency", "quarterly"),
            
            # Performance
            "performance": {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "ytd_return": 0.0,
                "mtd_return": 0.0,
                "wtd_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "last_updated": datetime.datetime.utcnow()
            },
            
            # Metadata
            "currency": d.get("currency", "USD"),
            "metadata": {
                "tags": d.get("tags", []),
                "description": d.get("description", ""),
                "risk_profile": d.get("risk_profile", "moderate"),  # conservative, moderate, aggressive
                "investment_goal": d.get("investment_goal", ""),  # growth, income, balanced
                "time_horizon": d.get("time_horizon", "")  # short, medium, long
            },
            
            # Status
            "is_active": True,
            "is_paper": d.get("is_paper", True),  # Paper trading vs real
            
            # Timestamps
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "last_rebalanced": datetime.datetime.utcnow()
        }
        
        result = portfolios_collection.insert_one(portfolio_doc)
        logger.info(f"✓ Portfolio sauvegardé: {name} (user: {user_id}, initial: ${initial_amount})")
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
        **kwargs: Filtres (method, model, is_active, etc.)
    
    Returns:
        Liste de portfolios
    """
    try:
        filter_query = {"user_id": ObjectId(user_id)}
        
        # Ajouter filtres optionnels
        if 'is_active' not in kwargs:
            filter_query["is_active"] = True  # Par défaut, seulement les actifs
        else:
            filter_query["is_active"] = kwargs.pop('is_active')
        
        filter_query.update(kwargs)
        
        portfolios = list(portfolios_collection.find(filter_query).sort("created_at", DESCENDING))
        
        # Convertir ObjectId en string
        for p in portfolios:
            p['_id'] = str(p['_id'])
            p['user_id'] = str(p['user_id'])
            
            # Calculer P&L si possible
            if 'current_value' in p and 'initial_amount' in p:
                initial = p['initial_amount']
                current = p['current_value']
                if initial > 0:
                    p['pnl'] = current - initial
                    p['pnl_pct'] = ((current - initial) / initial) * 100
        
        return portfolios
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolios: {e}")
        return []

def get_single_portfolio(user_id: str, name: str) -> Optional[Dict]:
    """Récupère un portfolio spécifique d'un utilisateur"""
    try:
        portfolio = portfolios_collection.find_one({
            "user_id": ObjectId(user_id),
            "name": name,
            "is_active": True
        })
        
        if portfolio:
            portfolio['_id'] = str(portfolio['_id'])
            portfolio['user_id'] = str(portfolio['user_id'])
            
            # Calculer P&L
            if 'current_value' in portfolio and 'initial_amount' in portfolio:
                initial = portfolio['initial_amount']
                current = portfolio['current_value']
                if initial > 0:
                    portfolio['pnl'] = current - initial
                    portfolio['pnl_pct'] = ((current - initial) / initial) * 100
        
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
            
            # Calculer P&L
            if 'current_value' in portfolio and 'initial_amount' in portfolio:
                initial = portfolio['initial_amount']
                current = portfolio['current_value']
                if initial > 0:
                    portfolio['pnl'] = current - initial
                    portfolio['pnl_pct'] = ((current - initial) / initial) * 100
        
        return portfolio
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération portfolio: {e}")
        return None

def update_portfolio(portfolio_id: str, updates: Dict) -> bool:
    """
    Met à jour un portfolio
    
    Args:
        portfolio_id: ID du portfolio
        updates: Dict des champs à mettre à jour
    """
    try:
        updates['updated_at'] = datetime.datetime.utcnow()
        
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"✓ Portfolio mis à jour: {portfolio_id}")
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour portfolio: {e}")
        return False

def update_portfolio_value(portfolio_id: str, new_current_value: float) -> bool:
    """
    Met à jour la valeur actuelle d'un portfolio et recalcule performance
    
    Args:
        portfolio_id: ID du portfolio
        new_current_value: Nouvelle valeur actuelle
    """
    try:
        portfolio = get_portfolio_by_id(portfolio_id)
        if not portfolio:
            return False
        
        initial = portfolio.get('initial_amount', 0)
        
        if initial > 0:
            total_return = new_current_value - initial
            total_return_pct = (total_return / initial) * 100
            
            updates = {
                "current_value": new_current_value,
                "performance.total_return": total_return,
                "performance.total_return_pct": total_return_pct,
                "performance.last_updated": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            result = portfolios_collection.update_one(
                {"_id": ObjectId(portfolio_id)},
                {"$set": updates}
            )
            
            return result.modified_count > 0
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur update value: {e}")
        return False

def update_portfolio_performance(portfolio_id: str, performance_metrics: Dict) -> bool:
    """
    Met à jour les métriques de performance
    
    Args:
        portfolio_id: ID du portfolio
        performance_metrics: Dict avec sharpe, sortino, var, etc.
    """
    try:
        updates = {}
        for key, value in performance_metrics.items():
            updates[f"performance.{key}"] = value
        
        updates["performance.last_updated"] = datetime.datetime.utcnow()
        updates["updated_at"] = datetime.datetime.utcnow()
        
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": updates}
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur update performance: {e}")
        return False

def delete_portfolio(portfolio_id: str, user_id: str) -> bool:
    """
    Supprime un portfolio (soft delete)
    
    Args:
        portfolio_id: ID du portfolio
        user_id: ID du user (pour vérification)
    """
    try:
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id), "user_id": ObjectId(user_id)},
            {"$set": {
                "is_active": False,
                "deleted_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }}
        )
        
        if result.modified_count > 0:
            logger.info(f"✓ Portfolio supprimé: {portfolio_id}")
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur suppression portfolio: {e}")
        return False

def restore_portfolio(portfolio_id: str, user_id: str) -> bool:
    """Restaure un portfolio supprimé"""
    try:
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id), "user_id": ObjectId(user_id)},
            {
                "$set": {"is_active": True, "updated_at": datetime.datetime.utcnow()},
                "$unset": {"deleted_at": ""}
            }
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur restore portfolio: {e}")
        return False

def rebalance_portfolio(portfolio_id: str, new_weights: List[float]) -> bool:
    """
    Rééquilibre un portfolio avec de nouveaux poids
    
    Args:
        portfolio_id: ID du portfolio
        new_weights: Nouveaux poids (doivent sommer à 1.0)
    """
    try:
        if abs(sum(new_weights) - 1.0) > 0.001:
            logger.error("❌ Les poids ne somment pas à 1.0")
            return False
        
        portfolio = get_portfolio_by_id(portfolio_id)
        if not portfolio:
            return False
        
        current_value = portfolio.get('current_value', 0)
        new_amounts = [current_value * w for w in new_weights]
        
        # Calculer nouvelles quantités (nécessite current prices)
        # TODO: Fetch current prices from Yahoo
        
        updates = {
            "weights": new_weights,
            "amounts": new_amounts,
            "last_rebalanced": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        result = portfolios_collection.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"✓ Portfolio rééquilibré: {portfolio_id}")
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur rebalance: {e}")
        return False

# =============================================================================
# Stats & Analytics
# =============================================================================

def get_portfolio_stats(user_id: str) -> Dict:
    """Statistiques des portfolios d'un user"""
    try:
        portfolios = get_portfolios(user_id, is_active=True)
        
        if not portfolios:
            return {
                "count": 0,
                "total_value": 0,
                "total_initial": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0
            }
        
        total_value = sum(p.get('current_value', 0) for p in portfolios)
        total_initial = sum(p.get('initial_amount', 0) for p in portfolios)
        total_pnl = total_value - total_initial
        total_pnl_pct = (total_pnl / total_initial * 100) if total_initial > 0 else 0
        
        return {
            "count": len(portfolios),
            "total_value": total_value,
            "total_initial": total_initial,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "by_model": {},  # TODO: Group by model
            "by_risk": {}    # TODO: Group by risk profile
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur get stats: {e}")
        return {}

def get_top_performers(user_id: str, limit: int = 5) -> List[Dict]:
    """Retourne les portfolios les plus performants"""
    try:
        portfolios = get_portfolios(user_id, is_active=True)
        
        # Trier par pnl_pct décroissant
        sorted_portfolios = sorted(
            portfolios,
            key=lambda p: p.get('pnl_pct', 0),
            reverse=True
        )
        
        return sorted_portfolios[:limit]
        
    except Exception as e:
        logger.error(f"❌ Erreur get top performers: {e}")
        return []

if __name__ == '__main__':
    print("🚀 Portfolios Management Module")
