#!/usr/bin/python3
"""
PyManager Database - MongoDB Multi-User
database/transactions
"""

import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
import logging
from typing import Dict, List, Optional, Any
from bson import ObjectId
from .database import logger,db ,transactions_collection
import hashlib


# Transactions Management
# =============================================================================

def add_transaction(user_id: str, portfolio_id: str, transaction_type: str, 
                   ticker: str, quantity: float, price: float, **kwargs) -> Optional[str]:
    """
    Ajoute une transaction
    
    Args:
        user_id: ID de l'utilisateur
        portfolio_id: ID du portfolio
        transaction_type: 'buy' ou 'sell'
        ticker: Symbole boursier
        quantity: Quantité
        price: Prix unitaire
        **kwargs: fees, notes, etc.
    """
    try:
        transaction_doc = {
            "user_id": ObjectId(user_id),
            "portfolio_id": ObjectId(portfolio_id),
            "transaction_type": transaction_type.lower(),
            "ticker": ticker.upper(),
            "quantity": quantity,
            "price": price,
            "total_amount": quantity * price,
            "fees": kwargs.get("fees", 0),
            "transaction_date": kwargs.get("transaction_date", datetime.datetime.utcnow()),
            "notes": kwargs.get("notes", ""),
            "created_at": datetime.datetime.utcnow()
        }
        
        result = transactions_collection.insert_one(transaction_doc)
        logger.info(f"✓ Transaction ajoutée: {transaction_type} {quantity} {ticker}")
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"❌ Erreur ajout transaction: {e}")
        return None

def get_transactions(user_id: str, portfolio_id: Optional[str] = None) -> List[Dict]:
    """Récupère les transactions d'un utilisateur"""
    try:
        filter_query = {"user_id": ObjectId(user_id)}
        if portfolio_id:
            filter_query["portfolio_id"] = ObjectId(portfolio_id)
        
        transactions = list(transactions_collection.find(filter_query).sort("transaction_date", DESCENDING))
        
        for txn in transactions:
            txn['_id'] = str(txn['_id'])
            txn['user_id'] = str(txn['user_id'])
            txn['portfolio_id'] = str(txn['portfolio_id'])
        
        return transactions
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération transactions: {e}")
        return []

# =============================================================================
# Main & Tests
# =============================================================================

if __name__ == '__main__':
    print("🚀 Initialisation de la base de données PyManager...")
    
    
