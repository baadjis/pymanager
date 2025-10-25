#!/usr/bin/python3
"""
PyManager Database - MongoDB Multi-User
Gestion complète: Users
"""

import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from .database import logger,users_collection,db
from typing import Dict, List, Optional, Any
from bson import ObjectId
import hashlib 


# =============================================================================
# Users Management
# =============================================================================

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, email: str, password: str, **kwargs) -> Optional[str]:
    """
    Crée un nouvel utilisateur
    
    Args:
        username: Nom d'utilisateur unique
        email: Email unique
        password: Mot de passe (sera hashé)
        **kwargs: Champs additionnels (first_name, last_name, etc.)
    
    Returns:
        user_id si succès, None sinon
    """
    try:
        user_doc = {
            "username": username,
            "email": email.lower(),
            "password": hash_password(password),
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "is_active": True,
            "preferences": {
                "currency": "USD",
                "theme": "dark",
                "language": "fr"
            },
            **kwargs
        }
        
        result = users_collection.insert_one(user_doc)
        logger.info(f"✓ Utilisateur créé: {username}")
        return str(result.inserted_id)
        
    except DuplicateKeyError:
        logger.error(f"❌ Utilisateur ou email déjà existant: {username}/{email}")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur création utilisateur: {e}")
        return None

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authentifie un utilisateur
    
    Returns:
        User document si succès, None sinon
    """
    try:
        hashed_password = hash_password(password)
        user = users_collection.find_one({
            "username": username,
            "password": hashed_password,
            "is_active": True
        })
        
        if user:
            # Convertir ObjectId en string pour JSON
            user['_id'] = str(user['_id'])
            logger.info(f"✓ Utilisateur authentifié: {username}")
            return user
        else:
            logger.warning(f"⚠ Authentification échouée: {username}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur authentification: {e}")
        return None

def get_user(user_id: str) -> Optional[Dict]:
    """Récupère un utilisateur par ID"""
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user['_id'] = str(user['_id'])
        return user
    except Exception as e:
        logger.error(f"❌ Erreur récupération utilisateur: {e}")
        return None

def update_user(user_id: str, updates: Dict) -> bool:
    """Met à jour un utilisateur"""
    try:
        updates['updated_at'] = datetime.datetime.utcnow()
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour utilisateur: {e}")
        return False




