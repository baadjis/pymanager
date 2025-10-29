#!/usr/bin/python3
"""
PyManager Database - Users Management
Support pour licensing/subscription
"""

import datetime
from pymongo.errors import DuplicateKeyError
try:
    from database.database import logger, users_collection
except:
    from .database import logger, users_collection
from typing import Dict, List, Optional
from bson import ObjectId
import hashlib
import re

# =============================================================================
# Password Hashing
# =============================================================================

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

# =============================================================================
# Users Management
# =============================================================================

def create_user(username: str, email: str, password: str, **kwargs) -> Optional[str]:
    """
    Crée un nouvel utilisateur avec structure complète
    
    Args:
        username: Nom d'utilisateur unique
        email: Email unique
        password: Mot de passe (sera hashé)
        **kwargs: first_name, last_name, license_type, etc.
    
    Returns:
        user_id si succès, None sinon
    """
    try:
        # Vérifier si email est .edu/.ac.* pour student auto-detection
        license_type = kwargs.get('license_type', 'free')
        if is_educational_email(email) and license_type == 'free':
            license_type = 'student'
            logger.info(f"📧 Email éducatif détecté: {email} → licence student")
        
        user_doc = {
            # Auth
            "username": username,
            "email": email.lower(),
            "password": hash_password(password),
            
            # Profile
            "profile": {
                "first_name": kwargs.get('first_name', ''),
                "last_name": kwargs.get('last_name', ''),
                "company": kwargs.get('company', ''),
                "role": kwargs.get('role', ''),
                "phone": kwargs.get('phone', ''),
                "timezone": kwargs.get('timezone', 'UTC'),
                "avatar_url": kwargs.get('avatar_url')
            },
            
            # Subscription & License
            "license_type": license_type,  # free, student, academic, individual, professional, institutional
            "subscription": {
                "tier": license_type,
                "status": "trial" if license_type in ['individual', 'professional'] else "active",
                "started_at": datetime.datetime.utcnow(),
                "expires_at": None,  # None = permanent, date = expiration
                "trial_ends_at": datetime.datetime.utcnow() + datetime.timedelta(days=30) if license_type == 'individual' else None,
                "auto_renew": False,
                "billing_cycle": None,  # monthly, yearly, biennial, lifetime
                "payment_method": None,
                "last_payment": None,
                "next_payment": None
            },
            
            # Verification (pour student/academic)
            "verification": {
                "email_verified": False,
                "student_verified": is_educational_email(email),  # Auto si .edu
                "academic_verified": False,
                "verification_method": "email_edu" if is_educational_email(email) else None,
                "verified_at": datetime.datetime.utcnow() if is_educational_email(email) else None,
                "expires_at": datetime.datetime.utcnow() + datetime.timedelta(days=365) if is_educational_email(email) else None
            },
            
            # Usage Tracking
            "usage": {
                "portfolios_count": 0,
                "ai_queries_today": 0,
                "ai_queries_date": datetime.datetime.utcnow().date().isoformat(),
                "market_explorer_today": 0,
                "market_explorer_date": datetime.datetime.utcnow().date().isoformat(),
                "last_ai_query": None,
                "last_login": None,
                "total_sessions": 0
            },
            
            # Preferences
            "preferences": {
                "currency": kwargs.get('currency', 'USD'),
                "theme": kwargs.get('theme', 'dark'),
                "language": kwargs.get('language', 'fr'),
                "notifications": {
                    "email": True,
                    "push": False,
                    "sms": False,
                    "alerts": True,
                    "newsletter": True,
                    "marketing": kwargs.get('marketing_emails', False)
                }
            },
            
            # Status
            "is_active": True,
            "is_verified": is_educational_email(email),
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "last_login": None
        }
        
        result = users_collection.insert_one(user_doc)
        logger.info(f"✓ User créé: {username} (license: {license_type})")
        
        # Envoyer email de vérification si nécessaire
        if license_type == 'student' and is_educational_email(email):
            logger.info(f"📧 Licence student activée automatiquement pour {email}")
        
        return str(result.inserted_id)
        
    except DuplicateKeyError:
        logger.error(f"❌ Username/email déjà existant: {username}/{email}")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur création user: {e}")
        return None

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Authentifie un utilisateur
    Met à jour last_login et usage stats
    
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
            # Convertir ObjectId en string
            user['_id'] = str(user['_id'])
            
            # Mettre à jour last_login et session count
            users_collection.update_one(
                {"_id": ObjectId(user['_id'])},
                {
                    "$set": {
                        "last_login": datetime.datetime.utcnow(),
                        "usage.last_login": datetime.datetime.utcnow()
                    },
                    "$inc": {"usage.total_sessions": 1}
                }
            )
            
            # Check subscription expiration
            if user.get('subscription', {}).get('expires_at'):
                expires_at = user['subscription']['expires_at']
                if datetime.datetime.utcnow() > expires_at:
                    logger.warning(f"⚠️  Subscription expirée pour {username}")
                    users_collection.update_one(
                        {"_id": ObjectId(user['_id'])},
                        {"$set": {"subscription.status": "expired"}}
                    )
                    user['subscription']['status'] = 'expired'
            
            logger.info(f"✓ User authentifié: {username}")
            return user
        else:
            logger.warning(f"⚠️  Auth échouée: {username}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erreur auth: {e}")
        return None

def get_user(user_id: str) -> Optional[Dict]:
    """Récupère un user par ID"""
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user['_id'] = str(user['_id'])
        return user
    except Exception as e:
        logger.error(f"❌ Erreur get user: {e}")
        return None

def get_user_by_email(email: str) -> Optional[Dict]:
    """Récupère un user par email"""
    try:
        user = users_collection.find_one({"email": email.lower()})
        if user:
            user['_id'] = str(user['_id'])
        return user
    except Exception as e:
        logger.error(f"❌ Erreur get user by email: {e}")
        return None

def update_user(user_id: str, updates: Dict) -> bool:
    """Met à jour un user"""
    try:
        updates['updated_at'] = datetime.datetime.utcnow()
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Erreur update user: {e}")
        return False

# =============================================================================
# Subscription Management
# =============================================================================

def upgrade_subscription(user_id: str, new_tier: str, billing_cycle: str = 'monthly', **kwargs) -> bool:
    """
    Upgrade subscription d'un user
    
    Args:
        user_id: ID du user
        new_tier: individual, professional, institutional
        billing_cycle: monthly, yearly, biennial, lifetime
        **kwargs: trial_days, auto_renew, etc.
    """
    try:
        duration_days = {
            'monthly': 30,
            'yearly': 365,
            'biennial': 730,
            'lifetime': 36500  # 100 ans
        }.get(billing_cycle, 30)
        
        expires_at = None if billing_cycle == 'lifetime' else datetime.datetime.utcnow() + datetime.timedelta(days=duration_days)
        
        updates = {
            "license_type": new_tier,
            "subscription.tier": new_tier,
            "subscription.status": "active",
            "subscription.billing_cycle": billing_cycle,
            "subscription.started_at": datetime.datetime.utcnow(),
            "subscription.expires_at": expires_at,
            "subscription.auto_renew": kwargs.get('auto_renew', True),
            "subscription.last_payment": datetime.datetime.utcnow(),
            "subscription.next_payment": expires_at if billing_cycle != 'lifetime' else None
        }
        
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        
        logger.info(f"✓ Subscription upgraded: user {user_id} → {new_tier} ({billing_cycle})")
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur upgrade subscription: {e}")
        return False

def cancel_subscription(user_id: str, immediate: bool = False) -> bool:
    """
    Annule une subscription
    
    Args:
        immediate: Si True, désactive immédiatement. Sinon, à expiration
    """
    try:
        updates = {
            "subscription.auto_renew": False,
            "subscription.status": "cancelled" if immediate else "active",
            "updated_at": datetime.datetime.utcnow()
        }
        
        if immediate:
            updates["subscription.expires_at"] = datetime.datetime.utcnow()
            updates["license_type"] = "free"
            updates["subscription.tier"] = "free"
        
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        
        logger.info(f"✓ Subscription cancelled: user {user_id} (immediate: {immediate})")
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur cancel subscription: {e}")
        return False

# =============================================================================
# Usage Tracking
# =============================================================================

def increment_usage(user_id: str, resource: str) -> bool:
    """
    Incrémente l'usage d'une ressource
    
    Args:
        resource: ai_queries, market_explorer, etc.
    """
    try:
        today = datetime.datetime.utcnow().date().isoformat()
        
        # Reset counter si nouvelle journée
        user = get_user(user_id)
        if user:
            usage_date_field = f"usage.{resource}_date"
            if user.get('usage', {}).get(f"{resource}_date") != today:
                users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {
                        "$set": {
                            f"usage.{resource}_today": 0,
                            usage_date_field: today
                        }
                    }
                )
        
        # Incrémenter
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$inc": {f"usage.{resource}_today": 1},
                "$set": {f"usage.last_{resource}": datetime.datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur increment usage: {e}")
        return False

def check_usage_limit(user_id: str, resource: str, limit: int) -> tuple[bool, int, int]:
    """
    Vérifie si l'usage dépasse la limite
    
    Returns:
        (can_use, current_usage, limit)
    """
    try:
        user = get_user(user_id)
        if not user:
            return False, 0, limit
        
        today = datetime.datetime.utcnow().date().isoformat()
        usage_date = user.get('usage', {}).get(f"{resource}_date")
        
        # Reset si nouvelle journée
        if usage_date != today:
            current = 0
        else:
            current = user.get('usage', {}).get(f"{resource}_today", 0)
        
        can_use = current < limit
        return can_use, current, limit
        
    except Exception as e:
        logger.error(f"❌ Erreur check usage: {e}")
        return False, 0, limit

# =============================================================================
# Verification (Student/Academic)
# =============================================================================

def is_educational_email(email: str) -> bool:
    """Vérifie si email est éducatif (.edu, .ac.*)"""
    email_lower = email.lower()
    
    edu_domains = [
        '.edu', '.ac.uk', '.ac.fr', '.ac.be', '.ac.ca',
        '.edu.au', '.edu.fr', '.edu.sg',
        'univ-', 'university', '@etudiant', '@student'
    ]
    
    return any(domain in email_lower for domain in edu_domains)

def verify_student(user_id: str, verification_method: str = 'email_edu', **kwargs) -> bool:
    """
    Vérifie et active licence student
    
    Args:
        verification_method: email_edu, github_student, manual
        **kwargs: github_username, document_url, etc.
    """
    try:
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=365)  # 1 an
        
        updates = {
            "license_type": "student",
            "subscription.tier": "student",
            "subscription.status": "active",
            "subscription.expires_at": expires_at,
            "verification.student_verified": True,
            "verification.verification_method": verification_method,
            "verification.verified_at": datetime.datetime.utcnow(),
            "verification.expires_at": expires_at,
            "is_verified": True,
            **{f"verification.{k}": v for k, v in kwargs.items()}
        }
        
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        
        logger.info(f"✓ Student vérifié: user {user_id} via {verification_method}")
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur verify student: {e}")
        return False

def verify_academic(user_id: str, institution: str, **kwargs) -> bool:
    """Vérifie et active licence academic"""
    try:
        updates = {
            "license_type": "academic",
            "subscription.tier": "academic",
            "subscription.status": "active",
            "subscription.expires_at": None,  # Permanent pour academic
            "verification.academic_verified": True,
            "verification.verification_method": "institution_email",
            "verification.verified_at": datetime.datetime.utcnow(),
            "verification.institution": institution,
            "is_verified": True
        }
        
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        
        logger.info(f"✓ Academic vérifié: user {user_id} ({institution})")
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"❌ Erreur verify academic: {e}")
        return False

# =============================================================================
# Stats
# =============================================================================

def get_user_stats() -> Dict:
    """Statistiques globales des users"""
    try:
        total = users_collection.count_documents({})
        active = users_collection.count_documents({"is_active": True})
        
        # Par license type
        by_license = {}
        for license_type in ['free', 'student', 'academic', 'individual', 'professional', 'institutional']:
            count = users_collection.count_documents({"license_type": license_type})
            if count > 0:
                by_license[license_type] = count
        
        # Par subscription status
        by_status = {}
        for status in ['active', 'trial', 'expired', 'cancelled']:
            count = users_collection.count_documents({"subscription.status": status})
            if count > 0:
                by_status[status] = count
        
        return {
            "total": total,
            "active": active,
            "by_license": by_license,
            "by_status": by_status
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur get stats: {e}")
        return {}
