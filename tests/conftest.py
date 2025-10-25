import pytest
from pymongo import MongoClient
from database import init_database
import os

@pytest.fixture(scope="session")
def mongo_client():
    """Client MongoDB pour les tests"""
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(uri)
    yield client
    client.close()

@pytest.fixture(scope="function")
def test_db(mongo_client):
    """Base de données de test (nettoyée après chaque test)"""
    db = mongo_client.pymanager_test
    
    # Setup
    init_database()
    
    yield db
    
    # Teardown
    for collection in db.list_collection_names():
        db[collection].delete_many({})

@pytest.fixture
def test_user(test_db):
    """Utilisateur de test"""
    from database import create_user
    user_id = create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123"
    )
    return user_id
