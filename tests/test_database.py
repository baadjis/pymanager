import pytest
from database import create_user, authenticate_user, save_portfolio

def test_create_user(test_db):
    user_id = create_user("testuser", "test@test.com", "pass123")
    assert user_id is not None

def test_auth(test_db):
    create_user("user1", "user1@test.com", "pass123")
    user = authenticate_user("user1", "pass123")
    assert user is not None
    assert user['username'] == "user1"

# Ajoutez plus de tests...
