import pytest
from database import (
    create_user, authenticate_user,
    save_portfolio, get_portfolios,
    add_to_watchlist, get_watchlist,
    create_alert, get_alerts
)

def test_create_user(test_db):
    """Test création d'utilisateur"""
    user_id = create_user(
        username="newuser",
        email="new@example.com",
        password="pass123"
    )
    assert user_id is not None

def test_authenticate_user(test_db, test_user):
    """Test authentification"""
    user = authenticate_user("testuser", "testpass123")
    assert user is not None
    assert user['username'] == "testuser"

def test_save_portfolio(test_db, test_user):
    """Test sauvegarde portfolio"""
    # Mock portfolio object
    class MockPortfolio:
        assets = ["AAPL", "MSFT"]
        weights = [0.5, 0.5]
        data = {"Adj Close": {0: [150, 300]}}
    
    portfolio = MockPortfolio()
    portfolio_id = save_portfolio(
        user_id=test_user,
        portfolio=portfolio,
        name="Test Portfolio",
        amount=10000,
        method="risk"
    )
    
    assert portfolio_id is not None

def test_get_portfolios(test_db, test_user):
    """Test récupération portfolios"""
    portfolios = get_portfolios(user_id=test_user)
    assert isinstance(portfolios, list)

def test_watchlist(test_db, test_user):
    """Test watchlist"""
    # Ajouter
    item_id = add_to_watchlist(
        user_id=test_user,
        ticker="AAPL",
        notes="Test note"
    )
    assert item_id is not None
    
    # Récupérer
    watchlist = get_watchlist(user_id=test_user)
    assert len(watchlist) > 0
    assert watchlist[0]['ticker'] == "AAPL"

def test_alerts(test_db, test_user):
    """Test alertes"""
    # Créer
    alert_id = create_alert(
        user_id=test_user,
        ticker="AAPL",
        alert_type="price_above",
        threshold=200
    )
    assert alert_id is not None
    
    # Récupérer
    alerts = get_alerts(user_id=test_user)
    assert len(alerts) > 0
