"""
config.py - NEW FILE
Centralized configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Financial Constants
TREASURY_BILL_RATE = float(os.getenv('TREASURY_BILL_RATE', 4.5))
TRADING_DAYS_PER_YEAR = 252

# Database
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DATABASE_NAME = 'financeai'

# App Settings
DEFAULT_PORTFOLIO_AMOUNT = 10000.0
DEFAULT_RISK_TOLERANCE = 20.0
