import os

API_URL = os.environ.get("API_URL", "http://api:8000")
ITEMS_URL = os.environ.get("ITEMS_URL", "mongodb://user:password@localhost:27017/lf?authSource=admin")