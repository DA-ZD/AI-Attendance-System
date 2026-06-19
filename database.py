# database.py
# Standalone MongoDB helper functions.
# The main DB connection lives in config.py.
# This module re-exports the key objects for convenience.

from config import client, db, DB_CONNECTED, get_db

__all__ = ["client", "db", "DB_CONNECTED", "get_db"]
