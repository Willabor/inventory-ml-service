"""Database connection utilities."""
import psycopg2
import pandas as pd
from typing import Optional
from config import settings


class DatabaseConnection:
    """Manage database connections for ML service."""

    def __init__(self):
        self.connection_string = settings.database_url

    def get_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(self.connection_string)

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an UPDATE/INSERT query and return affected rows."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected
        finally:
            conn.close()


# Global database instance
db = DatabaseConnection()
