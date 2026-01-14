"""
AeroRisk - Predictive Safety Risk Analytics Platform
Database Connection Module
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database configuration settings."""
    
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "aerorisk")
        self.user = os.getenv("POSTGRES_USER", "aerorisk")
        self.password = os.getenv("POSTGRES_PASSWORD", "aerorisk_secure_password_2024")
        
        # Construct URL if not provided directly
        self.url = os.getenv(
            "DATABASE_URL",
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
    
    @property
    def async_url(self) -> str:
        """Get async database URL for asyncpg."""
        return self.url.replace("postgresql://", "postgresql+asyncpg://")


# Global configuration instance
db_config = DatabaseConfig()

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    db_config.url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    echo=os.getenv("ENVIRONMENT", "development") == "development",
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    Yields a session and ensures it's closed after use.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    
    Usage:
        with get_db_session() as db:
            db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()


def verify_connection() -> bool:
    """
    Verify database connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            logger.success("✅ Database connection verified successfully!")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get information about the database.
    
    Returns:
        dict: Database information including version, tables, etc.
    """
    try:
        with engine.connect() as conn:
            # Get PostgreSQL version
            version_result = conn.execute(text("SELECT version()"))
            version = version_result.fetchone()[0]
            
            # Get table count
            tables_result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            table_count = tables_result.fetchone()[0]
            
            # Get database size
            size_result = conn.execute(text(f"""
                SELECT pg_size_pretty(pg_database_size('{db_config.database}'))
            """))
            db_size = size_result.fetchone()[0]
            
            return {
                "status": "connected",
                "version": version,
                "database": db_config.database,
                "host": db_config.host,
                "port": db_config.port,
                "table_count": table_count,
                "database_size": db_size,
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    # Test connection when running directly
    print("🔌 Testing database connection...")
    if verify_connection():
        info = get_database_info()
        print(f"📊 Database Info: {info}")
    else:
        print("❌ Could not connect to database. Is PostgreSQL running?")
