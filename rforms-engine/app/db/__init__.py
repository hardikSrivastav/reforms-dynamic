from app.db.postgres import init_postgres, close_postgres
from app.db.mongodb import init_mongodb, close_mongodb
from app.db.redis import init_redis, close_redis
from app.db.qdrant import init_qdrant, close_qdrant
from loguru import logger

async def init_db():
    """Initialize all database connections."""
    logger.info("Initializing database connections...")
    try:
        await init_postgres()
        await init_mongodb()
        await init_qdrant()
        try:
            await init_redis()
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            logger.warning("Continuing without Redis - some features may be unavailable")
        logger.info("All database connections initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connections: {e}")
        raise

async def close_db():
    """Close all database connections."""
    logger.info("Closing database connections...")
    try:
        await close_postgres()
        await close_mongodb()
        await close_qdrant()
        try:
            await close_redis()
        except Exception as e:
            logger.warning(f"Error while closing Redis connection: {e}")
        logger.info("All database connections closed successfully")
    except Exception as e:
        logger.error(f"Failed to close database connections: {e}")
        # Don't raise here to ensure all close attempts run 