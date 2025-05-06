import redis.asyncio as redis
from typing import Any, Optional
from app.core.config import settings
from loguru import logger
import json
from bson import ObjectId
import datetime

# Global Redis client
redis_client = None
redis_available = False

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles MongoDB ObjectId and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(JSONEncoder, self).default(obj)

async def init_redis():
    """Initialize Redis connection"""
    global redis_client, redis_available
    
    logger.info(f"Initializing Redis connection to {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    try:
        # Create Redis client
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password="redis123",  # Use fixed password for development
            db=settings.REDIS_DB,
            socket_timeout=5,      # Add timeout to avoid long blocking
            socket_connect_timeout=5,
            retry_on_timeout=True,
            max_connections=10,
            decode_responses=True
        )
        
        # Test connection
        await redis_client.ping()
        redis_available = True
        logger.info("Redis connection established successfully")
    except Exception as e:
        redis_available = False
        redis_client = None
        logger.error(f"Failed to initialize Redis connection: {e}")
        raise

async def close_redis():
    """Close Redis connection"""
    global redis_client, redis_available
    
    if redis_client is not None:
        logger.info("Closing Redis connection")
        await redis_client.close()
        redis_client = None
        redis_available = False

async def get_redis():
    """Get Redis client"""
    if redis_client is None or not redis_available:
        raise RuntimeError("Redis connection not available")
    return redis_client

async def get_cache(key: str) -> Optional[Any]:
    """Get a value from the cache."""
    global redis_client, redis_available
    
    if not redis_available or redis_client is None:
        return None
    
    try:
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        logger.error(f"Failed to get cache for key {key}: {e}")
        return None

async def set_cache(key: str, value: Any, ttl: int = 60) -> bool:
    """Set a value in the cache."""
    global redis_client, redis_available
    
    if not redis_available or redis_client is None:
        return False
    
    try:
        if value is None:
            # Delete the key if value is None
            await redis_client.delete(key)
            return True
            
        # Use custom JSONEncoder for MongoDB ObjectId
        json_data = json.dumps(value, cls=JSONEncoder)
        await redis_client.set(key, json_data, ex=ttl)
        return True
    except Exception as e:
        logger.error(f"Failed to set cache for key {key}: {e}")
        return False

async def delete_cache(key: str) -> bool:
    """Delete a value from the cache."""
    global redis_client, redis_available
    
    if not redis_available or redis_client is None:
        return False
    
    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Failed to delete cache for key {key}: {e}")
        return False

async def invalidate_pattern(pattern: str) -> bool:
    """Invalidate all keys matching a pattern.
    
    Args:
        pattern: The pattern to match
        
    Returns:
        True if successful, False otherwise
    """
    global redis_client
    
    try:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break
        return True
    except Exception as e:
        logger.error(f"Failed to invalidate pattern {pattern}: {e}")
        return False

# Cache decorator with fault tolerance
def cached(key_prefix: str, expiry: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Skip caching if Redis is not available
            if not redis_available or redis_client is None:
                return await func(*args, **kwargs)
                
            # Generate cache key from function name, args and kwargs
            key = f"{key_prefix}:{func.__name__}:"
            key += ":".join(str(arg) for arg in args)
            key += ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            try:
                # Try to get from cache
                cached_value = await get_cache(key)
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = await func(*args, **kwargs)
                await set_cache(key, result, expiry)
                return result
            except Exception as e:
                # If caching fails, just run the function
                logger.warning(f"Cache operation failed: {e}, executing function directly")
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator 