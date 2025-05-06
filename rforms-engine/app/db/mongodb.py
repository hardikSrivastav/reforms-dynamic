from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from app.core.config import settings
from loguru import logger

# MongoDB client and database instances
mongo_client = None
db = None

async def init_mongodb():
    """Initialize the MongoDB client and database."""
    global mongo_client, db
    
    try:
        mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
        
        # Test connection
        await mongo_client.admin.command('ping')
        
        # Get database
        db = mongo_client[settings.MONGO_DB]
        
        logger.info("MongoDB connection initialized successfully")
        
        # Create indexes for collections
        await create_indexes()
        
    except ConnectionFailure:
        logger.error("MongoDB connection failed")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        raise

async def close_mongodb():
    """Close the MongoDB connection."""
    global mongo_client
    
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

async def create_indexes():
    """Create indexes for MongoDB collections."""
    global db
    
    try:
        # Decision Trees collection
        await db.decision_trees.create_index("metric_id", unique=True)
        
        # Sessions collection
        await db.sessions.create_index("session_id", unique=True)
        await db.sessions.create_index("user_id")
        await db.sessions.create_index("created_at")
        
        # Responses collection
        await db.responses.create_index("session_id")
        await db.responses.create_index([("session_id", 1), ("question_id", 1)])
        
        # Question History collection
        await db.question_history.create_index([("session_id", 1), ("timestamp", 1)])
        
        # Vector Paths collection
        await db.vector_paths.create_index("session_vector", sparse=True)
        
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {e}")
        raise

# Collection accessor functions
def get_decision_trees_collection():
    """Get the decision trees collection."""
    global db
    return db.decision_trees

def get_sessions_collection():
    """Get the sessions collection."""
    global db
    return db.sessions

def get_responses_collection():
    """Get the responses collection."""
    global db
    return db.responses

def get_question_history_collection():
    """Get the question history collection."""
    global db
    return db.question_history

def get_vector_paths_collection():
    """Get the vector paths collection."""
    global db
    return db.vector_paths 