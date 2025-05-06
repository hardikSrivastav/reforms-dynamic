from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings
from loguru import logger
import uuid
from typing import List, Dict, Any, Optional

# Qdrant client
qdrant_client = None

async def init_qdrant():
    """Initialize the Qdrant client and create collections if they don't exist."""
    global qdrant_client
    
    try:
        qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        
        # Check if collections exist
        collections = qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        # Create main questions collection if it doesn't exist
        if settings.QDRANT_COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_DIMENSIONS,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")
        
        # Create question patterns collection if it doesn't exist
        if settings.QDRANT_QUESTION_PATTERNS_COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_QUESTION_PATTERNS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_DIMENSIONS,
                    distance=models.Distance.COSINE
                )
            )
            # Add payload index for category field
            qdrant_client.create_payload_index(
                collection_name=settings.QDRANT_QUESTION_PATTERNS_COLLECTION,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_QUESTION_PATTERNS_COLLECTION}")
        
        logger.info("Qdrant connection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant connection: {e}")
        raise

async def get_qdrant_client():
    """Get the Qdrant client instance."""
    global qdrant_client
    
    if qdrant_client is None:
        # Initialize if not already done
        await init_qdrant()
    
    return qdrant_client

async def search(
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: float = 0.7,
    query_filter: Optional[Dict[str, Any]] = None
):
    """Search for vectors in a collection."""
    client = await get_qdrant_client()
    
    try:
        # Perform search - QdrantClient methods are synchronous, don't use await
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
    except Exception as e:
        logger.error(f"Error searching vectors: {str(e)}")
        return []

async def close_qdrant():
    """Close the Qdrant client."""
    global qdrant_client
    
    if qdrant_client is not None:
        # No explicit close method, just set to None
        qdrant_client = None
        logger.info("Qdrant connection closed")

async def store_vector(vector, payload, collection_name=None):
    """Store a vector in the Qdrant collection.
    
    Args:
        vector: The vector to store
        payload: The payload associated with the vector
        collection_name: The collection name (default from settings)
        
    Returns:
        The ID of the stored vector
    """
    global qdrant_client
    
    if qdrant_client is None:
        logger.warning("Qdrant client not initialized, skipping vector storage")
        return None
    
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        # Generate a UUID string if no ID is provided
        vector_id = payload.get("id")
        if vector_id is None:
            vector_id = str(uuid.uuid4())
            logger.info(f"Generated new vector ID: {vector_id}")
            payload["id"] = vector_id
        
        # Try to convert to int if it's a numeric string
        point_id = None
        try:
            # First try as an integer ID
            if isinstance(vector_id, str) and vector_id.isdigit():
                point_id = int(vector_id)
                logger.debug(f"Using integer ID: {point_id}")
            else:
                # Otherwise use as string ID
                point_id = vector_id
                logger.debug(f"Using string ID: {point_id}")
        except (ValueError, TypeError):
            # Fallback to string ID
            point_id = str(vector_id)
            logger.debug(f"Falling back to string ID: {point_id}")
        
        # Create the point with the determined ID - QdrantClient methods are synchronous, don't use await
        result = qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        # Just return the ID we used since UpdateResult may not have upserted_ids
        return point_id
    except Exception as e:
        logger.error(f"Failed to store vector: {e}")
        # Don't fail the entire operation if vector storage fails
        return payload.get("id")

async def search_vectors(vector, limit=5, score_threshold=0.7, collection_name=None):
    """Search for similar vectors in the Qdrant collection.
    
    Args:
        vector: The query vector
        limit: The maximum number of results to return
        score_threshold: The minimum similarity score (0-1)
        collection_name: The collection name (default from settings)
        
    Returns:
        A list of search results
    """
    global qdrant_client
    
    if qdrant_client is None:
        logger.warning("Qdrant client not initialized, skipping vector search")
        return []
    
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        # QdrantClient methods are synchronous, don't use await
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return search_result
    except UnexpectedResponse as e:
        if "not found" in str(e):
            logger.warning(f"Collection {collection_name} not found for vector search")
            return []
        else:
            logger.error(f"Failed to search vectors: {e}")
            return []
    except Exception as e:
        logger.error(f"Failed to search vectors: {e}")
        return []

async def delete_vector(id, collection_name=None):
    """Delete a vector from the Qdrant collection.
    
    Args:
        id: The ID of the vector to delete
        collection_name: The collection name (default from settings)
        
    Returns:
        True if successful, False otherwise
    """
    global qdrant_client
    
    if qdrant_client is None:
        logger.warning("Qdrant client not initialized, skipping vector deletion")
        return False
    
    if collection_name is None:
        collection_name = settings.QDRANT_COLLECTION
    
    try:
        # QdrantClient methods are synchronous, don't use await
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[id]
            )
        )
        
        return True
    except Exception as e:
        logger.error(f"Failed to delete vector: {e}")
        return False 