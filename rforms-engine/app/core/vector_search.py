from typing import Dict, Any, List, Optional
from loguru import logger
from app.db.qdrant import search_vectors, store_vector
from app.core.config import settings
from app.core.embeddings import create_embedding, create_session_vector as embeddings_create_session_vector
import numpy as np
import json
import time

async def create_session_vector(session_state: Dict[str, Any]) -> List[float]:
    """Create a vector representation of the current session state.
    
    Args:
        session_state: The current session state
        
    Returns:
        A vector representation of the session
    """
    try:
        # Use the proper embedding implementation from embeddings.py
        return await embeddings_create_session_vector(session_state)
    except Exception as e:
        logger.error(f"Error creating session vector with embeddings: {e}")
        # Fall back to the placeholder implementation if there's an error
        return await create_placeholder_session_vector(session_state)

async def create_placeholder_session_vector(session_state: Dict[str, Any]) -> List[float]:
    """Create a placeholder vector representation of the current session state.
    This is used as a fallback when the proper embedding service is unavailable.
    
    Args:
        session_state: The current session state
        
    Returns:
        A vector representation of the session
    """
    # For MVP, we'll use a simple approach that combines:
    # - Previous 2-3 question/answer pairs (70% of signal)
    # - Current metric scores (20% of signal)
    # - Basic user context if available (10% of signal)
    
    # Initialize empty vector of the right dimension
    vector_dim = settings.VECTOR_DIMENSIONS
    session_vector = [0.0] * vector_dim
    
    try:
        # Get recent question history (last 3 questions)
        history = session_state.get("question_history", [])[-3:]
        
        if not history:
            # Random vector with slight bias for empty history
            np.random.seed(int(time.time()) % (2**32 - 1))  # Safe seed value
            return np.random.normal(0.1, 0.2, vector_dim).tolist()
        
        # Create a JSON representation of recent history and scores
        vector_data = {
            "recent_questions": [
                {
                    "question_id": item.get("id", ""),
                    "question_text": item.get("text", ""),
                    "response_text": item.get("response_text", ""),
                    "metric_id": item.get("metric_id", "")
                }
                for item in history
            ],
            "metric_scores": session_state.get("metrics", {}),
            "user_profile": session_state.get("user_profile", {})
        }
        
        # Serialize to string
        data_str = json.dumps(vector_data)
        
        # Generate a deterministic vector from the string
        # This is just a placeholder used as a fallback
        
        # Use a simple hash-based approach
        from hashlib import md5
        
        # Generate a hash and ensure it's within bounds for np.random.seed
        hash_val = int(md5(data_str.encode()).hexdigest(), 16) % (2**32 - 1)
        
        # Use the hash to seed a random number generator
        np.random.seed(hash_val)
        
        # Generate a random vector with the hash seed
        session_vector = np.random.normal(0, 0.1, vector_dim).tolist()
        
        return session_vector
        
    except Exception as e:
        logger.error(f"Error creating placeholder session vector: {e}")
        # Return random vector as fallback with a safe seed
        np.random.seed(int(time.time()) % (2**32 - 1))  # Safe seed value
        return np.random.normal(0, 0.1, vector_dim).tolist()

async def vector_based_question_selection(session_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select next question using vector similarity.
    
    Args:
        session_state: The current session state
        
    Returns:
        The selected question or None if no matching vector
    """
    try:
        # Skip if vector search is disabled
        if not settings.VECTOR_SEARCH_ENABLED:
            return None
        
        # Start timer for performance tracking
        start_time = time.time()
        
        # Create a vector from the session using embeddings
        session_vector = await create_session_vector(session_state)
        
        # Query vector database with reasonable timeout
        results = await search_vectors(
            vector=session_vector,
            limit=3,
            score_threshold=0.85  # High threshold for MVP
        )
        
        # No results
        if not results:
            logger.debug("No vector search results found")
            return None
            
        # Use the top match if it's very confident
        top_result = results[0]
        if top_result.score > 0.92:
            payload = top_result.payload
            
            question = {
                "id": payload.get("question_id", f"vector_{int(time.time())}"),
                "text": payload.get("question_text", ""),
                "type": payload.get("question_type", "text"),
                "metric_id": payload.get("metric_id", ""),
                "options": payload.get("options"),
                "source": "vector_db",
                "information_gain": payload.get("information_gain", 0.5),
                "confidence": top_result.score
            }
            
            # Log performance
            elapsed = time.time() - start_time
            logger.debug(f"Vector search completed in {elapsed:.2f}ms with score {top_result.score:.2f}")
            
            return question
            
        # No high-confidence match
        logger.debug(f"Vector search found results but below confidence threshold (top: {top_result.score:.2f})")
        return None
            
    except Exception as e:
        # Graceful fallback if vector search fails
        logger.error(f"Vector search failed: {str(e)}")
        return None

async def store_successful_path(session_state: Dict[str, Any], question: Dict[str, Any]) -> bool:
    """Store a successful question path in the vector database.
    
    Args:
        session_state: The current session state
        question: The question that was successfully used
        
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        # Skip if disabled
        if not settings.STORE_LLM_RESULTS:
            return False
        
        # Get session vector using embeddings
        session_vector = await create_session_vector(session_state)
        
        # Create payload
        payload = {
            "question_id": question.get("id", f"vector_{int(time.time())}"),
            "question_text": question.get("text", ""),
            "question_type": question.get("type", "text"),
            "metric_id": question.get("metric_id", ""),
            "options": question.get("options"),
            "information_gain": question.get("information_gain", 0.5),
            "usage_count": 1,
            "success_rate": 1.0,
            "stored_at": time.time()
        }
        
        # Store in vector database
        result_id = await store_vector(
            vector=session_vector,
            payload=payload
        )
        
        return result_id is not None
        
    except Exception as e:
        logger.error(f"Failed to store vector path: {e}")
        return False

async def update_vector_usage(vector_id: str, success: bool) -> bool:
    """Update the usage statistics for a vector.
    
    Args:
        vector_id: The ID of the vector
        success: Whether the question was successful
        
    Returns:
        True if updated successfully, False otherwise
    """
    # This is a placeholder for updating vector usage stats
    # In a real implementation, this would update the vector payload
    
    # For MVP, we'll just log it
    logger.info(f"Vector {vector_id} usage updated: success={success}")
    
    return True 