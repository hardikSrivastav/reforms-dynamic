import os
import logging
import json
from typing import List, Optional, Dict, Any
import numpy as np

import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

async def create_embedding(
    text: str,
    model: Optional[str] = None
) -> List[float]:
    """Create an embedding vector for a text string.
    
    Args:
        text: The text to create an embedding for
        model: The model to use (defaults to settings.EMBEDDING_MODEL)
        
    Returns:
        A list of floats representing the embedding vector
    """
    # Use the model from settings if not specified
    model = model or settings.EMBEDDING_MODEL
    
    try:
        # Determine the embedding provider based on the model name
        if "openai" in model or "text-embedding" in model:
            embedding = await create_openai_embedding(text, model)
            
            # Ensure embedding has the correct dimension (settings.VECTOR_DIMENSIONS)
            embedding = resize_embedding(embedding, settings.VECTOR_DIMENSIONS)
            return embedding
        else:
            # Default to a dummy embedding for now
            # In a production system, you would implement other providers
            # like Azure, Cohere, or local models
            logger.warning(f"Unsupported embedding model: {model}, using dummy embedding")
            return create_dummy_embedding()
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        # Return a dummy embedding in case of error
        return create_dummy_embedding()

def resize_embedding(embedding: List[float], target_dim: int) -> List[float]:
    """Resize an embedding vector to the target dimension.
    
    Args:
        embedding: The original embedding vector
        target_dim: The target dimension
        
    Returns:
        Resized embedding vector
    """
    orig_dim = len(embedding)
    
    # Log the dimensions for debugging
    logger.info(f"Resizing embedding from {orig_dim} to {target_dim} dimensions")
    
    if orig_dim == target_dim:
        return embedding
    
    # Convert to numpy array for easier manipulation
    embedding_array = np.array(embedding)
    
    if orig_dim > target_dim:
        # Dimensionality reduction: use PCA-like approach by taking first target_dim components
        # This is a simple approach - in production, you might want a proper PCA implementation
        return embedding_array[:target_dim].tolist()
    else:
        # Padding: repeat the vector or add zeros
        # We'll use repetition for better semantic preservation
        repeats = int(np.ceil(target_dim / orig_dim))
        repeated = np.tile(embedding_array, repeats)
        return repeated[:target_dim].tolist()

async def create_openai_embedding(
    text: str,
    model: str
) -> List[float]:
    """Create an embedding using OpenAI's embedding API.
    
    Args:
        text: The text to create an embedding for
        model: The OpenAI model to use
        
    Returns:
        A list of floats representing the embedding vector
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": text
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")
                
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            # Log the original dimension for debugging
            logger.debug(f"OpenAI embedding has dimension: {len(embedding)}")
            
            return embedding
    except Exception as e:
        logger.error(f"OpenAI embedding error: {str(e)}")
        # Reraise to let the caller handle it
        raise

def create_dummy_embedding() -> List[float]:
    """Create a dummy embedding vector for testing or fallback.
    
    Returns:
        A list of zeros with the correct dimensionality
    """
    return [0.0] * settings.VECTOR_DIMENSIONS

async def create_session_vector(session_state: Dict[str, Any]) -> List[float]:
    """Create a vector representation of the current session state.
    
    Args:
        session_state: The current session state
        
    Returns:
        A list of floats representing the session vector
    """
    try:
        # Format recent conversation
        recent_qa = format_recent_conversation(session_state)
        
        # Get metric states
        metrics_state = format_metrics_status(session_state)
        
        # Combine into a single text
        combined_text = f"{recent_qa}\n\nMETRICS STATUS:\n{metrics_state}"
        
        # Create embedding and ensure it has the correct dimensions
        embedding = await create_embedding(combined_text)
        
        # Double-check the dimension
        if len(embedding) != settings.VECTOR_DIMENSIONS:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.VECTOR_DIMENSIONS}")
            embedding = resize_embedding(embedding, settings.VECTOR_DIMENSIONS)
        
        return embedding
    except Exception as e:
        logger.error(f"Error creating session vector: {str(e)}")
        # Return a dummy vector in case of error
        return create_dummy_embedding()

def format_recent_conversation(session_state: Dict[str, Any], max_turns: int = 3) -> str:
    """Format the most recent conversation turns.
    
    Args:
        session_state: The session state
        max_turns: Maximum number of turns to include
        
    Returns:
        Formatted recent conversation
    """
    history = session_state.get("question_history", [])
    # Get the most recent turns
    recent = history[-max_turns:] if len(history) > max_turns else history
    
    formatted = []
    for item in recent:
        formatted.append(f"Q: {item.get('text', '')}")
        formatted.append(f"A: {item.get('response_text', '')}")
    
    return "\n".join(formatted)

def format_metrics_status(session_state: Dict[str, Any]) -> str:
    """Format the metrics status.
    
    Args:
        session_state: The session state
        
    Returns:
        Formatted metrics status
    """
    metrics = session_state.get("metrics", {})
    formatted = []
    
    for metric_id, metric_data in metrics.items():
        formatted.append(
            f"{metric_id}: Score={metric_data.get('score', 0):.2f}, "
            f"Confidence={metric_data.get('confidence', 0):.2f}"
        )
    
    return "\n".join(formatted) 