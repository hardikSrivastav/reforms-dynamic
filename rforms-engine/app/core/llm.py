import os
import logging
import json
from typing import Optional, Dict, Any
import hashlib
import time

import httpx
from app.core.config import settings
from app.db.redis import cached, get_redis

logger = logging.getLogger(__name__)

# Implement a more efficient caching mechanism for LLM responses
async def get_llm_cache_key(prompt: str, model: str, temperature: float, response_format: Optional[str], system_prompt: Optional[str]) -> str:
    """Generate a consistent cache key for LLM queries."""
    # Create a deterministic hash from the parameters that affect the response
    # Strip whitespace to normalize cache keys
    normalized_prompt = prompt.strip()
    normalized_system = (system_prompt or "").strip()
    
    # Create a string with all parameters that affect the response
    cache_string = f"{normalized_prompt}|{model}|{temperature}|{response_format or ''}|{normalized_system}"
    
    # Generate a hash for the cache key
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
    
    return f"llm:{cache_hash}"

# Cache LLM queries for 10 minutes (600 seconds) - adjust TTL based on your needs
@cached(key_prefix="llm_response", expiry=600)
async def cached_query_llm(
    prompt: str, 
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[str],
    system_prompt: Optional[str]
) -> str:
    """Cached version of query_llm to reduce redundant API calls."""
    # Generate a debug ID for logging
    debug_id = hashlib.md5(prompt[:50].encode()).hexdigest()[:8]
    logger.info(f"LLM Query [{debug_id}] cache miss - querying API with model {model}")
    
    # The actual API call logic
    start_time = time.time()
    result = await _query_llm_with_fallback(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        system_prompt=system_prompt
    )
    elapsed = time.time() - start_time
    logger.info(f"LLM Query [{debug_id}] completed in {elapsed:.2f}s")
    
    return result

async def query_llm(
    prompt: str, 
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 500,
    response_format: Optional[str] = None,
    timeout: float = 30.0,
    system_prompt: Optional[str] = None
) -> str:
    """Query an LLM with the given prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (defaults to settings.LLM_MODEL)
        temperature: The temperature to use (0.0-1.0)
        max_tokens: Maximum number of tokens to generate
        response_format: Optional format for response (e.g. "json")
        timeout: Timeout in seconds
        system_prompt: Optional custom system prompt to override default
        
    Returns:
        The LLM's response as a string
    """
    # Use the model from settings if not specified
    model = model or settings.LLM_MODEL
    
    # First try to get from cache
    try:
        # Check if this is a cacheable query
        if temperature < 0.3:  # Only cache deterministic or near-deterministic results
            cache_key = await get_llm_cache_key(prompt, model, temperature, response_format, system_prompt)
            redis = await get_redis()
            cached_response = await redis.get(cache_key)
            
            if cached_response:
                debug_id = hashlib.md5(prompt[:50].encode()).hexdigest()[:8]
                logger.info(f"LLM Query [{debug_id}] cache hit")
                return cached_response
            
            # Use cached function which handles TTL and serialization
            return await cached_query_llm(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                system_prompt=system_prompt
            )
    except Exception as e:
        logger.warning(f"Cache lookup failed: {str(e)} - falling back to direct query")
    
    # Fall back to direct API call without caching
    return await _query_llm_with_fallback(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        system_prompt=system_prompt
    )

async def _query_llm_with_fallback(
    prompt: str, 
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> str:
    """Internal implementation of LLM query with provider fallback."""
    try:
        # Verify that we have at least one API key available
        has_anthropic = settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your_anthropic_api_key_here"
        has_openai = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here" and settings.OPENAI_API_KEY != "your_openai_key_here"
        
        if not has_anthropic and not has_openai:
            raise Exception("No valid API keys configured for any LLM provider")
        
        # Shorter timeout to reduce latency impact of failed requests
        request_timeout = 15.0
        
        # Determine the LLM provider based on settings or model name
        if "claude" in model.lower():
            if not has_anthropic:
                logger.warning("Claude model requested but no Anthropic API key found. Falling back to OpenAI.")
                # Force an OpenAI model if we have that key
                if has_openai:
                    model = "gpt-3.5-turbo"
                    return await query_openai(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        timeout=request_timeout,
                        system_prompt=system_prompt
                    )
                else:
                    raise Exception("No valid API keys configured for any LLM provider")
                
            return await query_anthropic(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=request_timeout,
                system_prompt=system_prompt
            )
        else:
            # Default to OpenAI
            if not has_openai:
                logger.warning("OpenAI model requested but no OpenAI API key found. Falling back to Anthropic.")
                # Force an Anthropic model if we have that key
                if has_anthropic:
                    model = "claude-3-sonnet-20240229"
                    return await query_anthropic(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        timeout=request_timeout,
                        system_prompt=system_prompt
                    )
                else:
                    raise Exception("No valid API keys configured for any LLM provider")
                
            return await query_openai(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=request_timeout,
                system_prompt=system_prompt
            )
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        # Return a fallback response in case of error
        if response_format == "json":
            return json.dumps({
                "error": "Failed to query LLM",
                "recommended_approach": "fallback",
                "target_metric": "general"
            })
        return "I'm unable to provide a response at this time."

async def query_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[str],
    timeout: float,
    system_prompt: Optional[str] = None
) -> str:
    """Query Anthropic's Claude with the given prompt.
    
    Args:
        prompt: The prompt to send to Claude
        model: The model to use
        temperature: The temperature to use
        max_tokens: Maximum number of tokens to generate
        response_format: Optional format for response
        timeout: Timeout in seconds
        system_prompt: Optional custom system prompt to override default
        
    Returns:
        Claude's response as a string
    """
    # Check if API key is available
    if not settings.ANTHROPIC_API_KEY:
        raise Exception("ANTHROPIC_API_KEY not configured")
        
    headers = {
        "x-api-key": settings.ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Format the system prompt with custom content or default
    default_system = "You are a helpful AI assistant that helps generate questions."
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = default_system
    
    # For JSON responses, add instructions to the system prompt
    if response_format == "json":
        system_content += " Always format your response as a valid, properly structured JSON object."
        # If the prompt doesn't already contain json instructions, add them
        if "json" not in prompt.lower():
            prompt += "\n\nRespond with a valid JSON object that follows the structure described above."
    
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_content,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Add response format if specified
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    
    try:
        # Use a single client with HTTP/2 support for connection pooling
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Anthropic API error: {response.text}")
                raise Exception(f"Anthropic API error: {response.status_code}")
                
            result = response.json()
            return result["content"][0]["text"]
    except Exception as e:
        # Add more context to Anthropic errors
        logger.error(f"Error querying Anthropic: {str(e)}")
        if "401" in str(e):
            raise Exception(f"Anthropic authentication error (invalid API key)")
        raise

async def query_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[str],
    timeout: float,
    system_prompt: Optional[str] = None
) -> str:
    """Query OpenAI with the given prompt.
    
    Args:
        prompt: The prompt to send to OpenAI
        model: The model to use
        temperature: The temperature to use
        max_tokens: Maximum number of tokens to generate
        response_format: Optional format for response
        timeout: Timeout in seconds
        system_prompt: Optional custom system prompt to override default
        
    Returns:
        OpenAI's response as a string
    """
    # Check if API key is available
    if not settings.OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not configured")
        
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Determine system prompt - add JSON instruction if needed
    default_system = "You are a helpful AI assistant that helps generate questions."
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = default_system
    
    # OpenAI requires the word 'json' in the prompt when using json_object format
    if response_format == "json":
        system_content += " Format your response as a valid JSON object."
        # If the prompt doesn't contain the word 'json', add it
        if "json" not in prompt.lower():
            prompt += "\n\nRespond with a valid JSON object."
    
    # Format system prompt and user message
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Add response format if specified
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    
    try:
        # Use a single client with HTTP/2 support for connection pooling
        async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")
                
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error querying OpenAI: {str(e)}")
        raise 