from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from app.core.config import settings
from app.core.metrics import get_preferred_question_type
import json
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Import LLM clients based on available API keys
llm_clients = {}

if settings.ANTHROPIC_API_KEY:
    try:
        import anthropic
        llm_clients["anthropic"] = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        logger.info("Anthropic API client initialized")
    except ImportError:
        logger.warning("Anthropic package not installed, but API key provided")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")

if settings.OPENAI_API_KEY:
    try:
        import openai
        llm_clients["openai"] = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAI API client initialized")
    except ImportError:
        logger.warning("OpenAI package not installed, but API key provided")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")

async def format_recent_conversation(session_state: Dict[str, Any], max_turns: int = 3) -> str:
    """Format recent conversation history for LLM context.
    
    Args:
        session_state: The current session state
        max_turns: Maximum number of turns to include
        
    Returns:
        Formatted conversation history
    """
    history = session_state.get("question_history", [])[-max_turns:]
    
    if not history:
        return "No previous conversation."
    
    formatted = []
    for item in history:
        question = item.get("text", "")
        response = item.get("response_text", "")
        
        formatted.append(f"Question: {question}")
        formatted.append(f"User: {response}")
    
    return "\n".join(formatted)

async def create_llm_prompt(session_state: Dict[str, Any], target_metric: str) -> str:
    """Create a prompt for the LLM.
    
    Args:
        session_state: The current session state
        target_metric: The target metric
        
    Returns:
        The prompt for the LLM
    """
    from app.core.metrics import get_metric_details, get_unresolved_metrics_names
    
    # Get metric details and properly await it
    metric_details = await get_metric_details(target_metric)
    metric_name = metric_details.get("name", target_metric.replace("_", " ").title()) if metric_details else target_metric.replace("_", " ").title()
    
    # For MVP, use a simple prompt template
    # In production, this would be more sophisticated
    prompt = f"""
You are an expert survey designer helping create the next best question for a user.

Context:
- We are conducting a survey about: {session_state.get("survey_topic", "user preferences")}
- We need to assess these metrics: {', '.join(get_unresolved_metrics_names(session_state))}
- Current focus metric: {metric_name}
- The conversation so far:
{await format_recent_conversation(session_state)}

Task:
Generate exactly ONE follow-up question that will best help us assess the {metric_name} metric.

Requirements:
1. The question must be concise (max 20 words)
2. The question must be conversational in tone
3. The question must relate directly to {metric_name}
4. The question should avoid repeating information we already know
5. The question type should be {get_preferred_question_type(target_metric)}

Output format:
{{
  "question": "Your question text here?",
  "expected_information_gain": 0.X,  // 0.0-1.0 scale
  "rationale": "Brief explanation of why this question is valuable"
}}
"""
    return prompt

@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
async def query_anthropic(prompt: str, timeout: float = 5.0) -> str:
    """Query the Anthropic API.
    
    Args:
        prompt: The prompt to send
        timeout: Timeout in seconds
        
    Returns:
        The model's response
    """
    if "anthropic" not in llm_clients:
        raise ValueError("Anthropic client not initialized")
    
    client = llm_clients["anthropic"]
    
    try:
        # Set up the request with a timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=500,
                temperature=0.2,
                system="You are an expert survey designer, skilled at creating effective questions.",
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout
        )
        
        return response.content[0].text
        
    except asyncio.TimeoutError:
        logger.warning(f"Anthropic API request timed out after {timeout}s")
        raise TimeoutError(f"Anthropic API request timed out after {timeout}s")
    except Exception as e:
        logger.error(f"Error querying Anthropic API: {e}")
        raise

@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
async def query_openai(prompt: str, timeout: float = 5.0) -> str:
    """Query the OpenAI API.
    
    Args:
        prompt: The prompt to send
        timeout: Timeout in seconds
        
    Returns:
        The model's response
    """
    if "openai" not in llm_clients:
        raise ValueError("OpenAI client not initialized")
    
    client = llm_clients["openai"]
    
    try:
        # Set up the request with a timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4-turbo",
                max_tokens=500,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are an expert survey designer, skilled at creating effective questions."},
                    {"role": "user", "content": prompt}
                ]
            ),
            timeout=timeout
        )
        
        return response.choices[0].message.content
        
    except asyncio.TimeoutError:
        logger.warning(f"OpenAI API request timed out after {timeout}s")
        raise TimeoutError(f"OpenAI API request timed out after {timeout}s")
    except Exception as e:
        logger.error(f"Error querying OpenAI API: {e}")
        raise

async def query_llm(prompt: str, timeout: float = 5.0) -> Optional[str]:
    """Query an LLM API based on available clients.
    
    Args:
        prompt: The prompt to send
        timeout: Timeout in seconds
        
    Returns:
        The model's response or None on failure
    """
    # Try Anthropic first if available
    if "anthropic" in llm_clients:
        try:
            return await query_anthropic(prompt, timeout)
        except Exception as e:
            logger.warning(f"Anthropic query failed, falling back to OpenAI: {e}")
    
    # Try OpenAI if available
    if "openai" in llm_clients:
        try:
            return await query_openai(prompt, timeout)
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            return None
    
    # No clients available
    logger.error("No LLM clients available")
    return None

def validate_llm_response(response: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate that LLM response meets quality standards.
    
    Args:
        response: The LLM's response
        
    Returns:
        Tuple of (is_valid, parsed_response/error_message)
    """
    if not response:
        return False, {"error": "Empty response"}
    
    try:
        # Extract JSON from response (in case there's surrounding text)
        import re
        json_match = re.search(r'```json\s*({.+?})\s*```|({.+})', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            parsed = json.loads(json_str)
        else:
            # Try loading the entire response as JSON
            parsed = json.loads(response)
        
        # Check required fields
        if "question" not in parsed:
            return False, {"error": "Missing question field"}
            
        # Check question quality
        question = parsed["question"]
        if len(question.split()) > 30:  # Allow slight buffer over requirement
            return False, {"error": "Question too long"}
            
        # All checks passed
        return True, parsed
        
    except json.JSONDecodeError:
        # Try to extract just the question if JSON parsing fails
        question_match = re.search(r'"question"\s*:\s*"([^"]+)"', response)
        if question_match:
            question = question_match.group(1)
            return True, {"question": question, "expected_information_gain": 0.5}
        
        return False, {"error": "Invalid JSON format"}

async def llm_question_selection(session_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the next question using LLM generation.
    
    Args:
        session_state: The current session state
        
    Returns:
        The selected question or None
    """
    from app.core.metrics import select_target_metric_for_llm
    
    # Skip if LLM fallback is disabled
    if not settings.LLM_FALLBACK_ENABLED:
        return None
    
    # Check if we have any LLM clients
    if not llm_clients:
        logger.warning("No LLM clients available, skipping LLM question selection")
        return None
    
    try:
        # Start timer for performance tracking
        start_time = time.time()
        
        # Identify the highest priority metric with lowest confidence
        target_metric = await select_target_metric_for_llm(session_state)
        
        if not target_metric:
            logger.warning("No target metric for LLM question selection")
            return None
        
        # Create prompt for LLM (note the await here)
        prompt = await create_llm_prompt(session_state, target_metric)
        
        # Query LLM with timeout and retry logic
        llm_result = await query_llm(prompt, timeout=5.0)
        
        if not llm_result:
            logger.warning("LLM returned no result")
            return None
        
        # Validate and process the LLM response
        is_valid, processed_result = validate_llm_response(llm_result)
        
        if not is_valid:
            error = processed_result.get("error", "Unknown validation error")
            logger.warning(f"LLM validation failed: {error}")
            return None
        
        # Create question object
        question = {
            "id": f"llm_{int(time.time())}",
            "text": processed_result["question"],
            "type": get_preferred_question_type(target_metric),
            "metric_id": target_metric,
            "source": "llm",
            "information_gain": processed_result.get("expected_information_gain", 0.5)
        }
        
        # Store successful LLM generations for future reference
        try:
            if settings.STORE_LLM_RESULTS:
                from app.core.vector_search import store_successful_path
                # Make sure we don't await the result if it's not awaitable
                storage_result = store_successful_path(session_state, question)
                if hasattr(storage_result, "__await__"):
                    await storage_result
                else:
                    logger.debug("store_successful_path returned non-awaitable result")
        except Exception as e:
            logger.error(f"Failed to store successful path: {e}")
            # Continue anyway - this is not critical
            
        # Log performance
        elapsed = time.time() - start_time
        logger.info(f"LLM question generated in {elapsed:.2f}s")
        
        return question
            
    except Exception as e:
        logger.error(f"Error in LLM question selection: {e}")
        return None

def get_fallback_question(metric_id: str) -> Dict[str, Any]:
    """Get a fallback question if all else fails.
    
    Args:
        metric_id: The metric ID
        
    Returns:
        A fallback question
    """
    # Simple library of generic fallback questions per metric
    fallbacks = {
        "interest_level": "How interested are you in this program?",
        "perceived_value": "What do you think would be most valuable about this program?",
        "relevance_to_goals": "How does this program align with your future goals?",
        "program_benefits": "Which aspects of the program sound most beneficial to you?",
        "likelihood_to_recommend": "Would you recommend a program like this to your friends?",
        "engagement_level": "How engaged would you want to be in program activities?",
        "awareness": "Had you heard about our program before this conversation?"
    }
    
    question_text = fallbacks.get(
        metric_id, 
        f"What aspects of {metric_id.replace('_', ' ')} interest you most?"
    )
    
    return {
        "id": f"fallback_{int(time.time())}",
        "text": question_text,
        "type": get_preferred_question_type(metric_id),
        "metric_id": metric_id,
        "source": "fallback_library"
    } 