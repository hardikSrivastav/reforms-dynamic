from typing import Dict, Any, List, Optional
from loguru import logger
from app.db.mongodb import get_sessions_collection, get_question_history_collection, get_responses_collection
from app.db.redis import get_cache, set_cache, delete_cache, get_redis
from app.models.question import SessionState, Response, Question
from app.core.scoring import update_metric_score
from app.core.decision_tree import rule_based_question_selection, check_tree_shortcuts
from app.core.vector_search import vector_based_question_selection, store_successful_path
from app.services.llm import llm_question_selection, get_fallback_question, query_llm
from app.core.metrics import prioritize_metrics_for_questioning, is_metric_complete
from app.core.config import settings
from app.core.advanced_questioning import advanced_question_engine, process_question_feedback
import json
import uuid
import time
from datetime import datetime
import asyncio

# Temporary experiment flag - set to True to use only LLM-based question generation
LLM_ONLY_EXPERIMENT = False  # Set to False to revert to normal behavior

async def create_session(survey_id: str, user_id: Optional[str] = None, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new session for a survey.
    
    Args:
        survey_id: The ID of the survey
        user_id: Optional user ID (for compatibility with API)
        user_profile: Optional user profile data
        
    Returns:
        New session state
    """
    try:
        # Get survey information
        from app.api.endpoints.survey import get_survey
        from app.core.metrics import get_metrics_for_survey
        
        survey = await get_survey(survey_id)
        if not survey:
            logger.error(f"Survey {survey_id} not found")
            raise ValueError(f"Survey {survey_id} not found")
    
        # Get metrics for the survey
        metrics = await get_metrics_for_survey(survey_id)
        if not metrics:
            logger.error(f"No metrics found for survey {survey_id}")
            raise ValueError(f"No metrics found for survey {survey_id}")
            
        # Create metrics state
        metrics_state = {}
        metrics_pending = []
        
        for metric_id, metric_data in metrics.items():
            metrics_state[metric_id] = {
                "score": 0.0,
                "confidence": 0.0,
                "questions_asked": 0,
                "questions_limit": metric_data.get("max_questions", 5)
            }
            metrics_pending.append(metric_id)
            
        # Generate session ID
        session_id = str(uuid.uuid4())
    
        # Create session state with enhanced survey context
        session_state = {
            "session_id": session_id,
            "survey_id": survey_id,
            "user_id": user_id,  # Store user_id if provided
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "question_history": [],
            "metrics": metrics_state,
            "metrics_pending": metrics_pending,
            "user_profile": user_profile or {},
            # Add survey context for better question generation
            "survey_context": {
                "title": survey.get("title", ""),
                "description": survey.get("description", ""),
                "purpose": survey.get("purpose", ""),
                "metrics_metadata": {
                    metric_id: {
                        "name": metric_data.get("name", metric_id),
                        "description": metric_data.get("description", ""),
                        "importance": metric_data.get("importance", 1.0),
                        "min_questions": metric_data.get("min_questions", 1),
                        "max_questions": metric_data.get("max_questions", 5)
                    }
                    for metric_id, metric_data in metrics.items()
                }
            }
        }
        
        # Store session in redis
        redis = await get_redis()
        await redis.set(
            f"session:{session_id}", 
            json.dumps(session_state),
            ex=60*60*24  # Expire after 24 hours
        )
        
        return session_state
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise

async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session state.
    
    Args:
        session_id: The session ID
        
    Returns:
        The session state or None if not found
    """
    # Try to get from cache first
    cache_key = f"session:{session_id}"
    cached_session = await get_cache(cache_key)
    if cached_session:
        return cached_session
    
    # If not in cache, get from database
    sessions_collection = get_sessions_collection()
    session = await sessions_collection.find_one({"session_id": session_id})
    
    if session:
        # Remove MongoDB _id for serialization
        if "_id" in session:
            del session["_id"]
        
        # Cache for future use (1 hour TTL)
        await set_cache(cache_key, session, 3600)
        
    return session

async def update_session(session_id: str, updates: Dict[str, Any]) -> bool:
    """Update session state."""
    # Get current session state from Redis
    session_state = await get_session(session_id)
    if not session_state:
        logger.warning(f"Session {session_id} not found")
        return False
    
    # Update the session state in memory
    session_state.update(updates)
    session_state["last_updated"] = datetime.utcnow().isoformat()
    
    # Store updated session in Redis
    redis = await get_redis()
    await redis.set(
        f"session:{session_id}", 
        json.dumps(session_state),
        ex=60*60*24  # Maintain 24-hour TTL
    )
    
    # Also update MongoDB for persistence, use upsert=True
    try:
        sessions_collection = get_sessions_collection()
        await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": session_state},
            upsert=True  # Create if not exists
        )
    except Exception as e:
        logger.error(f"MongoDB update failed, but Redis cache maintained: {e}")
    
    return True

async def get_next_question(session_id: str) -> Dict[str, Any]:
    """Get the next question for a session.
    
    Args:
        session_id: The session ID
        
    Returns:
        Next question or None if all metrics are complete
    """
    try:
        # First check if there's a cached next question from a recent submit_response
        redis = await get_redis()
        cached_next_question_key = f"next_question:{session_id}"
        cached_next_question = await redis.get(cached_next_question_key)
        
        if cached_next_question:
            # Use the cached question and remove it from cache
            next_question = json.loads(cached_next_question)
            await redis.delete(cached_next_question_key)
            logger.info(f"Using cached next question for session {session_id} (avoiding redundant LLM call)")
            return next_question
            
        # Get the session
        session_state = await get_session(session_id)
        if not session_state:
            logger.error(f"Session {session_id} not found")
            raise ValueError(f"Session {session_id} not found")
            
        logger.info(f"Getting next question for session {session_id}")
        
        # Check if session is active
        if session_state.get("status") != "active":
            logger.info(f"Session {session_id} is not active (status: {session_state.get('status')})")
            return {
                "session_complete": True,
                "reason": f"Session status is {session_state.get('status')}"
            }
            
        # Update metrics_pending list before checking
        updated_metrics_pending = []
        for metric_id in session_state.get("metrics_pending", []):
            try:
                is_complete = await is_metric_complete(metric_id, session_state)
                if not is_complete:
                    updated_metrics_pending.append(metric_id)
            except Exception as e:
                logger.error(f"Error checking if metric {metric_id} is complete: {e}")
                # Keep the metric in the pending list if there was an error
                updated_metrics_pending.append(metric_id)
        
        session_state["metrics_pending"] = updated_metrics_pending
        
        metrics_pending = session_state.get("metrics_pending", [])
        
        if not metrics_pending:
            await end_session(session_id, "completed")
            return {"session_complete": True}
            
        # Track question selection performance
        start_time = time.time()
        
        # EXPERIMENT: If LLM_ONLY_EXPERIMENT is enabled, skip directly to LLM-based generation
        if LLM_ONLY_EXPERIMENT:
            logger.info(f"LLM-only experiment enabled, using only LLM for question generation for session {session_id}")
            # Get the highest priority pending metric
            highest_priority_metric = _get_highest_priority_pending_metric(session_state)
            
            llm_question = await generate_fallback_question(highest_priority_metric, session_state)
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"LLM-only experiment generated question in {elapsed:.2f}s: {llm_question.get('question_text', '')[:50]}...")
            
            # Save question to session history
            await _save_question_to_history(session_id, llm_question, selection_method="llm_experiment")
            
            # Ensure question has both backend and frontend field names
            _normalize_question_fields(llm_question)
            
            return llm_question
        
        # Try advanced question engine if enabled
        if settings.ADVANCED_QUESTION_ENGINE_ENABLED:
            try:
                logger.info(f"Using advanced question engine for session {session_id}")
                question = await advanced_question_engine(session_state)
                
                if question:
                    # Log performance
                    elapsed = time.time() - start_time
                    logger.info(f"Advanced question engine selected question in {elapsed:.2f}s: {question.get('question_text', '')[:50]}...")
                    
                    # Save question to session history
                    await _save_question_to_history(session_id, question, selection_method="advanced")
                    
                    # Remove internal fields
                    if "selection_method" in question:
                        del question["selection_method"]
                    
                    # Ensure question has both backend and frontend field names
                    _normalize_question_fields(question)
                    
                    return question
            except Exception as e:
                logger.error(f"Advanced question engine failed: {str(e)}")
                # Fall back to other methods
        
        # Try rule-based question selection
        try:
            logger.info(f"Using rule-based question selection for session {session_id}")
            rule_question = await rule_based_question_selection(session_state)
            
            if rule_question:
                # Log performance
                elapsed = time.time() - start_time
                logger.info(f"Rule-based question selection in {elapsed:.2f}s: {rule_question.get('question_text', '')[:50]}...")
                
                # Save question to session history
                await _save_question_to_history(session_id, rule_question, selection_method="rule_based")
                
                # Ensure question has both backend and frontend field names
                _normalize_question_fields(rule_question)
                
                return rule_question
        except Exception as e:
            logger.error(f"Rule-based question selection failed: {str(e)}")
            # Fall back to vector similarity
            
        # Try vector-based question selection
        try:
            logger.info(f"Using vector-based question selection for session {session_id}")
            vector_question = await vector_based_question_selection(session_state)
            
            if vector_question:
                # Log performance
                elapsed = time.time() - start_time
                logger.info(f"Vector-based question selection in {elapsed:.2f}s: {vector_question.get('question_text', '')[:50]}...")
                
                # Save question to session history
                await _save_question_to_history(session_id, vector_question, selection_method="vector_db")
                
                # Ensure question has both backend and frontend field names
                _normalize_question_fields(vector_question)
                
                return vector_question
        except Exception as e:
            logger.error(f"Vector-based question selection failed: {str(e)}")
            # Fall back to LLM
            
        # Finally, use LLM as a fallback
        try:
            logger.info(f"Using LLM fallback for session {session_id}")
            # Get the highest priority pending metric
            highest_priority_metric = _get_highest_priority_pending_metric(session_state)
            
            llm_question = await generate_fallback_question(highest_priority_metric, session_state)
            
            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"LLM fallback generated question in {elapsed:.2f}s: {llm_question.get('question_text', '')[:50]}...")
            
            # Save question to session history
            await _save_question_to_history(session_id, llm_question, selection_method="llm_fallback")
            
            # Ensure question has both backend and frontend field names
            _normalize_question_fields(llm_question)
            
            return llm_question
        except Exception as e:
            logger.error(f"LLM fallback failed: {str(e)}")
            # Use a hardcoded fallback question
        
        # If all else fails, use a hardcoded question
        logger.warning(f"All question selection methods failed, using emergency fallback")
        
        # Get a random pending metric
        if metrics_pending:
            metric_id = metrics_pending[0]
            
            fallback_question = {
                "question_id": f"fallback_{int(time.time())}",
                "question_text": f"Please tell me about your experience with {metric_id.replace('_', ' ')}.",
                "metric_id": metric_id,
                "type": "text"
            }
            
            # Save question to session history
            await _save_question_to_history(session_id, fallback_question, selection_method="emergency_fallback")
            
            # Ensure question has both backend and frontend field names
            _normalize_question_fields(fallback_question)
            
            return fallback_question
        else:
            # This shouldn't happen, but just in case
            return {
                "session_complete": True,
                "reason": "No metrics to assess"
            }
    except Exception as e:
        logger.error(f"Error getting next question: {str(e)}")
        raise

def _normalize_question_fields(question: Dict[str, Any]) -> None:
    """Ensure question has both backend and frontend field names.
    
    Backend uses question_id, question_text while frontend expects id, text.
    This function adds any missing fields to ensure compatibility with both.
    
    Args:
        question: The question dictionary to normalize
    """
    # Map backend field names to frontend field names
    field_mapping = {
        "question_id": "id",
        "question_text": "text",
    }
    
    # Add frontend fields if backend fields exist
    for backend_field, frontend_field in field_mapping.items():
        if backend_field in question and frontend_field not in question:
            question[frontend_field] = question[backend_field]
            
    # Add backend fields if frontend fields exist
    for backend_field, frontend_field in field_mapping.items():
        if frontend_field in question and backend_field not in question:
            question[backend_field] = question[frontend_field]
    
    # Log the normalized question for debugging
    logger.debug(f"Normalized question fields: {question}")

def _get_highest_priority_pending_metric(session_state: Dict[str, Any]) -> str:
    """Get the highest priority pending metric to focus on.
    
    Args:
        session_state: The session state
        
    Returns:
        Highest priority metric ID
    """
    metrics = session_state.get("metrics", {})
    pending = session_state.get("metrics_pending", [])
    
    if not metrics or not pending:
        # Default to first metric or fallback
        return pending[0] if pending else "general"
    
    # Prioritize metrics with low confidence and few questions asked
    sorted_metrics = sorted(
        [(metric_id, data) for metric_id, data in metrics.items() if metric_id in pending],
        key=lambda x: (x[1].get("confidence", 0), x[1].get("questions_asked", 0))
    )
    
    return sorted_metrics[0][0] if sorted_metrics else pending[0]

async def generate_fallback_question(metric_id: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a fallback question using LLM with enhanced prompt engineering.
    
    Args:
        metric_id: The metric to assess
        session_state: The session state
        
    Returns:
        Generated question
    """
    start_time = time.time()
    session_id = session_state.get("session_id", "unknown")
    
    # Create a cache key based on relevant session state
    history_len = len(session_state.get("question_history", []))
    cache_key = f"question_{metric_id}_{history_len}"
    
    # Add hash of last response to make cache key more specific
    if history_len > 0:
        last_response = session_state["question_history"][-1].get("response_text", "")
        # Use a simple hash of the response
        import hashlib
        response_hash = hashlib.md5(last_response.encode()).hexdigest()[:8]
        cache_key += f"_{response_hash}"
    
    logger.debug(f"Question cache key: {cache_key}")
    
    # Try to get from cache first
    redis = await get_redis()
    cached_question = await redis.get(cache_key)
    if cached_question:
        cached_result = json.loads(cached_question)
        logger.info(f"Retrieved cached question in {time.time() - start_time:.3f}s")
        return cached_result
    
    # Check if we have precomputed context
    precomputed_context = None
    precomputed_context_data = await redis.get(f"question_context:{session_id}")
    if precomputed_context_data:
        try:
            precomputed_context = json.loads(precomputed_context_data)
            context_age = time.time() - precomputed_context.get("timestamp", 0)
            logger.debug(f"Found precomputed context (age: {context_age:.1f}s)")
        except Exception as e:
            logger.warning(f"Error parsing precomputed context: {e}")
            precomputed_context = None
    
    # Log if we're generating a fresh question
    logger.info(f"Generating fresh question for metric {metric_id}")
    
    try:
        context_start_time = time.time()
        
        # Extract survey context for better question generation
        survey_context = session_state.get("survey_context", {})
        metric_metadata = survey_context.get("metrics_metadata", {}).get(metric_id, {})
        
        # Extract metrics status for better context - use precomputed if available
        if precomputed_context:
            metrics_status = precomputed_context.get("metrics_status", {})
            metric_data = metrics_status.get(metric_id, {})
        else:
            metrics = session_state.get("metrics", {})
            metric_data = metrics.get(metric_id, {})
            
        current_confidence = metric_data.get("confidence", 0)
        questions_asked = metric_data.get("questions_asked", 0)
        
        # Get min/max questions from metadata or use defaults
        min_questions = metric_metadata.get("min_questions", 2)
        max_questions = metric_metadata.get("max_questions", 5) 
        
        # Format a readable metric name instead of using raw ID
        metric_name = metric_metadata.get("name", metric_id.replace("_", " ").title())
        metric_description = metric_metadata.get("description", f"Assessment of {metric_id.replace('_', ' ')}")
        
        # Create a summary of the entire survey
        survey_summary = f"""
SURVEY OVERVIEW:
Title: {survey_context.get('title', 'Customer Feedback Survey')}
Purpose: {survey_context.get('purpose', survey_context.get('description', 'To collect customer feedback and improve our product.'))}
Progress: Gathered data on {len(session_state.get('metrics', {}))}/{len(survey_context.get('metrics_metadata', {}))} metrics
Metrics Needing Assessment: {', '.join(session_state.get('metrics_pending', []))}
        """
        
        # Use precomputed verified facts if available, otherwise generate them
        if precomputed_context and "verified_facts" in precomputed_context:
            verified_user_facts = []
            for fact in precomputed_context["verified_facts"]:
                verified_user_facts.append(
                    f"- In response to '{fact['question'][:50]}...', user said: '{fact['response']}'"
                )
            has_meaningful_responses = precomputed_context.get("has_meaningful_responses", False)
        else:
            # Get conversation history
            history = session_state.get("question_history", [])
            
            # Create a list of verified facts the user has actually shared
            verified_user_facts = []
            for item in history:
                response = item.get("response_text", "").strip()
                question = item.get("text", "").strip()
                
                # Only include meaningful responses (not empty, not "don't know", etc.)
                if (response and len(response) > 5 and 
                    response.lower() not in ["don't know", "don't remember", "no", "yes", "not sure", "i don't know"]):
                    
                    verified_user_facts.append(f"- In response to '{question[:50]}...', user said: '{response}'")
            
            has_meaningful_responses = bool(verified_user_facts)
        
        # Create a summary of conversation history with verified facts
        conversation_summary = "VERIFIED USER RESPONSES:\n"
        if verified_user_facts:
            conversation_summary += "\n".join(verified_user_facts)
        else:
            conversation_summary += "No substantive user responses recorded yet.\n"
        
        # Format the raw conversation history (last 3 exchanges) - use precomputed if available
        if precomputed_context and "recent_history" in precomputed_context:
            raw_history = "DETAILED CONVERSATION HISTORY (RECENT EXCHANGES):\n"
            for i, exchange in enumerate(precomputed_context["recent_history"]):
                raw_history += f"Q{i+1}: {exchange.get('question', '')}\n"
                raw_history += f"A{i+1}: {exchange.get('response', '')}\n\n"
        else:
            # Get conversation history
            history = session_state.get("question_history", [])
            
            raw_history = "DETAILED CONVERSATION HISTORY (RECENT EXCHANGES):\n"
            recent_history = history[-3:] if len(history) >= 3 else history
            
            for i, item in enumerate(recent_history):
                question = item.get("text", "")
                response = item.get("response_text", "")
                
                raw_history += f"Q{i+1}: {question}\n"
                raw_history += f"A{i+1}: {response}\n\n"
        
        # Add metrics status
        if precomputed_context and "metrics_status" in precomputed_context:
            metrics_status_str = "METRICS STATUS:\n"
            for m_id, m_data in precomputed_context["metrics_status"].items():
                m_name = survey_context.get("metrics_metadata", {}).get(m_id, {}).get("name", m_id.replace("_", " ").title())
                metrics_status_str += f"{m_name}: Score={m_data.get('score', 0):.2f}, Confidence={m_data.get('confidence', 0):.2f}, Questions Asked={m_data.get('questions_asked', 0)}\n"
        else:
            metrics = session_state.get("metrics", {})
            metrics_status_str = "METRICS STATUS:\n"
            for m_id, m_data in metrics.items():
                m_name = survey_context.get("metrics_metadata", {}).get(m_id, {}).get("name", m_id.replace("_", " ").title())
                metrics_status_str += f"{m_name}: Score={m_data.get('score', 0):.2f}, Confidence={m_data.get('confidence', 0):.2f}, Questions Asked={m_data.get('questions_asked', 0)}\n"
        
        context_prep_time = time.time() - context_start_time
        logger.debug(f"Context preparation took {context_prep_time:.3f}s")
        
        # Create the enhanced prompt with all the pieces
        prompt = f"""
You are an expert survey designer tasked with generating a highly effective question to collect maximum data in minimum time.

{survey_summary}

TARGET METRIC: {metric_name}
METRIC DESCRIPTION: {metric_description}
METRIC PROGRESS: Confidence {current_confidence:.2f}/1.0, Questions {questions_asked}/{max_questions}

{conversation_summary}

{metrics_status_str}

{raw_history}

IMPORTANT INSTRUCTION ON REFERENCING PREVIOUS RESPONSES:
{'Based on the conversation history, the user has NOT yet provided any substantial information. DO NOT refer to specific details the user has shared, as they have not shared any.' if not has_meaningful_responses else conversation_summary}

Your task is to generate ONE effective question that:
1. Is HIGHLY SPECIFIC and DETAILED about the {metric_name} metric
2. {'DOES NOT REFERENCE specific user details, as no substantial information has been shared yet' if not has_meaningful_responses else 'BUILDS ON VERIFIED USER RESPONSES listed above - do not reference information the user has not explicitly shared'}
3. Is CONVERSATIONAL and ENGAGING - natural and easy to answer
4. Is EFFICIENT - designed to maximize information gain with minimal questions

{'FIRST-TIME QUESTION EXAMPLES (since user has not provided substantial responses):' if not has_meaningful_responses else 'FOLLOW-UP QUESTION EXAMPLES:'}
BAD EXAMPLE: {'"Tell me more about the code integrator you mentioned"' if not has_meaningful_responses else '"You mentioned liking our interface, can you elaborate?"'} (Don't reference details the user hasn't explicitly shared)
GOOD EXAMPLE: {'"When comparing our product to alternatives you have used, which specific features provide the most value relative to the price?"' if not has_meaningful_responses else '"Based on your comment that [verified detail], what specific aspects of our product justify its price point compared to alternatives?"'}

IMPORTANT: The question should be direct and focused on gathering the most information about {metric_name} in a single question. {'DO NOT REFERENCE information the user has not explicitly provided.' if not has_meaningful_responses else ''}

SYSTEM CONTEXT: You are an expert survey designer focused on efficiently gathering maximum information with minimal questions. Your goal is to collect high-quality data about {metric_name}.

QUESTION:
"""
        
        # Log the complete prompt for debugging
        logger.info("=========== FULL LLM PROMPT START ===========")
        logger.info(prompt)
        logger.info("=========== FULL LLM PROMPT END ===========")
        
        # Measure LLM query time
        llm_start_time = time.time()
        
        # Use the LLM to generate a question - only pass params that query_llm accepts
        response = await query_llm(
            prompt=prompt,
            timeout=10.0  # Increased timeout for more reliable responses
        )
        
        llm_query_time = time.time() - llm_start_time
        logger.info(f"LLM query took {llm_query_time:.3f}s")
        
        # Log the LLM response
        logger.info(f"LLM RESPONSE: {response[:100]}..." if len(response) > 100 else response)
        
        # Validate the LLM response doesn't reference non-existent information
        if not has_meaningful_responses and any(phrase in response.lower() for phrase in ["you mentioned", "you said", "you noted", "you indicated", "you shared", "you told me", "as you described"]):
            logger.warning("LLM generated response with invalid references to non-existent user information. Generating fallback question.")
            # Generate a simple, safe question without references
            fallback_result = {
                "question_id": f"llm_safe_{int(time.time())}",
                "question_text": f"What specific features or aspects of our product do you find most valuable relative to its price?",
                "metric_id": metric_id,
                "type": "text"
            }
            
            # Cache the fallback result too
            await redis.set(cache_key, json.dumps(fallback_result), ex=300)  # 5 minute TTL
            
            total_time = time.time() - start_time
            logger.info(f"Generated fallback question in {total_time:.3f}s (cache miss)")
            
            return fallback_result
        
        # Extract just the question text and clean up
        question_text = response.strip()
        
        # Format the response
        result = {
            "question_id": f"llm_{int(time.time())}",
            "question_text": question_text,
            "metric_id": metric_id,
            "type": "text"
        }
        
        # Cache the result with a short TTL (5 minutes)
        await redis.set(cache_key, json.dumps(result), ex=300)
        
        total_time = time.time() - start_time
        logger.info(f"Generated fresh question in {total_time:.3f}s (cache miss)")
        
        return result
    except Exception as e:
        logger.error(f"Error generating fallback question: {str(e)}")
        # Return a very simple fallback
        fallback = {
            "question_id": f"fallback_{int(time.time())}",
            "question_text": f"Can you share your thoughts about {metric_id.replace('_', ' ')}?",
            "metric_id": metric_id,
            "type": "text"
        }
        
        total_time = time.time() - start_time
        logger.info(f"Generated emergency fallback in {total_time:.3f}s after error")
        
        return fallback

async def submit_response(session_id: str, question_id: str, response_text: str, response_value: Any = None) -> Dict[str, Any]:
    """Submit a response to a question.
    
    Args:
        session_id: The session ID
        question_id: The question ID
        response_text: The text response
        response_value: Optional structured response value
        
    Returns:
        Updated metrics and next question (if available)
    """
    start_time = time.time()
    
    # Get session state
    session_state = await get_session(session_id)
    
    if not session_state:
        logger.error(f"Session {session_id} not found")
        return {"error": "Session not found"}
    
    # Start precomputing context for next question in the background
    # This will run in parallel with the rest of this function
    precompute_task = asyncio.create_task(precompute_next_question_context(session_id))
    
    # Get question from history if it exists
    question_history = session_state.get("question_history", [])
    question = None
    
    # Search through history for the question
    for item in question_history:
        if item.get("id") == question_id:
            question = item
            break
    
    # If question not found in history, look for the most recent question
    # that hasn't been answered yet (likely the current one)
    if not question and question_history:
        # Try to find most recent question that doesn't have a response
        for item in reversed(question_history):
            if not item.get("response_text"):
                question = item
                break
    
    # If still not found, create a placeholder with best available info
    if not question:
        # Determine response type based on question ID prefix
        if question_id.startswith("llm_"):
            source = "llm"
        elif question_id.startswith("vector_"):
            source = "vector_db"
        elif question_id.startswith("adv_"):
            source = "advanced_engine"
        elif question_id.startswith("fallback_"):
            source = "fallback_library"
        else:
            source = "rule_engine"
        
        # Get metric ID from last prioritized metric if not available
        if "metrics_pending" in session_state and session_state["metrics_pending"]:
            metric_id = session_state["metrics_pending"][0]
        else:
            metric_id = "general"
        
        # Attempt to find the question text from in-flight questions
        # (might be stored in a temporary state or cache)
        question_text = "Unknown question"
        try:
            # If Redis has the current question stored, get it
            redis = await get_redis()
            current_question_data = await redis.get(f"current_question:{session_id}")
            if current_question_data:
                current_question = json.loads(current_question_data)
                if current_question.get("question_id") == question_id or current_question.get("id") == question_id:
                    question_text = current_question.get("question_text") or current_question.get("text", "Unknown question")
        except Exception as e:
            logger.error(f"Failed to retrieve current question from Redis: {e}")
        
        # Create a placeholder question object
        question = {
            "id": question_id,
            "text": question_text,
            "metric_id": metric_id,
            "source": source,
            "response_type": "text"  # Default to text
        }
    
    # Determine the response type
    response_type = question.get("type", "text")
    metric_id = question.get("metric_id")
    
    # Store the response
    response_id = str(uuid.uuid4())
    response_obj = {
        "id": response_id,
        "session_id": session_id,
        "question_id": question_id,
        "question_text": question.get("text", ""),  # Store question text with response
        "response_text": response_text,
        "response_value": response_value,
        "response_type": response_type,
        "metric_id": metric_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Track metric state before update for feedback
    metrics_before = {}
    if metric_id and metric_id in session_state.get("metrics", {}):
        metrics_before[metric_id] = {
            "score": session_state["metrics"][metric_id].get("score", 0),
            "confidence": session_state["metrics"][metric_id].get("confidence", 0)
    }
    
    # Calculate scoring updates
    if metric_id and metric_id != "general":
        score_updates = update_metric_score(
            metric=metric_id,
            response=response_text if response_value is None else response_value,
            response_type=response_type,
            session_state=session_state
        )
        
        # Store metric updates
        response_obj["metric_updates"] = score_updates
        
        # Update session metrics
        metrics = session_state.get("metrics", {})
        if metric_id in metrics:
            metrics[metric_id].update(score_updates)
    
    # Get MongoDB collection
    responses_collection = get_responses_collection()
    redis = await get_redis()
    
    # Create tasks list for operations we want to run and await together
    tasks = []
    
    # Start getting next question early (in parallel)
    # This is a proper coroutine, so we can use create_task
    next_question_task = asyncio.create_task(get_next_question(session_id))
    
    # Process recent responses in Redis
    recent_responses = []
    try:
        recent_responses_key = f"recent_responses:{session_id}"
        
        # Get existing recent responses
        recent_responses_data = await redis.get(recent_responses_key)
        
        if recent_responses_data:
            recent_responses = json.loads(recent_responses_data)
            # Limit to last 5 responses
            if len(recent_responses) >= 5:
                recent_responses = recent_responses[-4:]  # Keep last 4 to add new one
        
        # Add this response
        recent_responses.append({
            "question_id": question_id,
            "response_text": response_text,
            "response_value": response_value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Store back in Redis with 1-hour TTL
        # Redis operation returns a coroutine
        tasks.append(redis.set(
            recent_responses_key,
            json.dumps(recent_responses),
            ex=60*60  # 1 hour TTL
        ))
        
        logger.debug(f"Adding Redis cache update to tasks for session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to prepare Redis update for recent responses: {e}")
    
    # Add to question history with complete information
    question_with_response = {
        "id": question_id,
        "text": question.get("text", ""),
        "type": response_type,
        "metric_id": metric_id,
        "response_text": response_text,
        "response_value": response_value,
        "response_id": response_id,
        "source": question.get("source", "unknown"),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log the complete response for debugging
    logger.debug(f"Adding response to history: question='{question_with_response['text'][:50]}...', response='{response_text[:50]}...'")
    
    question_history.append(question_with_response)
    
    # Update session state
    total_questions = session_state.get("total_questions", 0) + 1
    
    # Update session with all changes at once
    update_data = {
        "question_history": question_history,
        "total_questions": total_questions,
        "metrics": session_state.get("metrics", {})  # Update with latest metrics
    }
    
    # Update session returns a coroutine
    tasks.append(update_session(session_id, update_data))
    
    # Prepare metrics update for feedback
    metrics_update = {}
    
    if metric_id and metric_id in session_state.get("metrics", {}):
        metrics_after = session_state["metrics"][metric_id]
        metrics_update[metric_id] = {
            "confidence_before": metrics_before.get(metric_id, {}).get("confidence", 0),
            "confidence_after": metrics_after.get("confidence", 0),
            "score_before": metrics_before.get(metric_id, {}).get("score", 0),
            "score_after": metrics_after.get("score", 0)
        }
    
    # Process feedback for the advanced questioning engine
    if settings.VECTOR_SEARCH_ENABLED and settings.LLM_FALLBACK_ENABLED:
        try:
            # Construct response data
            response_data = {
                "response_text": response_text,
                "response_value": response_value,
                "response_type": response_type
            }
            
            # Process feedback asynchronously - proper coroutine
            feedback_task = asyncio.create_task(
                process_question_feedback(
                    session_id=session_id,
                    question_id=question_id,
                    response=response_data,
                    metrics_update=metrics_update
                )
            )
            # We don't add this to tasks since it's not critical for response
        except Exception as e:
            logger.error(f"Error creating task for processing question feedback: {str(e)}")
    
    # If this was a vector-based question, update its success stats
    if question.get("source") == "vector_db" and "vector_id" in question:
        try:
            from app.core.vector_search import update_vector_usage
            # This is a proper coroutine
            vector_update_task = asyncio.create_task(update_vector_usage(question["vector_id"], True))
            # We don't add this to tasks since it's not critical for response
        except Exception as e:
            logger.error(f"Failed to create task for updating vector usage: {str(e)}")
    
    # Now run database operations in parallel (MongoDB and Redis)
    try:
        # First insert the response - MongoDB operation directly without create_task
        # MongoDB operations from Motor already return Futures
        insert_result = await responses_collection.insert_one(response_obj)
        logger.debug(f"Stored response in MongoDB with ID: {insert_result.inserted_id}")
        
        # Then wait for all other critical tasks to complete
        if tasks:
            await asyncio.gather(*tasks)
            
        logger.debug(f"Database operations completed in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error in database operations: {str(e)}")
    
    # Get next question with error handling
    try:
        next_question = await next_question_task
        logger.debug(f"Next question retrieved in {time.time() - start_time:.3f}s")
        
        # Cache the next question in Redis for 30 seconds
        # This allows a subsequent direct get_next_question call to use this result
        # without generating a new question
        try:
            await redis.set(
                f"next_question:{session_id}",
                json.dumps(next_question),
                ex=30  # 30 second TTL
            )
            logger.debug(f"Cached next question for session {session_id} (30s TTL)")
        except Exception as e:
            logger.warning(f"Failed to cache next question: {e}")
            
    except Exception as e:
        logger.error(f"Error getting next question: {str(e)}")
        # Use an emergency fallback question
        next_question = {
            "id": f"emergency_fallback_{int(time.time())}",
            "text": "Can you tell me more about what you're looking for?",
            "type": "text",
            "metric_id": "general",
            "source": "emergency_fallback"
        }
        # Normalize field names for the emergency fallback question
        _normalize_question_fields(next_question)
    
    # Background tasks (feedback, vector updates, etc.) will continue running
    # We don't need to await them as they're not critical for response
    
    total_time = time.time() - start_time
    logger.info(f"submit_response completed in {total_time:.3f}s")
    
    return {
        "response_id": response_id,
        "metrics": session_state.get("metrics", {}),
        "next_question": next_question
    }

async def end_session(session_id: str, reason: str = "completed") -> Dict[str, Any]:
    """End a session and return final metrics.
    
    Args:
        session_id: The session ID
        reason: Optional reason for ending the session
        
    Returns:
        Final metrics and session summary
    """
    # Get session state
    session_state = await get_session(session_id)
    
    if not session_state:
        logger.error(f"Session {session_id} not found")
        return {"error": "Session not found"}
    
    # Mark session as complete
    await update_session(session_id, {
        "status": "completed",
        "end_time": datetime.utcnow().isoformat(),
        "completion_reason": reason
    })
    
    # Prepare summary
    summary = {
        "session_id": session_id,
        "metrics": session_state.get("metrics", {}),
        "total_questions": session_state.get("total_questions", 0),
        "start_time": session_state.get("start_time"),
        "end_time": datetime.utcnow().isoformat(),
        "duration_seconds": (datetime.utcnow() - datetime.fromisoformat(session_state.get("start_time"))).total_seconds(),
        "completion_reason": reason
    }
    
    logger.info(f"Session {session_id} completed with {summary['total_questions']} questions (reason: {reason})")
    
    return summary

async def process_session_feedback(session_id: str, feedback_data: Dict[str, Any]) -> bool:
    """Process end-of-session feedback to improve the system.
    
    Args:
        session_id: The session ID
        feedback_data: The feedback data
        
    Returns:
        True if successful, False otherwise
    """
    # Get session state
    session_state = await get_session(session_id)
    
    if not session_state:
        logger.error(f"Session {session_id} not found")
        return False
    
    # Store feedback in session
    await update_session(session_id, {"feedback": feedback_data})
    
    # Update question effectiveness scores (would be implemented in a real system)
    # This is a placeholder for now
    question_ratings = feedback_data.get("question_ratings", {})
    for q_id, rating in question_ratings.items():
        logger.info(f"Question {q_id} rated {rating}")
    
    # Analyze path efficiency
    question_count = session_state.get("total_questions", 0)
    data_quality = feedback_data.get("data_quality", 0)
    
    logger.info(f"Session {session_id} efficiency: {question_count} questions, data quality {data_quality}")
    
    return True

def log_performance(source: str, elapsed_time: float):
    """Log performance metrics.
    
    Args:
        source: The source of the question (rule_engine, vector_engine, llm_engine)
        elapsed_time: The elapsed time in seconds
    """
    logger.info(f"{source.upper()} question selection took {elapsed_time*1000:.2f}ms") 

async def _save_question_to_history(session_id: str, question: Dict[str, Any], selection_method: str = "unknown") -> bool:
    """Save a question to the session history.
    
    Args:
        session_id: The session ID
        question: The question object
        selection_method: Method used to select the question
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get session state
        session_state = await get_session(session_id)
        if not session_state:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Convert question format for history storage
        question_record = {
            "id": question.get("question_id", ""),
            "text": question.get("question_text", ""),
            "type": question.get("type", "text"),
            "metric_id": question.get("metric_id", ""),
            "source": question.get("source", selection_method),
            "selection_method": selection_method,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metrics data to track questions asked
        metric_id = question.get("metric_id")
        metrics = session_state.get("metrics", {})
        
        if metric_id and metric_id in metrics:
            metrics[metric_id]["questions_asked"] = metrics[metric_id].get("questions_asked", 0) + 1
        
        # Add to history
        history = session_state.get("question_history", [])
        history.append(question_record)
        
        # Update session
        updates = {
            "question_history": history,
            "metrics": metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        # Store in Redis
        redis = await get_redis()
        session_state.update(updates)
        await redis.set(
            f"session:{session_id}", 
            json.dumps(session_state),
            ex=60*60*24  # Expire after 24 hours
        )
        
        # Also update MongoDB for persistence
        try:
            sessions_collection = get_sessions_collection()
            await sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": updates},
                upsert=True  # Create if not exists
            )
            logger.debug(f"MongoDB session updated after adding question to history")
        except Exception as e:
            logger.error(f"MongoDB update failed when saving question to history, but Redis cache maintained: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving question to history: {e}")
        return False 

# Add new function to precompute context for next question
async def precompute_next_question_context(session_id: str):
    """Precompute common context needed for next question generation.
    
    Args:
        session_id: The session ID
        
    Returns:
        None
    """
    start_time = time.time()
    
    # Get session state
    session_state = await get_session(session_id)
    if not session_state:
        logger.warning(f"Cannot precompute context: Session {session_id} not found")
        return
        
    # Get metrics and history
    metrics = session_state.get("metrics", {})
    history = session_state.get("question_history", [])
    
    # Extract verified facts from conversation history
    verified_facts = []
    for item in history:
        response = item.get("response_text", "").strip()
        question = item.get("text", "").strip()
        
        # Only include meaningful responses
        if (response and len(response) > 5 and 
            response.lower() not in ["don't know", "don't remember", "no", "yes", "not sure", "i don't know"]):
            
            verified_facts.append({
                "question": question[:100],
                "response": response[:200]
            })
    
    # Create metrics status summary
    metrics_status = {}
    for m_id, m_data in metrics.items():
        metrics_status[m_id] = {
            "score": m_data.get("score", 0),
            "confidence": m_data.get("confidence", 0),
            "questions_asked": m_data.get("questions_asked", 0)
        }
    
    # Package context data
    context = {
        "verified_facts": verified_facts,
        "metrics_status": metrics_status,
        "has_meaningful_responses": bool(verified_facts),
        "recent_history": [
            {
                "question": item.get("text", "")[:150],
                "response": item.get("response_text", "")[:150]
            } 
            for item in history[-3:] if item.get("text")
        ],
        "timestamp": time.time()
    }
    
    # Store in Redis with short TTL (1 minute)
    redis = await get_redis()
    await redis.set(f"question_context:{session_id}", json.dumps(context), ex=60)
    
    elapsed = time.time() - start_time
    logger.info(f"Precomputed question context for session {session_id} in {elapsed:.3f}s") 