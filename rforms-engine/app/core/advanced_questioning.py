import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid
import re  # Add import for regular expressions
import asyncio  # For parallel operations
import time  # For performance tracking
import hashlib

# Import database clients
from app.db.qdrant import get_qdrant_client
from app.db.redis import get_redis, cached
from app.core.llm import query_llm
from app.core.embeddings import create_session_vector
from app.core.config import settings

# Change to using loguru for more detailed logging
from loguru import logger

# Cache expensive session vector creation with shorter TTL (5 minutes)
@cached(key_prefix="session_vector", expiry=300)
async def cached_create_session_vector(session_state: Dict[str, Any]) -> List[float]:
    """Cached wrapper for create_session_vector to improve performance."""
    # Generate a cache key based on relevant parts of session state that affect the vector
    history_len = len(session_state.get("question_history", []))
    session_id = session_state.get("session_id", "unknown")
    
    logger.debug(f"[SESSION VECTOR] Starting vector creation for session {session_id}")
    logger.debug(f"[SESSION VECTOR] History length: {history_len}, Metrics count: {len(session_state.get('metrics', {}))}")
    
    start_time = time.time()
    vector = await create_session_vector(session_state)
    elapsed = time.time() - start_time
    
    logger.debug(f"[SESSION VECTOR] Vector created in {elapsed:.3f}s with dimension: {len(vector)}")
    logger.debug(f"[SESSION VECTOR] Vector sample (first 5 elements): {vector[:5]}")
    
    return vector

async def advanced_question_engine(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Integrated question selection system that intelligently combines all approaches.
    
    Args:
        session_state: Current session state with history and metrics
        
    Returns:
        The next optimal question with metadata
    """
    try:
        # Start timing for performance tracking
        start_time = time.time()
        session_id = session_state.get("session_id", "unknown")
        
        logger.info(f"[ADV_ENGINE] Starting advanced question engine for session {session_id}")
        
        # Get fresh session state directly from Redis to avoid using stale data
        try:
            redis = await get_redis()
            fresh_session_data = await redis.get(f"session:{session_id}")
            if fresh_session_data:
                fresh_session = json.loads(fresh_session_data)
                # Only use the fresh session if it has history and it's not older
                if (fresh_session.get("question_history", []) and 
                    len(fresh_session.get("question_history", [])) >= len(session_state.get("question_history", []))):
                    logger.debug(f"[ADV_ENGINE] Using fresh session data from Redis with {len(fresh_session.get('question_history', []))} history items")
                    session_state = fresh_session
        except Exception as e:
            logger.warning(f"[ADV_ENGINE] Could not get fresh session data: {str(e)}")
        
        # Log the session state structure for debugging
        history_count = len(session_state.get("question_history", []))
        metrics_count = len(session_state.get("metrics", {}))
        pending_metrics = session_state.get("metrics_pending", [])
        
        logger.debug(f"[ADV_ENGINE] Session state summary: {history_count} questions, {metrics_count} metrics, {len(pending_metrics)} pending metrics")
        logger.debug(f"[ADV_ENGINE] Pending metrics: {pending_metrics}")
        
        # Log the most recent question and response if available
        if history_count > 0:
            last_item = session_state.get("question_history", [])[-1]
            logger.debug(f"[ADV_ENGINE] Last question: '{last_item.get('text', 'N/A')}'")
            logger.debug(f"[ADV_ENGINE] Last response: '{last_item.get('response_text', 'N/A')[:100]}...'")
        
        # Determine if the last response was meaningful enough to build upon
        meaningful_response = True
        last_response = ""
        
        # First check history for a response
        if history_count > 0:
            last_response = session_state.get("question_history", [])[-1].get("response_text", "")
        
        # If no meaningful response in history, check Redis for any in-flight responses
        # This handles race conditions where responses are being processed but not yet in history
        if not last_response or len(last_response) < 10:
            try:
                # Look for recent responses in Redis
                redis = await get_redis()
                recent_responses_key = f"recent_responses:{session_id}"
                recent_responses_data = await redis.get(recent_responses_key)
                
                if recent_responses_data:
                    recent_responses = json.loads(recent_responses_data)
                    # Get the most recent response if available
                    if recent_responses and isinstance(recent_responses, list) and len(recent_responses) > 0:
                        recent_response = recent_responses[-1].get("response_text", "")
                        if recent_response and len(recent_response) >= 10:
                            logger.debug(f"[ADV_ENGINE] Found in-flight response in Redis: '{recent_response[:50]}...'")
                            last_response = recent_response
                            # Since we found a meaningful response, override the flag
                            meaningful_response = True
            except Exception as e:
                logger.warning(f"[ADV_ENGINE] Error checking for in-flight responses: {str(e)}")
        
        # Check if the response is too short or generic
        if len(last_response) < 10 or last_response.lower() in ["yes", "no", "sure", "ok", "maybe", "idk", "i don't know"]:
            logger.info(f"[ADV_ENGINE] Last response too brief to build upon: '{last_response}'")
            meaningful_response = False
        
        logger.info(f"[ADV_ENGINE] Session analysis: {history_count} questions asked, {metrics_count} metrics tracked, {len(pending_metrics)} pending metrics, meaningful last response: {meaningful_response}")
        
        # 1. First analyze the session context with enhanced detail
        try:
            # Use cached vector creation to speed up repeated calls
            session_id = session_state.get("session_id", "")
            cache_key = f"vector_{session_id}_{history_count}"
            
            logger.debug(f"[ADV_ENGINE] Attempting to get session vector with cache key: {cache_key}")
            
            # Try to get from cache first
            redis = await get_redis()
            cached_vector = await redis.get(cache_key)
            
            session_vector = None
            vector_source = "unknown"
            
            if cached_vector:
                logger.debug(f"[ADV_ENGINE] Cache hit! Using cached session vector from Redis")
                session_vector = json.loads(cached_vector)
                vector_source = "redis_cache"
            else:
                logger.debug(f"[ADV_ENGINE] Cache miss. Creating new session vector")
                # Use the cached function which will handle internal caching logic
                start_vector_time = time.time()
                session_vector = await cached_create_session_vector(session_state)
                vector_time = time.time() - start_vector_time
                logger.debug(f"[ADV_ENGINE] Vector creation took {vector_time:.3f}s")
                
                # Cache the vector for 5 minutes (short TTL as sessions evolve)
                await redis.set(cache_key, json.dumps(session_vector), ex=300)
                vector_source = "newly_created"
                
            logger.info(f"[ADV_ENGINE] Session vector ready ({vector_source}) with dimension: {len(session_vector)}")
        except Exception as e:
            logger.error(f"[ADV_ENGINE] Session vector creation failed: {str(e)}")
            # Create a fallback vector if embeddings fail
            session_vector = await create_placeholder_session_vector(session_state)
            logger.info(f"[ADV_ENGINE] Using placeholder session vector due to embedding error, dimension: {len(session_vector)}")
            vector_source = "placeholder"
        
        # Log the vector characteristics
        logger.debug(f"[ADV_ENGINE] Vector statistics: source={vector_source}, dimension={len(session_vector)}, sample={session_vector[:3]}...")
        
        # List of truly generic question patterns that should be avoided
        # Only detect obvious boilerplate patterns, not legitimate follow-up questions
        generic_patterns = [
            r"^tell me more about",
            r"^(can|could) you (please )?(tell|share|explain)",
            r"^how (do|would) you feel about",
            r"^what (are|is) your thoughts on",
            r"^(please )?(share|tell me) your (thoughts|opinion|perspective)"
        ]
        
        logger.debug(f"[ADV_ENGINE] Using {len(generic_patterns)} patterns to detect generic questions")
        
        # Set up parallel operations for better performance
        vector_matches = []
        
        # Check if we should perform vector search (minor optimization)
        should_search_vectors = history_count >= 1 and meaningful_response
        
        logger.info(f"[ADV_ENGINE] Vector search decision: should_search_vectors={should_search_vectors} (history_count={history_count}, meaningful_response={meaningful_response})")
        
        # If we need vectors, start the search - complete this first before strategy determination
        if should_search_vectors:
            # Perform vector search
            logger.debug(f"[ADV_ENGINE] Starting vector search with session vector")
            vector_search_start = time.time()
            try:
                vector_matches = await perform_vector_search(session_vector, session_state)
                vector_search_time = time.time() - vector_search_start
                logger.info(f"[ADV_ENGINE] Vector search completed in {vector_search_time:.3f}s with {len(vector_matches)} matches")
                
                if vector_matches and len(vector_matches) > 0:
                    top_match = vector_matches[0]
                    if hasattr(top_match, 'score') and hasattr(top_match, 'payload'):
                        logger.debug(f"[ADV_ENGINE] Top match score: {top_match.score}, question: '{top_match.payload.get('question_text', 'N/A')[:50]}...'")
            except Exception as e:
                logger.error(f"[ADV_ENGINE] Vector search failed: {str(e)}")
                vector_matches = []
        
        # 4. Extract previously asked questions for avoiding repetition (do this early)
        asked_questions = extract_asked_questions(session_state)
        logger.debug(f"[ADV_ENGINE] Extracted {len(asked_questions)} previously asked questions")
        
        asked_question_topics = extract_question_topics(asked_questions)
        logger.debug(f"[ADV_ENGINE] Extracted topics: {asked_question_topics}")
        
        # Get vector examples for strategy determination
        vector_examples = []
        if vector_matches:
            vector_examples = [m.payload for m in vector_matches] if hasattr(vector_matches[0], 'payload') else vector_matches
            logger.debug(f"[ADV_ENGINE] Using {len(vector_examples)} vector examples for strategy determination")
        
        # Determine strategy with all the information we have
        logger.debug(f"[ADV_ENGINE] Starting strategy determination")
        strategy_start_time = time.time()
        try:
            # Use deterministic strategy selection instead of LLM-based selection
            strategy = await deterministic_strategy(
                session_state=session_state,
                vector_examples=vector_examples,
                asked_questions=asked_questions
            )
            strategy_time = time.time() - strategy_start_time
            
            logger.info(f"[ADV_ENGINE] Strategy determined in {strategy_time:.3f}s: {strategy.get('recommended_approach')}")
            logger.debug(f"[ADV_ENGINE] Full strategy: {strategy}")
        except asyncio.TimeoutError:
            logger.warning(f"[ADV_ENGINE] Strategy determination timed out after 3.0s, using default strategy")
            strategy = {
                "recommended_approach": "custom_generation",
                "target_metric": get_highest_priority_metric(session_state),
                "specific_focus": "general experience",
                "question_format": "open-ended",
                "rationale": "Default strategy due to timeout"
            }
            logger.debug(f"[ADV_ENGINE] Default strategy: {strategy}")
        except Exception as e:
            logger.error(f"[ADV_ENGINE] Strategy determination failed: {str(e)}")
            strategy = {
                "recommended_approach": "custom_generation",
                "target_metric": get_highest_priority_metric(session_state),
                "specific_focus": "general experience",
                "question_format": "open-ended",
                "rationale": "Default strategy due to error"
            }
        
        logger.info(f"[ADV_ENGINE] Final strategy: approach={strategy['recommended_approach']}, target={strategy.get('target_metric')}, focus={strategy.get('specific_focus')}")
        
        # 6. Select the question based on confidence and strategy
        logger.info(f"[ADV_ENGINE] Question selection phase starting with strategy: {strategy['recommended_approach']}")
        
        if strategy["recommended_approach"] == "reuse_past_pattern" and history_count >= 2 and meaningful_response:
            logger.debug(f"[ADV_ENGINE] Attempting to reuse past pattern with {len(vector_matches)} vector matches")
            
            # Check if the high-confidence match is about a topic we already covered
            if not vector_matches:
                logger.warning(f"[ADV_ENGINE] Strategy suggests reusing past pattern but no vector matches found")
            else:
                match_question = vector_matches[0].payload.get("question_text", "").lower()
                match_score = vector_matches[0].score if hasattr(vector_matches[0], 'score') else 0
                
                logger.debug(f"[ADV_ENGINE] Top match (score: {match_score}): '{match_question}'")
                
                # Skip if too similar to previously asked questions (increased threshold)
                similarity_check_result = is_question_too_similar(match_question, asked_questions, similarity_threshold=0.55)
                logger.debug(f"[ADV_ENGINE] Similarity check result: is_too_similar={similarity_check_result}")
                
                if not similarity_check_result:
                    # Use the high-confidence vector match
                    try:
                        logger.info(f"[ADV_ENGINE] Using high-confidence vector match (score: {match_score})")
                        question = format_question_from_vector(vector_matches[0].payload)
                        question["selection_method"] = "vector_exact"
                        logger.info(f"[ADV_ENGINE] Selected exact vector match: '{question.get('question_text')[:50]}...'")
                        
                        # Log the full question details
                        logger.debug(f"[ADV_ENGINE] Final question details: {question}")
                        
                        logger.info(f"[ADV_ENGINE] Advanced question engine completed in {time.time() - start_time:.3f}s")
                        return question
                    except Exception as e:
                        logger.error(f"[ADV_ENGINE] Error formatting question from vector: {str(e)}")
                        # Continue to fallback methods
                else:
                    logger.info(f"[ADV_ENGINE] Skipping high-confidence match because it's too similar to previous questions")
            
        elif strategy["recommended_approach"] == "category_search" and history_count >= 2 and meaningful_response:
            # Use the LLM-suggested category to guide vector search
            category = strategy.get("category", "general")
            logger.info(f"[ADV_ENGINE] Searching for questions in category: {category}")
            
            try:
                logger.debug(f"[ADV_ENGINE] Starting category-based vector search for: {category}")
                start_category_search = time.time()
                category_matches = await search_vectors_by_category(
                    vector=session_vector,
                    category=category,
                    limit=2
                )
                category_search_time = time.time() - start_category_search
                
                logger.debug(f"[ADV_ENGINE] Category search completed in {category_search_time:.3f}s with {len(category_matches) if category_matches else 0} matches")
                
                if category_matches and category_matches[0].score > 0.85:
                    match_question = category_matches[0].payload.get("question_text", "").lower()
                    match_score = category_matches[0].score
                    
                    logger.debug(f"[ADV_ENGINE] Top category match (score: {match_score}): '{match_question}'")
                    
                    # Check for suspicious content
                    is_generic = any(re.search(pattern, match_question) for pattern in generic_patterns)
                    is_similar = is_question_too_similar(match_question, asked_questions, similarity_threshold=0.55)
                    
                    logger.debug(f"[ADV_ENGINE] Question checks: is_generic={is_generic}, is_similar={is_similar}")
                    
                    if is_generic:
                        logger.warning(f"[ADV_ENGINE] Rejected suspicious category match: '{match_question}'")
                    elif not is_similar:
                        logger.info(f"[ADV_ENGINE] Using category match (score: {match_score})")
                        question = format_question_from_vector(category_matches[0].payload)
                        question["selection_method"] = "vector_category"
                        logger.info(f"[ADV_ENGINE] Selected category vector match: '{question.get('question_text')[:50]}...'")
                        
                        # Log the full question details
                        logger.debug(f"[ADV_ENGINE] Final question details: {question}")
                        
                        logger.info(f"[ADV_ENGINE] Advanced question engine completed in {time.time() - start_time:.3f}s")
                        return question
                    else:
                        logger.info("Skipping category match because it's too similar to previous questions")
                else:
                    if category_matches:
                        logger.debug(f"[ADV_ENGINE] Best category match score {category_matches[0].score} below threshold (0.85)")
                    else:
                        logger.debug(f"[ADV_ENGINE] No category matches found")
            except Exception as e:
                logger.error(f"[ADV_ENGINE] Category search failed: {str(e)}")
                # Continue to fallback methods
        
        # 7. If no good matches or LLM recommends custom question, use enhanced RAG approach
        examples = [m.payload for m in vector_matches] if vector_matches else []
        logger.info(f"[ADV_ENGINE] Using custom question generation with {len(examples)} example questions")
        
        # Generate a tailored question using the LLM with vector examples as context
        try:
            # Add the conversation context and metrics data for more relevant questions
            target_metric = strategy.get('target_metric', get_highest_priority_metric(session_state))
            logger.info(f"[ADV_ENGINE] Generating custom question for metric: {target_metric}")
            
            # Extract key information for context
            logger.debug(f"[ADV_ENGINE] Creating conversation summary and metrics insights")
            start_summary_time = time.time()
            conversation_summary = summarize_conversation(session_state)
            metrics_insights = analyze_metrics(session_state)
            summary_time = time.time() - start_summary_time
            
            logger.debug(f"[ADV_ENGINE] Context preparation took {summary_time:.3f}s")
            logger.debug(f"[ADV_ENGINE] Conversation summary length: {len(conversation_summary)}")
            logger.debug(f"[ADV_ENGINE] Conversation summary: {conversation_summary}")
            logger.debug(f"[ADV_ENGINE] Metrics insights length: {len(metrics_insights)}")
            logger.debug(f"[ADV_ENGINE] Metrics insights: {metrics_insights}")
            
            # Enhanced question generation with richer context
            logger.info(f"[ADV_ENGINE] Calling LLM for question generation")
            start_llm_time = time.time()
            llm_question = await llm_question_with_examples(
                session_state=session_state,
                metric_id=target_metric,
                examples=examples,
                strategy=strategy,
                conversation_summary=conversation_summary,
                metrics_insights=metrics_insights,
                avoid_topics=asked_question_topics
            )
            llm_time = time.time() - start_llm_time
            
            logger.debug(f"[ADV_ENGINE] LLM question generation took {llm_time:.3f}s")
            logger.debug(f"[ADV_ENGINE] Raw LLM question: {llm_question}")
            
            # Perform a sanity check on the generated question
            question_text = llm_question.get("question_text", "").lower() if llm_question else ""
            
            # If llm_question is None, log error and continue to fallback
            if not llm_question or not question_text:
                logger.error(f"[ADV_ENGINE] LLM returned empty or null question")
                raise ValueError("LLM returned empty question")
                
            # Strip quotes if they're present (some LLMs add quotes)
            question_text = question_text.strip('"\'')
            logger.debug(f"[ADV_ENGINE] Processed question text: '{question_text}'")
            
            # Check if it's a truly generic question with no specific context
            is_generic = False
            
            # Only reject if it matches one of our generic patterns AND has no specific context
            generic_match = next((pattern for pattern in generic_patterns if re.search(pattern, question_text)), None)
            if generic_match:
                logger.debug(f"[ADV_ENGINE] Question matches generic pattern: {generic_match}")
                
                # Look for specific context that would make this question valid
                # Check for references to previous responses or specific metrics
                context_keywords = ["previously", "earlier", "mentioned", "fellowship program"]
                context_patterns = [r"\b(skill|experience|background|challenge|project)\b"]
                
                has_context_keyword = any(keyword in question_text for keyword in context_keywords)
                has_context_pattern = any(re.search(pattern, question_text) for pattern in context_patterns)
                has_specific_context = has_context_keyword or has_context_pattern
                
                logger.debug(f"[ADV_ENGINE] Context check: has_keyword={has_context_keyword}, has_pattern={has_context_pattern}")
                
                if not has_specific_context:
                    # Only reject truly generic questions without context
                    logger.warning(f"[ADV_ENGINE] Rejected generic question without context: '{question_text}'")
                    is_generic = True
                else:
                    # Log but allow questions with generic patterns but specific context
                    logger.info(f"[ADV_ENGINE] Allowing question with specific context: '{question_text}'")
            
            if is_generic:
                # Generate a safer question using rule-based approach
                logger.info(f"[ADV_ENGINE] Using safe fallback for generic question")
                metric_id = target_metric
                safe_question = {
                    "question_id": f"llm_{int(datetime.now().timestamp())}",
                    "question_text": f"How would you describe your experience with {metric_id.replace('_', ' ')}?",
                    "metric_id": metric_id,
                    "type": "text",
                    "source": "llm_fallback",
                    "selection_method": "llm_safe"
                }
                
                logger.debug(f"[ADV_ENGINE] Safe fallback question: {safe_question}")
                logger.info(f"[ADV_ENGINE] Advanced question engine completed in {time.time() - start_time:.3f}s")
                return safe_question
            
            llm_question["selection_method"] = "llm_rag"
            logger.info(f"[ADV_ENGINE] Generated custom LLM question: '{llm_question.get('question_text')[:50]}...'")
            
            # 8. Store this new question pattern for future reuse
            try:
                logger.debug(f"[ADV_ENGINE] Storing new question pattern in vector database")
                await store_new_question_pattern(
                    session_vector=session_vector,
                    question=llm_question,
                    strategy=strategy
                )
                logger.debug(f"[ADV_ENGINE] Question pattern stored successfully")
            except Exception as e:
                logger.error(f"[ADV_ENGINE] Failed to store new question pattern: {str(e)}")
                # This is non-critical, so we continue
            
            # Log the full question details before returning
            logger.debug(f"[ADV_ENGINE] Final LLM question details: {llm_question}")
            logger.info(f"[ADV_ENGINE] Advanced question engine completed in {time.time() - start_time:.3f}s")
            return llm_question
        except Exception as e:
            logger.error(f"[ADV_ENGINE] LLM question generation failed: {str(e)}")
            # Fall through to emergency fallback
        
        # Track total processing time
        elapsed = time.time() - start_time
        logger.info(f"[ADV_ENGINE] Advanced question engine completed in {elapsed:.3f}s, using emergency fallback")
        
        # If a question was already returned in the process above, it would have returned
        # This is the fallback path
        metric_id = get_highest_priority_metric(session_state)
        topic = metric_id.replace('_', ' ')
        
        fallback_questions = [
            f"Can you share more about your experience with {topic}?",
            f"What aspects of {topic} are most important to you?",
            f"How would you describe your ideal {topic} experience?",
            f"What specific challenges have you faced regarding {topic}?",
            f"What improvements would you suggest for {topic}?"
        ]
        
        import random
        fallback_index = random.randint(0, len(fallback_questions)-1)
        fallback_text = fallback_questions[fallback_index]
        
        logger.info(f"[ADV_ENGINE] Using emergency fallback question ({fallback_index+1}/{len(fallback_questions)})")
        fallback_question = {
            "question_id": f"adv_{int(datetime.now().timestamp())}",
            "question_text": fallback_text,
            "metric_id": metric_id,
            "type": "text",
            "source": "fallback",
            "selection_method": "emergency_fallback"
        }
        
        logger.debug(f"[ADV_ENGINE] Emergency fallback question: {fallback_question}")
        return fallback_question
    except Exception as e:
        logger.error(f"[ADV_ENGINE] Error in advanced question engine: {str(e)}")
        # Emergency fallback
        metric_id = get_highest_priority_metric(session_state)
        emergency_question = {
            "question_id": f"adv_{int(datetime.now().timestamp())}",
            "question_text": f"What's most important to you about {metric_id.replace('_', ' ')}?",
            "metric_id": metric_id,
            "type": "text",
            "source": "fallback",
            "selection_method": "error_fallback"
        }
        logger.debug(f"[ADV_ENGINE] Error fallback question: {emergency_question}")
        return emergency_question

async def process_question_feedback(
    session_id: str, 
    question_id: str, 
    response: Dict[str, Any], 
    metrics_update: Dict[str, Any], 
    user_satisfaction: Optional[float] = None
):
    """Process feedback after a question has been answered to improve future questions.
    
    Args:
        session_id: The session ID
        question_id: The question ID
        response: The user's response
        metrics_update: Changes to metrics after the response
        user_satisfaction: Optional user satisfaction rating
    """
    try:
        logger.info(f"[FEEDBACK] Processing question feedback for session: {session_id}, question: {question_id}")
        logger.debug(f"[FEEDBACK] Response length: {len(response.get('response_text', ''))}")
        logger.debug(f"[FEEDBACK] Metrics update: {metrics_update}")
        if user_satisfaction:
            logger.debug(f"[FEEDBACK] User satisfaction: {user_satisfaction}")
        
        # Get session state
        start_time = time.time()
        session_state = await get_session(session_id)
        if not session_state:
            logger.error(f"[FEEDBACK] Session {session_id} not found, cannot process feedback")
            return
        
        # Find the question in history
        question = None
        history_length = len(session_state.get("question_history", []))
        logger.debug(f"[FEEDBACK] Searching for question in history of {history_length} items")
        
        for item in session_state.get("question_history", []):
            if item.get("id") == question_id:
                question = item
                break
        
        if not question:
            logger.warning(f"[FEEDBACK] Question {question_id} not found in session history")
            return
        
        logger.debug(f"[FEEDBACK] Found question: {question.get('text', '')[:50]}...")
        
        # Calculate effectiveness score based on metrics update
        logger.debug(f"[FEEDBACK] Calculating effectiveness score")
        effectiveness = calculate_question_effectiveness(
            question=question,
            response=response,
            metrics_update=metrics_update,
            user_satisfaction=user_satisfaction
        )
        
        logger.info(f"[FEEDBACK] Question effectiveness score: {effectiveness:.2f}")
        
        # Update the question's success stats in the vector database
        if question.get("selection_method") in ("vector_exact", "vector_category"):
            # Update existing vector
            logger.info(f"[FEEDBACK] Updating vector success rate for {question.get('selection_method')} question")
            try:
                await update_vector_success_rate(
                    question_id=question_id,
                    success_score=effectiveness,
                    response=response
                )
                logger.debug(f"[FEEDBACK] Vector success rate updated successfully")
            except Exception as e:
                logger.error(f"[FEEDBACK] Failed to update vector success rate: {str(e)}")
        
        elif question.get("selection_method") == "llm_rag":
            # For LLM-generated questions that were effective, add to vector DB
            if effectiveness > 0.7:  # Only store effective questions
                logger.info(f"[FEEDBACK] Storing effective LLM question (score: {effectiveness:.2f}) in vector DB")
                try:
                    logger.debug(f"[FEEDBACK] Creating session vector for storage")
                    session_vector = await create_session_vector(session_state)
                    await store_successful_question(
                        session_vector=session_vector,
                        question=question,
                        response=response,
                        effectiveness=effectiveness
                    )
                    logger.debug(f"[FEEDBACK] Successful question stored in vector DB")
                except Exception as e:
                    logger.error(f"[FEEDBACK] Failed to store successful question: {str(e)}")
            else:
                logger.debug(f"[FEEDBACK] LLM question not stored (effectiveness {effectiveness:.2f} below threshold 0.7)")
        
        elapsed = time.time() - start_time
        logger.info(f"[FEEDBACK] Question feedback processing completed in {elapsed:.3f}s")
    except Exception as e:
        logger.error(f"[FEEDBACK] Error in process_question_feedback: {str(e)}")

# Helper functions

async def search_vectors(vector: List[float], limit: int = 3, score_threshold: float = 0.75):
    """Search for similar vectors in the question patterns collection.
    
    Args:
        vector: The session vector to search with
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold
        
    Returns:
        List of matching vectors with their payloads
    """
    # Get qdrant client
    try:
        logger.debug(f"[VECTOR_SEARCH] Starting vector search (limit: {limit}, threshold: {score_threshold})")
        start_time = time.time()
        
        client = await get_qdrant_client()
        logger.debug(f"[VECTOR_SEARCH] Qdrant client initialized")
        
        # Search for similar vectors
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        logger.debug(f"[VECTOR_SEARCH] Searching collection: {collection_name}")
        
        # QdrantClient methods are synchronous, don't use await
        results = client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        elapsed = time.time() - start_time
        result_count = len(results) if results else 0
        logger.info(f"[VECTOR_SEARCH] Search completed in {elapsed:.3f}s with {result_count} results")
        
        if result_count > 0:
            top_score = results[0].score if results else None
            logger.debug(f"[VECTOR_SEARCH] Top match score: {top_score}")
            
        return results
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] Error searching vectors: {str(e)}")
        # Log the raw response if available
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"[VECTOR_SEARCH] Raw response content:\n{e.response.content}")
        raise  # Re-raise to let the caller handle it

async def search_vectors_by_category(
    vector: List[float],
    category: str,
    limit: int = 1
):
    """Search vectors within a specific category.
    
    Args:
        vector: The session vector
        category: The category to search within
        limit: Maximum number of results
        
    Returns:
        List of matching vectors
    """
    try:
        logger.debug(f"[CATEGORY_SEARCH] Starting category search for '{category}' (limit: {limit})")
        start_time = time.time()
        
        client = await get_qdrant_client()
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        
        # Use filtering to search within categories
        filter_condition = {
            "must": [
                {
                    "key": "category",
                    "match": {
                        "value": category
                    }
                }
            ]
        }
        
        logger.debug(f"[CATEGORY_SEARCH] Using filter condition: {filter_condition}")
        
        # Search with category filter - QdrantClient methods are synchronous, don't use await
        results = client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=filter_condition,
            limit=limit,
            score_threshold=0.65  # Lower threshold when filtering by category
        )
        
        elapsed = time.time() - start_time
        result_count = len(results) if results else 0
        logger.info(f"[CATEGORY_SEARCH] Search completed in {elapsed:.3f}s with {result_count} results")
        
        if result_count > 0:
            top_score = results[0].score if results else None
            logger.debug(f"[CATEGORY_SEARCH] Top match score: {top_score}")
            if top_score and results[0].payload and 'question_text' in results[0].payload:
                logger.debug(f"[CATEGORY_SEARCH] Top match question: '{results[0].payload['question_text'][:50]}...'")
        
        return results
    except Exception as e:
        logger.error(f"[CATEGORY_SEARCH] Error searching vectors by category: {str(e)}")
        # Log the raw response if available
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"[CATEGORY_SEARCH] Raw response content:\n{e.response.content}")
        raise  # Re-raise to let the caller handle it

def format_question_from_vector(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Format a question from a vector search result payload.
    
    Args:
        payload: The payload from the vector search result
        
    Returns:
        A formatted question
    """
    # Generate a valid question_id if not present
    question_id = payload.get("question_id")
    if not question_id:
        question_id = f"vector_{int(datetime.now().timestamp())}"
    
    # Ensure question type is valid
    question_type = payload.get("question_type", "text")
    
    return {
        "question_text": payload.get("question_text", ""),
        "metric_id": payload.get("metric_id", ""),
        "question_id": question_id,
        "source": "vector_db",
        "type": question_type
    }

async def get_survey_context(session_state: Dict[str, Any]) -> str:
    """Get formatted survey context for LLM prompts.
    
    Args:
        session_state: The current session state
        
    Returns:
        Formatted survey context as a string
    """
    try:
        from app.api.endpoints.survey import get_survey
        from app.core.metrics import get_metric_details
        
        survey_id = session_state.get("survey_id")
        if not survey_id:
            return "No survey information available."
        
        # Get survey details
        survey = await get_survey(survey_id)
        if not survey:
            return "Survey information not found."
        
        # Get metrics information
        metrics = session_state.get("metrics", {})
        total_metrics = len(metrics)
        completed_metrics = sum(1 for m_id, m in metrics.items() 
                               if m.get("confidence", 0) >= 0.75)
        
        # Format survey context
        context = f"""
SURVEY GOAL: {survey.get('title', 'Untitled Survey')}
SURVEY PURPOSE: {survey.get('description', 'No description available')}
PROGRESS: {completed_metrics}/{total_metrics} metrics adequately assessed

METRICS NEEDING ASSESSMENT:
"""
        
        # Add details for pending metrics
        pending_metrics = session_state.get("metrics_pending", [])
        for metric_id in pending_metrics:
            metric_details = await get_metric_details(metric_id)
            if metric_details:
                # Get current confidence
                current_confidence = metrics.get(metric_id, {}).get("confidence", 0)
                questions_asked = metrics.get(metric_id, {}).get("questions_asked", 0)
                min_questions = metric_details.get("min_questions", 2)
                max_questions = metric_details.get("max_questions", 5)
                
                context += f"- {metric_details.get('name', metric_id)}: {metric_details.get('description', 'No description')}\n"
                context += f"  Importance: {metric_details.get('importance', 1.0)}/5.0, "
                context += f"Confidence: {current_confidence:.2f}/1.0, "
                context += f"Questions: {questions_asked}/{max_questions}\n"
        
        # Add efficiency reminder
        context += """
EFFICIENCY GOAL: Extract maximum insight with minimum questions. Each question should 
provide valuable information about the target metric. The ideal survey has high data 
quality with the fewest possible questions.
"""
        
        return context
    except Exception as e:
        logger.error(f"Error getting survey context: {str(e)}")
        return "Error retrieving survey context."

async def llm_determine_strategy(
    session_state: Dict[str, Any],
    vector_examples: List[Dict[str, Any]],
    asked_questions: List[str] = None
) -> Dict[str, Any]:
    """Use LLM to determine the best questioning strategy with enhanced context awareness.
    
    Args:
        session_state: Current session state
        vector_examples: Similar past interactions from vector search
        asked_questions: Previously asked questions to avoid repetition
        
    Returns:
        Strategy recommendation with approach and target metric
    """
    # Check LLM response cache first to avoid duplicate API calls
    session_id = session_state.get("session_id", "")
    history_len = len(session_state.get("question_history", []))
    metric_count = len(session_state.get("metrics_pending", []))
    
    # Create a cache key that's stable for similar conversation states
    cache_key = f"llm_strategy_{history_len}_{metric_count}"
    
    # Add the latest question to the cache key for better differentiation
    if history_len > 0:
        latest_q = session_state.get("question_history", [])[-1].get("text", "")[:30]
        # Create a stable hash of the latest question
        q_hash = hashlib.md5(latest_q.encode()).hexdigest()[:8]
        cache_key += f"_{q_hash}"
    
    # Try to get from cache
    try:
        redis = await get_redis()
        cached_response = await redis.get(cache_key)
        if cached_response:
            logger.info(f"Using cached LLM strategy response")
            return json.loads(cached_response)
    except Exception as e:
        logger.warning(f"Error checking LLM cache: {str(e)}")
    
    # Debug log for API keys (don't log actual keys, just their presence)
    logger.debug(f"ANTHROPIC_API_KEY available: {bool(settings.ANTHROPIC_API_KEY)}")
    logger.debug(f"OPENAI_API_KEY available: {bool(settings.OPENAI_API_KEY)}")
    
    # Get survey context
    survey_context = await get_survey_context(session_state)
    
    # Format questions to avoid
    avoid_questions = ""
    if asked_questions:
        avoid_questions = "PREVIOUSLY ASKED QUESTIONS (AVOID REPETITION):\n" + "\n".join([f"- {q}" for q in asked_questions[-5:]])
    
    # Format context for LLM with more detailed instructions
    prompt = f"""
You are an expert survey designer for a data collection system. Your goal is to determine the optimal questioning strategy that maximizes information with minimal questions.

{survey_context}

CONVERSATION HISTORY:
{format_conversation_history(session_state)}

CURRENT METRICS STATUS:
{format_metrics_status(session_state)}

{avoid_questions}

SIMILAR PAST INTERACTIONS:
{format_vector_examples(vector_examples)}

Based on this information, what is the best strategy to ask the next question?
The question should be specific, detailed, build on previous responses, and efficiently gather information for the target metric.
Each question should maximize information gain toward completing the survey with high confidence scores.
Generic questions rarely provide valuable insights.

Consider these approaches:
1. reuse_past_pattern: Use a question pattern from similar past interactions
2. category_search: Focus on a specific category of questions
3. custom_generation: Generate a completely custom question

Your response should include:
- recommended_approach: One of the approaches above
- target_metric: Which metric to focus on
- category: If using category_search, which category to search in
- avoid_topics: List of topics to avoid (to prevent repetition)
- specific_focus: A specific aspect of the metric to focus on
- question_format: Suggested format (open-ended, scaling, comparison, etc.)
- rationale: Brief explanation of this choice emphasizing how it helps achieve the survey goals efficiently
"""
    
    # Add full prompt logging
    logger.info("=========== FULL STRATEGY PROMPT START ===========")
    logger.info(prompt)
    logger.info("=========== FULL STRATEGY PROMPT END ===========")
    
    try:
        # Check for available API keys - first try OpenAI, then fall back to Anthropic
        has_openai = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here" and settings.OPENAI_API_KEY != "your_openai_key_here"
        has_anthropic = settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your_anthropic_api_key_here"
        
        if not has_openai and not has_anthropic:
            logger.error("No LLM API keys properly configured")
            # Return a default strategy
            return {
                "recommended_approach": "custom_generation",
                "target_metric": get_highest_priority_metric(session_state),
                "specific_focus": "general experience",
                "question_format": "open-ended",
                "rationale": "Default strategy due to missing API keys"
            }
        
        # Try models in sequence with proper fallback
        llm_response = None
        llm_error = None
        
        # First try OpenAI (more reliable)
        if has_openai:
            try:
                model = "gpt-3.5-turbo"
                logger.info("Using OpenAI model for strategy determination")
                llm_response = await query_llm(
                    prompt=prompt, 
                    response_format="json",
                    model=model,
                    system_prompt="You are an expert survey designer optimizing questions to collect maximum data in minimum time."
                )
            except Exception as e:
                logger.warning(f"OpenAI strategy determination failed: {str(e)}")
                llm_error = e
        
        # Try Anthropic as fallback
        if not llm_response and has_anthropic:
            try:
                model = "claude-3-sonnet-20240229"
                logger.info("Using Anthropic model for strategy determination")
                llm_response = await query_llm(
                    prompt=prompt, 
                    response_format="json",
                    model=model,
                    system_prompt="You are an expert survey designer optimizing questions to collect maximum data in minimum time."
                )
            except Exception as e:
                logger.warning(f"Anthropic strategy determination failed: {str(e)}")
                if llm_error is None:
                    llm_error = e
        
        # If both failed, raise the error
        if not llm_response:
            if llm_error:
                raise llm_error
            else:
                raise Exception("Failed to get LLM response")
        
        # Parse and validate the response
        strategy = json.loads(llm_response)
        required_fields = ["recommended_approach", "target_metric"]
        
        if all(field in strategy for field in required_fields):
            # Cache the LLM response for 10 minutes
            try:
                redis = await get_redis()
                await redis.set(cache_key, json.dumps(strategy), ex=600)  # 10 minute TTL
                logger.debug(f"Cached LLM strategy response with key: {cache_key}")
            except Exception as e:
                logger.warning(f"Error caching LLM response: {str(e)}")
                
            return strategy
        else:
            # Default strategy if LLM response is invalid
            logger.warning("Invalid LLM strategy response, using default")
            return {
                "recommended_approach": "custom_generation",
                "target_metric": get_highest_priority_metric(session_state),
                "specific_focus": "general experience",
                "question_format": "open-ended",
                "rationale": "Default strategy due to invalid LLM response"
            }
            
    except Exception as e:
        logger.error(f"Error determining strategy with LLM: {str(e)}")
        # Fallback to default strategy
        return {
            "recommended_approach": "custom_generation",
            "target_metric": get_highest_priority_metric(session_state),
            "specific_focus": "general experience",
            "question_format": "open-ended",
            "rationale": "Default strategy due to LLM error"
        }

async def llm_question_with_examples(
    session_state: Dict[str, Any],
    metric_id: str,
    examples: List[Dict[str, Any]],
    strategy: Dict[str, Any],
    conversation_summary: str = "",
    metrics_insights: str = "",
    avoid_topics: List[str] = None
) -> Dict[str, Any]:
    """Generate a question using LLM with enhanced context awareness.
    
    Args:
        session_state: Current session state
        metric_id: Target metric ID
        examples: Similar question examples from vector search
        strategy: Strategy determined by LLM
        conversation_summary: Summary of the conversation so far
        metrics_insights: Insights from metrics analysis
        avoid_topics: Topics to avoid in the question
        
    Returns:
        Generated question
    """
    session_id = session_state.get("session_id", "unknown")
    logger.info(f"[LLM_QUESTION] Generating question for metric '{metric_id}' in session {session_id}")
    logger.debug(f"[LLM_QUESTION] Strategy: {strategy.get('recommended_approach')}, Examples: {len(examples)}")
    
    # Debug log for API keys (don't log actual keys, just their presence)
    has_anthropic = bool(settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your_anthropic_api_key_here")
    has_openai = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here" and settings.OPENAI_API_KEY != "your_openai_key_here")
    
    logger.debug(f"[LLM_QUESTION] API availability: ANTHROPIC={has_anthropic}, OPENAI={has_openai}")
    
    # Get detailed metric information
    from app.core.metrics import get_metric_details
    start_time = time.time()
    metric_details = await get_metric_details(metric_id)
    logger.debug(f"[LLM_QUESTION] Got metric details in {(time.time() - start_time):.3f}s: {metric_details is not None}")
    if metric_details:
        logger.debug(f"[LLM_QUESTION] Metric details keys: {metric_details.keys() if metric_details else None}")
    
    # Get survey context
    start_time = time.time()
    survey_context = await get_survey_context(session_state)
    logger.debug(f"[LLM_QUESTION] Got survey context in {(time.time() - start_time):.3f}s, length: {len(survey_context)}")
    
    # Format examples for LLM context
    start_time = time.time()
    formatted_examples = "\n\n".join([
        f"EXAMPLE {i+1}:\n"
        f"Context: {e.get('context', 'No context')}\n"
        f"Question: {e.get('question_text', 'No question')}\n"
        f"Success Rate: {e.get('success_rate', 'Unknown')}"
        for i, e in enumerate(examples)
    ])
    logger.debug(f"[LLM_QUESTION] Formatted {len(examples)} examples in {(time.time() - start_time):.3f}s")
    
    # Format topics to avoid
    avoid_text = ""
    if avoid_topics and len(avoid_topics) > 0:
        avoid_text = "TOPICS TO AVOID (already covered):\n" + ", ".join(avoid_topics)
        logger.debug(f"[LLM_QUESTION] Topics to avoid: {avoid_topics}")
    
    # Format specific focus from strategy
    specific_focus = strategy.get("specific_focus", "general experience")
    question_format = strategy.get("question_format", "open-ended")
    logger.debug(f"[LLM_QUESTION] Focus: {specific_focus}, Format: {question_format}")
    
    # Enhance topic avoidance with explicit list of previously asked questions
    # This helps the LLM understand what questions have already been asked
    question_history = session_state.get("question_history", [])
    previous_questions_list = []
    
    # Extract the most recent questions about this metric (up to 5)
    metric_questions = [item for item in question_history 
                      if item.get("question", {}).get("metric_id") == metric_id][-5:]
    
    if metric_questions:
        previous_questions_list = [f"- {q.get('question', {}).get('question_text', '')}" 
                                for q in metric_questions]
        logger.debug(f"[LLM_QUESTION] Found {len(previous_questions_list)} previous questions for this metric")
    
    # Combine with avoid topics
    previous_questions_text = ""
    if previous_questions_list:
        previous_questions_text = "PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):\n" + "\n".join(previous_questions_list) + "\n\n"
    
    # Get metric description from details or fallback to function
    metric_description = ""
    if metric_details is not None:
        metric_description = metric_details.get("description", get_metric_description(metric_id))
    else:
        metric_description = get_metric_description(metric_id)
    
    # Get human-readable metric name instead of using UUID
    metric_name = metric_id.replace('_', ' ')
    if metric_details is not None:
        metric_name = metric_details.get("name", metric_name)
    
    logger.debug(f"[LLM_QUESTION] Metric description: '{metric_description[:50]}...'")
    logger.debug(f"[LLM_QUESTION] Using human-readable metric name: '{metric_name}'")
    
    # Get metric confidence and questions asked so far
    metric_data = session_state.get("metrics", {}).get(metric_id, {})
    current_confidence = metric_data.get("confidence", 0)
    questions_asked = metric_data.get("questions_asked", 0)
    
    # Add null check for metric_details before accessing min_questions and max_questions
    min_questions = 2
    max_questions = 5
    if metric_details is not None:
        min_questions = metric_details.get("min_questions", 2)
        max_questions = metric_details.get("max_questions", 5)
    
    logger.debug(f"[LLM_QUESTION] Metric progress: confidence={current_confidence:.2f}, questions={questions_asked}/{max_questions}")
    
    # Create prompt with enhanced context and more specific instructions
    prompt = f"""
You are an expert survey designer tasked with generating a highly effective question to collect maximum data in minimum time.

{survey_context}

TARGET METRIC: {metric_name}
METRIC DESCRIPTION: {metric_description}
SPECIFIC FOCUS: {specific_focus}
RECOMMENDED FORMAT: {question_format}
METRIC PROGRESS: Confidence {current_confidence:.2f}/1.0, Questions {questions_asked}/{max_questions}

CONVERSATION SUMMARY:
{conversation_summary if conversation_summary else format_recent_conversation(session_state, max_turns=3)}

METRICS INSIGHTS:
{metrics_insights if metrics_insights else format_metrics_status(session_state)}

{previous_questions_text}{avoid_text}

STRATEGY RATIONALE:
{strategy.get('rationale', 'Generate the best question to assess the target metric')}

SIMILAR SUCCESSFUL QUESTIONS:
{formatted_examples}

Based on this context, generate ONE question that will effectively assess the {metric_name} metric.
Your question should be:
1. HIGHLY SPECIFIC - avoid generic questions like "Tell me about your experience" or "What do you think about X"
2. DETAILED - include specific details from previous responses to show continuity
3. FOCUSED - target the exact information needed about {specific_focus}
4. CONVERSATIONAL - natural and engaging, as if asked by a skilled interviewer
5. FRESH - not repetitive of previous questions, NEVER ask a question similar to those listed in PREVIOUSLY ASKED QUESTIONS
6. EFFICIENT - designed to maximize information gain with minimal questions

BAD EXAMPLE: "How would you describe your {metric_name}?"
GOOD EXAMPLE: "You mentioned dealing with [specific issue] during [specific time]. How did that impact your ability to [specific activity related to metric]?"

IMPORTANT REMINDER: The goal is to collect maximum information with minimum questions. Your question should provide high data value toward completing the assessment of this metric.

QUESTION:
"""
    
    logger.debug(f"[LLM_QUESTION] Prepared prompt of length {len(prompt)}")
    
    # Add full prompt logging similar to session.py
    logger.info("=========== FULL LLM QUESTION PROMPT START ===========")
    logger.info(prompt)
    logger.info("=========== FULL LLM QUESTION PROMPT END ===========")
    
    try:
        # Check for available API keys
        if not has_openai and not has_anthropic:
            logger.error("[LLM_QUESTION] No LLM API keys properly configured")
            # Return a fallback question
            fallback = {
                "question_text": f"Can you tell me specifically about how {specific_focus} affects your {metric_id.replace('_', ' ')}?",
                "metric_id": metric_id,
                "question_id": f"fallback_{int(datetime.now().timestamp())}",
                "source": "fallback",
                "strategy": "fallback",
                "type": "text"
            }
            logger.info(f"[LLM_QUESTION] Returning configuration fallback question")
            return fallback
        
        # Try models in sequence with proper fallback
        question_text = None
        llm_error = None
        
        # First try OpenAI (more reliable)
        if has_openai:
            try:
                model = "gpt-3.5-turbo"
                logger.info(f"[LLM_QUESTION] Using OpenAI model {model} for question generation")
                start_time = time.time()
                question_text = await query_llm(
                    prompt=prompt, 
                    max_tokens=150,  # Allow for longer, more detailed questions
                    model=model,
                    system_prompt="You are an expert survey designer creating questions to collect maximum data in minimum time."
                )
                elapsed = time.time() - start_time
                
                if question_text:
                    logger.info(f"[LLM_QUESTION] OpenAI returned response in {elapsed:.3f}s, length: {len(question_text)}")
                    logger.debug(f"[LLM_QUESTION] OpenAI response: '{question_text[:100]}...'")
                else:
                    logger.warning(f"[LLM_QUESTION] OpenAI returned empty response after {elapsed:.3f}s")
            except Exception as e:
                logger.warning(f"[LLM_QUESTION] OpenAI question generation failed: {str(e)}")
                llm_error = e
        
        # Try Anthropic as fallback
        if not question_text and has_anthropic:
            try:
                model = "claude-3-sonnet-20240229"
                logger.info(f"[LLM_QUESTION] Using Anthropic model {model} for question generation")
                start_time = time.time()
                question_text = await query_llm(
                    prompt=prompt, 
                    max_tokens=150,  # Allow for longer, more detailed questions
                    model=model,
                    system_prompt="You are an expert survey designer creating questions to collect maximum data in minimum time."
                )
                elapsed = time.time() - start_time
                
                if question_text:
                    logger.info(f"[LLM_QUESTION] Anthropic returned response in {elapsed:.3f}s, length: {len(question_text)}")
                    logger.debug(f"[LLM_QUESTION] Anthropic response: '{question_text[:100]}...'")
                else:
                    logger.warning(f"[LLM_QUESTION] Anthropic returned empty response after {elapsed:.3f}s")
            except Exception as e:
                logger.warning(f"[LLM_QUESTION] Anthropic question generation failed: {str(e)}")
                if llm_error is None:
                    llm_error = e
        
        # If both failed, use improved fallback question
        if not question_text:
            if llm_error:
                logger.error(f"[LLM_QUESTION] All LLM providers failed: {str(llm_error)}")
            
            # Create a more specific fallback based on the strategy
            fallback = {
                "question_text": f"Based on what you've shared, could you elaborate on how {specific_focus} specifically impacts your {metric_id.replace('_', ' ')}?",
                "metric_id": metric_id,
                "question_id": f"fallback_{int(datetime.now().timestamp())}",
                "source": "fallback",
                "strategy": "fallback",
                "type": "text"
            }
            logger.info(f"[LLM_QUESTION] Returning LLM failure fallback question")
            return fallback
        
        # Generate a valid question ID
        question_id = f"llm_{int(datetime.now().timestamp())}"
        
        # Format the response
        result = {
            "question_text": question_text.strip(),
            "metric_id": metric_id,
            "question_id": question_id,
            "source": "llm_with_examples",
            "strategy": strategy.get("recommended_approach"),
            "type": "text"
        }
        
        logger.info(f"[LLM_QUESTION] Successfully generated question: '{result['question_text'][:50]}...'")
        return result
    except Exception as e:
        logger.error(f"[LLM_QUESTION] Error generating question with LLM: {str(e)}")
        fallback = {
            "question_text": f"How does {specific_focus} affect your {metric_id.replace('_', ' ')}?",
            "metric_id": metric_id,
            "question_id": f"fallback_{int(datetime.now().timestamp())}",
            "source": "fallback",
            "strategy": "fallback",
            "type": "text"
        }
        logger.info(f"[LLM_QUESTION] Returning error fallback question")
        return fallback

async def store_new_question_pattern(
    session_vector: List[float],
    question: Dict[str, Any],
    strategy: Dict[str, Any]
):
    """Store a new question pattern in the vector database.
    
    Args:
        session_vector: The session vector
        question: The generated question
        strategy: The strategy used to generate the question
    """
    try:
        client = await get_qdrant_client()
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        
        # Generate a valid UUID for the point ID
        point_id = str(uuid.uuid4())
        
        # Get session_id for tracking origin
        session_id = question.get("session_id", "")
        if not session_id and "session_state" in question:
            session_id = question.get("session_state", {}).get("session_id", "")
        
        # Prepare payload
        payload = {
            "question_id": question.get("question_id", f"gen_{int(datetime.now().timestamp())}"),
            "question_text": question["question_text"],
            "metric_id": question["metric_id"],
            "question_type": question.get("type", "text"),
            "category": strategy.get("category", "general"),
            "usage_count": 1,
            "success_rate": 0.5,  # Initial value
            "last_used": datetime.now().isoformat(),
            "generated_by": "llm",
            "effectiveness_history": [],
            "session_id": session_id  # Add session_id to track origin
        }
        
        # Store in vector database - QdrantClient methods are synchronous, don't use await
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": session_vector,
                "payload": payload
            }]
        )
    except Exception as e:
        logger.error(f"Error storing new question pattern: {str(e)}")
        # Log the raw response if available
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"Raw response content:\n{e.response.content}")
        raise  # Re-raise to let the caller handle it

async def update_vector_success_rate(
    question_id: str,
    success_score: float,
    response: Dict[str, Any]
):
    """Update the success rate of a question in the vector database.
    
    Args:
        question_id: The question ID
        success_score: The success score
        response: The user's response
    """
    try:
        client = await get_qdrant_client()
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        
        # Get current payload using search by payload
        filter_condition = {
            "must": [
                {
                    "key": "question_id",
                    "match": {
                        "value": question_id
                    }
                }
            ]
        }
        
        # QdrantClient methods are synchronous, don't use await
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=1
        )
        
        if not results or not results[0].payload:
            logger.warning(f"Question ID {question_id} not found in vector database")
            return
            
        point = results[0]
        payload = point.payload
        
        # Update usage count and success rate
        usage_count = payload.get("usage_count", 0) + 1
        history = payload.get("effectiveness_history", [])
        history.append(success_score)
        
        # Calculate new success rate
        new_success_rate = sum(history) / len(history) if history else 0.5
        
        # Update payload
        new_payload = {
            **payload,
            "usage_count": usage_count,
            "success_rate": new_success_rate,
            "last_used": datetime.now().isoformat(),
            "effectiveness_history": history
        }
        
        # Update in vector database - QdrantClient methods are synchronous, don't use await
        client.update_payload(
            collection_name=collection_name,
            points=[{
                "id": point.id,
                "payload": new_payload
            }]
        )
    except Exception as e:
        logger.error(f"Error updating vector success rate: {str(e)}")
        # Log the raw response if available
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"Raw response content:\n{e.response.content}")
        raise  # Re-raise to let the caller handle it

async def store_successful_question(
    session_vector: List[float],
    question: Dict[str, Any],
    response: Dict[str, Any],
    effectiveness: float
):
    """Store a successful LLM-generated question in the vector database.
    
    Args:
        session_vector: The session vector
        question: The question
        response: The user's response
        effectiveness: The effectiveness score
    """
    try:
        client = await get_qdrant_client()
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        
        # Generate a valid UUID for the point ID
        point_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = {
            "question_id": question.get("question_id", f"gen_{int(datetime.now().timestamp())}"),
            "question_text": question.get("question_text", ""),
            "metric_id": question.get("metric_id", ""),
            "question_type": question.get("type", "text"),
            "category": "successful_llm_generation",
            "usage_count": 1,
            "success_rate": effectiveness,
            "last_used": datetime.now().isoformat(),
            "generated_by": "llm",
            "effectiveness_history": [effectiveness]
        }
        
        # Store in vector database - QdrantClient methods are synchronous, don't use await
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": point_id,
                "vector": session_vector,
                "payload": payload
            }]
        )
    except Exception as e:
        logger.error(f"Error storing successful question: {str(e)}")
        # Log the raw response if available
        if hasattr(e, 'response') and hasattr(e.response, 'content'):
            logger.error(f"Raw response content:\n{e.response.content}")
        raise  # Re-raise to let the caller handle it

# Utility functions

def format_conversation_history(session_state: Dict[str, Any]) -> str:
    """Format the conversation history for LLM context.
    
    Args:
        session_state: The session state
        
    Returns:
        Formatted conversation history
    """
    history = session_state.get("question_history", [])
    formatted = []
    
    for item in history:
        formatted.append(f"Q: {item.get('text', '')}")
        formatted.append(f"A: {item.get('response_text', '')}")
    
    return "\n".join(formatted)

def format_recent_conversation(session_state: Dict[str, Any], max_turns: int = 3) -> str:
    """Format the most recent conversation turns for LLM context.
    
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
    """Format the metrics status for LLM context.
    
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
            f"Confidence={metric_data.get('confidence', 0):.2f}, "
            f"Questions Asked={metric_data.get('questions_asked', 0)}"
        )
    
    return "\n".join(formatted)

def format_vector_examples(examples: List[Dict[str, Any]]) -> str:
    """Format vector examples for LLM context.
    
    Args:
        examples: List of vector examples
        
    Returns:
        Formatted vector examples
    """
    if not examples:
        return "No similar past interactions found."
        
    formatted = []
    for i, example in enumerate(examples):
        formatted.append(f"Example {i+1}:")
        formatted.append(f"Question: {example.get('question_text', 'No question')}")
        formatted.append(f"Metric: {example.get('metric_id', 'Unknown')}")
        formatted.append(f"Success Rate: {example.get('success_rate', 'Unknown')}")
        formatted.append("")
    
    return "\n".join(formatted)

def get_highest_priority_metric(session_state: Dict[str, Any]) -> str:
    """Get the highest priority metric to focus on.
    
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
        key=lambda x: (x[1].get("confidence", 1), x[1].get("questions_asked", 0))
    )
    
    return sorted_metrics[0][0] if sorted_metrics else pending[0]

def get_metric_description(metric_id: str) -> str:
    """Get a description for a metric.
    
    Args:
        metric_id: The metric ID
        
    Returns:
        Description of the metric
    """
    # TODO: Get this from a proper metadata store
    descriptions = {
        "sleep_quality": "How well the user sleeps, including duration and disturbances",
        "stress_level": "The user's perceived stress in daily life",
        "work_life_balance": "How well the user balances professional and personal life",
        "exercise_frequency": "How often the user engages in physical activity",
        "satisfaction": "Overall satisfaction with the program or service",
        "likelihood_to_recommend": "Likelihood of recommending the program to others"
    }
    
    return descriptions.get(metric_id, f"Assessment of {metric_id.replace('_', ' ')}")

def calculate_question_effectiveness(
    question: Dict[str, Any],
    response: Dict[str, Any],
    metrics_update: Dict[str, Any],
    user_satisfaction: Optional[float] = None
) -> float:
    """Calculate the effectiveness of a question based on metrics update.
    
    Args:
        question: The question
        response: The user's response
        metrics_update: Changes to metrics after the response
        user_satisfaction: Optional user satisfaction rating
        
    Returns:
        Effectiveness score between 0 and 1
    """
    # Calculate base effectiveness from metrics update
    metric_id = question.get("metric_id", "")
    
    if not metric_id or metric_id not in metrics_update:
        # If no metrics were updated, low effectiveness
        base_effectiveness = 0.3
    else:
        # Calculate effectiveness based on confidence change
        confidence_before = metrics_update.get(metric_id, {}).get("confidence_before", 0)
        confidence_after = metrics_update.get(metric_id, {}).get("confidence_after", confidence_before)
        
        # If confidence improved significantly, high effectiveness
        confidence_improvement = max(0, confidence_after - confidence_before)
        base_effectiveness = min(1.0, 0.5 + confidence_improvement)
    
    # Consider user satisfaction if available
    if user_satisfaction is not None:
        # Scale satisfaction to 0-1
        satisfaction_factor = user_satisfaction / 5.0 if user_satisfaction <= 5 else 1.0
        
        # Combine base effectiveness with satisfaction (weighted)
        effectiveness = 0.7 * base_effectiveness + 0.3 * satisfaction_factor
    else:
        effectiveness = base_effectiveness
    
    return min(1.0, max(0.0, effectiveness))

async def get_session(session_id: str) -> Dict[str, Any]:
    """Get session state from the database.
    
    Args:
        session_id: The session ID
        
    Returns:
        Session state or None if not found
    """
    # TODO: Implement actual session retrieval
    try:
        redis = await get_redis()
        session_data = await redis.get(f"session:{session_id}")
        
        if session_data:
            return json.loads(session_data)
        
        return None
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        return None

def extract_asked_questions(session_state: Dict[str, Any]) -> List[str]:
    """Extract previously asked questions from the session history.
    
    Args:
        session_state: The session state
        
    Returns:
        List of question texts
    """
    history = session_state.get("question_history", [])
    return [item.get("text", "") for item in history if "text" in item]

def extract_question_topics(questions: List[str]) -> List[str]:
    """Extract main topics from a list of questions.
    This is a simple implementation that could be enhanced with NLP.
    
    Args:
        questions: List of question texts
        
    Returns:
        List of main topics
    """
    # A simple implementation that just extracts key words
    topics = set()
    
    # Common stop words to filter out
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "with", "by", "about", 
                  "how", "what", "when", "where", "why", "who", "which", "can", "could", 
                  "would", "should", "me", "you", "your", "tell", "describe", "explain"}
    
    for question in questions:
        # Simple tokenization by splitting on whitespace
        words = question.lower().replace("?", "").replace(".", "").replace(",", "").split()
        
        # Filter out stop words and short words
        for word in words:
            if word not in stop_words and len(word) > 3:
                topics.add(word)
    
    return list(topics)

def is_question_too_similar(new_question: str, asked_questions: List[str], 
                          similarity_threshold: float = 0.55) -> bool:
    """Check if a new question is too similar to previously asked questions.
    This is a simple implementation that could be enhanced with NLP.
    
    Args:
        new_question: The new question text
        asked_questions: List of previously asked question texts
        similarity_threshold: Threshold for similarity (higher = more strict)
        
    Returns:
        True if the question is too similar to any previous question
    """
    # A simple token overlap implementation
    new_tokens = set(new_question.lower().replace("?", "").replace(".", "").replace(",", "").split())
    
    # Track similarity scores for logging
    similarities = []
    
    for asked in asked_questions:
        asked_tokens = set(asked.lower().replace("?", "").replace(".", "").replace(",", "").split())
        
        # Calculate Jaccard similarity
        if not new_tokens or not asked_tokens:
            continue
            
        intersection = len(new_tokens.intersection(asked_tokens))
        union = len(new_tokens.union(asked_tokens))
        
        similarity = intersection / union if union > 0 else 0
        similarities.append((asked, similarity))
        
        if similarity > similarity_threshold:
            logger.info(f"Question similarity {similarity:.2f} > threshold {similarity_threshold}: '{new_question}' vs '{asked}'")
            return True
    
    # Log the top similarity for debugging
    if similarities:
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_match, top_score = similarities[0]
        logger.debug(f"Top question similarity: {top_score:.2f} for '{new_question}' vs '{top_match}'")
    
    return False

def summarize_conversation(session_state: Dict[str, Any]) -> str:
    """Create a concise summary of the conversation so far.
    
    Args:
        session_state: The session state
        
    Returns:
        A summary of the conversation
    """
    history = session_state.get("question_history", [])
    
    if not history:
        return "No conversation history yet."
    
    summary_parts = []
    
    # Group by metrics
    metric_conversations = {}
    
    for item in history:
        metric_id = item.get("metric_id", "general")
        
        if metric_id not in metric_conversations:
            metric_conversations[metric_id] = []
            
        metric_conversations[metric_id].append({
            "question": item.get("text", ""),
            "response": item.get("response_text", "")
        })
    
    # Create summary for each metric
    for metric_id, conversations in metric_conversations.items():
        metric_summary = f"About {metric_id.replace('_', ' ')}:"
        
        for convo in conversations:
            summary = f"When asked '{convo['question']}', user responded: '{convo['response'][:100]}...'"
            metric_summary += f"\n- {summary}"
            
        summary_parts.append(metric_summary)
    
    return "\n\n".join(summary_parts)

def analyze_metrics(session_state: Dict[str, Any]) -> str:
    """Analyze the current metrics to identify gaps and priorities.
    
    Args:
        session_state: The session state
        
    Returns:
        Analysis of metrics
    """
    metrics = session_state.get("metrics", {})
    pending = session_state.get("metrics_pending", [])
    
    if not metrics:
        return "No metrics data available."
    
    analysis_parts = []
    
    # Analyze each metric
    for metric_id, data in metrics.items():
        score = data.get("score", 0)
        confidence = data.get("confidence", 0)
        questions_asked = data.get("questions_asked", 0)
        
        status = "pending" if metric_id in pending else "complete"
        
        analysis = f"{metric_id.replace('_', ' ')}: "
        analysis += f"Score={score:.1f}, Confidence={confidence:.1f}, Questions={questions_asked}, Status={status}"
        
        # Add insights
        if confidence < 0.4:
            analysis += " - Need more information"
        elif score < 0.3:
            analysis += " - Low score area"
        elif score > 0.7:
            analysis += " - High score area"
            
        analysis_parts.append(analysis)
    
    return "\n".join(analysis_parts)

async def create_placeholder_session_vector(session_state: Dict[str, Any]) -> List[float]:
    """Create a placeholder vector representation of the current session state.
    
    Args:
        session_state: The current session state
        
    Returns:
        A vector representation of the session
    """
    import numpy as np
    import time
    from hashlib import md5
    
    # Initialize with the appropriate dimension
    vector_dim = settings.VECTOR_DIMENSIONS
    
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

# Extracted function for vector search to enable parallel processing
async def perform_vector_search(session_vector, session_state):
    """Perform vector search operations asynchronously."""
    target_metric = get_highest_priority_metric(session_state)
    session_id = session_state.get("session_id", "unknown")
    history_len = len(session_state.get("question_history", []))
    
    logger.info(f"[VECTOR_SEARCH] Starting vector search for session {session_id}, focused on metric: {target_metric}")
    
    # First try to get from cache
    redis = await get_redis()
    cache_key = f"vector_search_{target_metric}_{history_len}"
    
    logger.debug(f"[VECTOR_SEARCH] Checking cache with key: {cache_key}")
    cached_results = await redis.get(cache_key)
    
    if cached_results:
        logger.info(f"[VECTOR_SEARCH] Cache hit! Using cached vector search results")
        results = json.loads(cached_results)
        logger.debug(f"[VECTOR_SEARCH] Loaded {len(results)} cached results")
        return results
    
    logger.debug(f"[VECTOR_SEARCH] Cache miss. Performing fresh vector search")
    
    try:
        # Extract question IDs from the current session to exclude them
        current_session_question_ids = []
        for question in session_state.get("question_history", []):
            if "question_id" in question:
                current_session_question_ids.append(question["question_id"])
            elif "id" in question:
                current_session_question_ids.append(question["id"])
        
        # Build filter that includes target metric AND excludes questions from current session
        if current_session_question_ids:
            logger.debug(f"[VECTOR_SEARCH] Will exclude {len(current_session_question_ids)} questions from current session")
            metric_filter = {
                "must": [
                    {
                        "key": "metric_id",
                        "match": {
                            "value": target_metric
                        }
                    }
                ],
                "must_not": [
                    {
                        "key": "question_id",
                        "match": {
                            "any": current_session_question_ids
                        }
                    },
                    {
                        "key": "session_id",
                        "match": {
                            "value": session_id
                        }
                    }
                ]
            }
        else:
            # If no questions in history yet, just filter by metric
            metric_filter = {
                "must": [
                    {
                        "key": "metric_id",
                        "match": {
                            "value": target_metric
                        }
                    }
                ],
                "must_not": [
                    {
                        "key": "session_id",
                        "match": {
                            "value": session_id
                        }
                    }
                ]
            }
        
        logger.debug(f"[VECTOR_SEARCH] Using metric filter with session exclusion: {metric_filter}")
        
        collection_name = settings.QDRANT_QUESTION_PATTERNS_COLLECTION
        
        # Use the async search function from the qdrant module
        from app.db.qdrant import search as qdrant_search
        
        # Run both searches in parallel using asyncio.gather
        logger.debug(f"[VECTOR_SEARCH] Starting metric-specific and general searches")
        search_start_time = time.time()
        
        # Create general filter that also excludes current session questions
        general_filter = {
            "must_not": [
                {
                    "key": "session_id",
                    "match": {
                        "value": session_id
                    }
                }
            ]
        }
        
        if current_session_question_ids:
            general_filter["must_not"].append({
                "key": "question_id",
                "match": {
                    "any": current_session_question_ids
                }
            })
        
        # Use asyncio.gather with the async search wrapper function
        metric_search_task = qdrant_search(
            collection_name=collection_name,
            query_vector=session_vector,
            query_filter=metric_filter,
            limit=2,
            score_threshold=0.80
        )
        
        general_search_task = qdrant_search(
            collection_name=collection_name,
            query_vector=session_vector,
            query_filter=general_filter,
            limit=3,
            score_threshold=0.80
        )
        
        # Wait for both searches to complete
        metric_matches, general_matches = await asyncio.gather(
            metric_search_task, 
            general_search_task
        )
        
        search_elapsed = time.time() - search_start_time
        logger.info(f"[VECTOR_SEARCH] Searches completed in {search_elapsed:.3f}s")
        logger.debug(f"[VECTOR_SEARCH] Results: metric={len(metric_matches)} matches, general={len(general_matches)} matches")
        
        # Combine and deduplicate matches
        seen_ids = set()
        vector_matches = []
        
        # Priority to metric-specific matches
        for match in metric_matches:
            if match.id not in seen_ids:
                vector_matches.append(match)
                seen_ids.add(match.id)
                logger.debug(f"[VECTOR_SEARCH] Added metric match: score={match.score}, id={match.id}")
        
        # Then add general matches if needed
        for match in general_matches:
            if match.id not in seen_ids and len(vector_matches) < 4:
                vector_matches.append(match)
                seen_ids.add(match.id)
        
        # Cache results for 2 minutes (relatively short TTL)
        if vector_matches:
            # Convert to serializable format before caching
            serializable_matches = [
                {
                    "id": str(match.id),
                    "score": match.score,
                    "payload": match.payload
                } for match in vector_matches
            ]
            await redis.set(cache_key, json.dumps(serializable_matches), ex=120)
            
        return vector_matches
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return []

# New deterministic strategy function
async def deterministic_strategy(session_state, vector_examples, asked_questions):
    """Determine questioning strategy based on question count.
    
    For sessions with <10 questions, always use custom_generation.
    For sessions with ≥10 questions, always use reuse_past_pattern (vector search).
    
    Args:
        session_state: Current session state
        vector_examples: Similar past interactions from vector search
        asked_questions: Previously asked questions to avoid repetition
        
    Returns:
        Strategy recommendation with approach and target metric
    """
    # Get question history count
    history_len = len(session_state.get("question_history", []))
    target_metric = get_highest_priority_metric(session_state)
    
    # Deterministic decision based on question count
    '''
    if history_len < 10:
        # For early questions, always use custom generation
        strategy = {
            "recommended_approach": "custom_generation",
            "target_metric": target_metric,
            "specific_focus": "general experience",
            "question_format": "open-ended",
            "rationale": "Deterministic strategy - early in session (<10 questions), using custom generation"
        }
        logger.info(f"[DETERMINISTIC] Question count {history_len} < 10: Using custom_generation")
    else:
        # For later questions, always use vector search
        strategy = {
            "recommended_approach": "reuse_past_pattern",
            "target_metric": target_metric,
            "specific_focus": "prior successes",
            "question_format": "proven patterns",
            "rationale": "Deterministic strategy - later in session (≥10 questions), using vector patterns"
        }
        logger.info(f"[DETERMINISTIC] Question count {history_len} >= 10: Using reuse_past_pattern")
    '''

    strategy = {
            "recommended_approach": "custom_generation",
            "target_metric": target_metric,
            "specific_focus": "general experience",
            "question_format": "open-ended",
            "rationale": "Deterministic strategy - early in session (<10 questions), using custom generation"
        }
    logger.info(f"[DETERMINISTIC] Using custom_generation")
    
    return strategy

# Extracted function for strategy determination to enable parallel processing
async def determine_strategy(session_state, vector_examples, asked_questions):
    """Determine questioning strategy, potentially in parallel with other operations."""
    # This is the original LLM-based strategy determination, kept for reference but no longer used
    # Try to get from cache first for similar sessions
    session_id = session_state.get("session_id", "")
    history_len = len(session_state.get("question_history", []))
    
    # Only cache strategies for sessions with some history
    if history_len > 0:
        redis = await get_redis()
        cache_key = f"strategy_{session_id}_{history_len}"
        cached_strategy = await redis.get(cache_key)
        
        if cached_strategy:
            logger.info("Using cached strategy")
            return json.loads(cached_strategy)
    
    # If not cached, determine strategy
    strategy = await llm_determine_strategy(
        session_state=session_state,
        vector_examples=vector_examples,
        asked_questions=asked_questions
    )
    
    # Cache the strategy if we have some history (5 minute TTL)
    if history_len > 0:
        redis = await get_redis()
        await redis.set(f"strategy_{session_id}_{history_len}", json.dumps(strategy), ex=300)
    
    return strategy 