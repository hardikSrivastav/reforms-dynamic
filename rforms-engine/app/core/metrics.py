from typing import Dict, Any, List, Optional
from loguru import logger
from app.core.scoring import get_confidence_threshold
from app.db.postgres import get_db_conn
from app.db.redis import get_cache, set_cache
import json

async def get_metric_details(metric_id: str) -> Optional[Dict[str, Any]]:
    """Get details for a specific metric.
    
    Args:
        metric_id: The ID of the metric
        
    Returns:
        The metric details or None if not found
    """
    # Try to get from cache first
    cache_key = f"metric:{metric_id}"
    cached_metric = await get_cache(cache_key)
    if cached_metric:
        return cached_metric
    
    # If not in cache, get from database
    metric_dict = None
    conn = await get_db_conn()
    try:
        query = """
        SELECT id, name, description, importance, min_questions, max_questions
        FROM metrics
        WHERE id = $1
        """
        metric = await conn.fetchrow(query, metric_id)
        
        if metric:
            metric_dict = {
                "id": metric['id'],
                "name": metric['name'],
                "description": metric['description'],
                "importance": metric['importance'],
                "min_questions": metric['min_questions'],
                "max_questions": metric['max_questions']
            }
            
            # Get dependencies
            dep_query = """
            SELECT depends_on_metric_id, weight
            FROM metric_dependencies
            WHERE metric_id = $1
            """
            deps = await conn.fetch(dep_query, metric_id)
            
            metric_dict["dependencies"] = [
                {"metric_id": d['depends_on_metric_id'], "weight": d['weight']} for d in deps
            ]
            
            # Cache for future use (1 hour TTL)
            await set_cache(cache_key, metric_dict, 3600)
    except Exception as e:
        logger.error(f"Error fetching metric details: {e}")
    finally:
        await conn.close()
    
    return metric_dict

async def get_survey_metrics(survey_id: str) -> List[Dict[str, Any]]:
    """Get all metrics for a survey.
    
    Args:
        survey_id: The ID of the survey
        
    Returns:
        List of metrics
    """
    # Try to get from cache first
    cache_key = f"survey_metrics:{survey_id}"
    cached_metrics = await get_cache(cache_key)
    if cached_metrics:
        return cached_metrics
    
    metrics = []
    conn = await get_db_conn()
    try:
        query = """
        SELECT id, name, description, importance, min_questions, max_questions
        FROM metrics
        WHERE survey_id = $1
        ORDER BY importance DESC
        """
        rows = await conn.fetch(query, survey_id)
        
        for row in rows:
            metric = {
                "id": row['id'],
                "name": row['name'],
                "description": row['description'],
                "importance": row['importance'],
                "min_questions": row['min_questions'],
                "max_questions": row['max_questions']
            }
            metrics.append(metric)
        
        # Cache for future use (1 hour TTL)
        await set_cache(cache_key, metrics, 3600)
    
    except Exception as e:
        logger.error(f"Error fetching survey metrics: {e}")
    finally:
        await conn.close()
    
    return metrics

async def get_metric_dependencies(metric_id: str) -> List[Dict[str, Any]]:
    """Get dependencies for a metric.
    
    Args:
        metric_id: The ID of the metric
        
    Returns:
        List of metric dependencies
    """
    dependencies = []
    conn = await get_db_conn()
    try:
        query = """
        SELECT md.depends_on_metric_id, md.weight, m.name
        FROM metric_dependencies md
        JOIN metrics m ON md.depends_on_metric_id = m.id
        WHERE md.metric_id = $1
        """
        rows = await conn.fetch(query, metric_id)
        
        for row in rows:
            dependency = {
                "metric_id": row['depends_on_metric_id'],
                "weight": row['weight'],
                "name": row['name']
            }
            dependencies.append(dependency)
                
    except Exception as e:
        logger.error(f"Error fetching metric dependencies: {e}")
    finally:
        await conn.close()
    
    return dependencies

async def is_metric_complete(metric_id: str, session_state: Dict[str, Any]) -> bool:
    """Check if a metric has been sufficiently assessed.
    
    Args:
        metric_id: The ID of the metric
        session_state: The current session state
        
    Returns:
        True if the metric is complete, False otherwise
    """
    metrics = session_state.get("metrics", {})
    metric_data = metrics.get(metric_id, {})
    
    # No data yet for this metric
    if not metric_data:
        return False
    
    # Check if minimum questions have been asked
    questions_asked = metric_data.get("questions_asked", 0)
    metric_details = await get_metric_details(metric_id)
    
    if not metric_details:
        return False
    
    min_questions = metric_details.get("min_questions", 2)
    max_questions = metric_details.get("max_questions", 5)
    
    # If we haven't asked the minimum number of questions, not complete
    if questions_asked < min_questions:
        return False
    
    # If we've asked the maximum number of questions, complete
    if questions_asked >= max_questions:
        return True
    
    # Check confidence against threshold
    confidence = metric_data.get("confidence", 0)
    # Don't await this function since it's not async
    thresholds = get_confidence_threshold(metric_id, session_state)
    
    # If confidence exceeds high threshold, complete
    if confidence >= thresholds.get("high", 0.75):
        return True
    
    # Not enough confidence yet
    return False

async def calculate_expected_information_gain(metric_id: str, session_state: Dict[str, Any]) -> float:
    """Calculate the expected information gain from asking about a metric.
    
    Args:
        metric_id: The ID of the metric
        session_state: The current session state
        
    Returns:
        Expected information gain (0-1)
    """
    metrics = session_state.get("metrics", {})
    metric_data = metrics.get(metric_id, {})
    
    # Current confidence
    current_confidence = metric_data.get("confidence", 0)
    
    # Get metric details
    metric_details = await get_metric_details(metric_id)
    if not metric_details:
        return 0.5  # Default value
    
    # Metric importance
    importance = metric_details.get("importance", 1.0)
    
    # Questions asked
    questions_asked = metric_data.get("questions_asked", 0)
    max_questions = metric_details.get("max_questions", 5)
    
    # Diminishing returns factor
    question_factor = max(0.2, 1.0 - (questions_asked / max_questions))
    
    # Information still needed
    information_needed = 1.0 - current_confidence
    
    # Combine factors
    return importance * question_factor * information_needed

async def count_satisfied_dependencies(metric_id: str, session_state: Dict[str, Any]) -> int:
    """Count how many dependencies are satisfied for a metric.
    
    Args:
        metric_id: The ID of the metric
        session_state: The current session state
        
    Returns:
        Number of satisfied dependencies
    """
    dependencies = await get_metric_dependencies(metric_id)
    
    if not dependencies:
        return 0  # No dependencies
    
    satisfied = 0
    for dep in dependencies:
        dep_id = dep.get("metric_id")
        dep_weight = dep.get("weight", 1.0)
        
        # Check if dependency is complete
        if await is_metric_complete(dep_id, session_state):
            satisfied += 1
    
    return satisfied

async def prioritize_metrics_for_questioning(session_state: Dict[str, Any]) -> List[str]:
    """Order metrics by priority for questioning.
    
    Args:
        session_state: The current session state
        
    Returns:
        Ordered list of metric IDs
    """
    # Get metrics that still need assessment
    metrics = []
    
    if "metrics_pending" in session_state:
        # Use pre-defined pending metrics list
        metrics = session_state["metrics_pending"]
    else:
        # Use all metrics from survey
        survey_id = session_state.get("survey_id")
        if survey_id:
            survey_metrics = await get_survey_metrics(survey_id)
            metrics = [m["id"] for m in survey_metrics]
    
    # Calculate prioritization factors for each metric
    metric_priorities = []
    for metric_id in metrics:
        # Skip already completed metrics
        if await is_metric_complete(metric_id, session_state):
            continue
        
        # Information gain
        info_gain = await calculate_expected_information_gain(metric_id, session_state)
        
        # Questions already asked
        questions_asked = session_state.get("metrics", {}).get(metric_id, {}).get("questions_asked", 0)
        
        # Current confidence
        confidence = session_state.get("metrics", {}).get(metric_id, {}).get("confidence", 0)
        
        # Dependency satisfaction
        deps_satisfied = await count_satisfied_dependencies(metric_id, session_state)
        
        # Get metric details
        metric_details = await get_metric_details(metric_id)
        importance = metric_details.get("importance", 1.0) if metric_details else 1.0
        
        # Store metrics with their priority factors
        metric_priorities.append({
            "metric_id": metric_id,
            "info_gain": info_gain,
            "questions_asked": questions_asked,
            "confidence": confidence,
            "deps_satisfied": deps_satisfied,
            "importance": importance
        })
    
    # Sort based on multiple factors
    sorted_metrics = sorted(metric_priorities, key=lambda m: (
        # 1. Dependencies satisfied (more = better)
        -m["deps_satisfied"],
        # 2. Information value (higher = better)
        -m["info_gain"],
        # 3. Questions already asked (fewer = better)
        m["questions_asked"],
        # 4. Current confidence (lower = better)
        m["confidence"]
    ))
    
    # Return just the metric IDs in priority order
    return [m["metric_id"] for m in sorted_metrics]

async def select_target_metric_for_llm(session_state: Dict[str, Any]) -> Optional[str]:
    """Select the best metric to target with an LLM-generated question.
    
    Args:
        session_state: The current session state
        
    Returns:
        The selected metric ID or None
    """
    # Simple implementation: just get the highest priority metric
    priorities = await prioritize_metrics_for_questioning(session_state)
    
    if not priorities:
        return None
    
    return priorities[0]

def get_preferred_question_type(metric_id: str) -> str:
    """Get the preferred question type for a metric.
    
    Args:
        metric_id: The ID of the metric
        
    Returns:
        Preferred question type
    """
    # This is a placeholder - in a real implementation, this would be stored in the metric
    # metadata or learned from interaction patterns
    
    # Simple mapping based on metric naming conventions
    metric_id_lower = metric_id.lower()
    
    if "satisfaction" in metric_id_lower or "rating" in metric_id_lower:
        return "likert"
    elif "frequency" in metric_id_lower or "often" in metric_id_lower:
        return "multiple_choice"
    elif "amount" in metric_id_lower or "count" in metric_id_lower:
        return "number"
    elif "interest" in metric_id_lower or "preference" in metric_id_lower:
        return "text"
    elif "aware" in metric_id_lower or "heard" in metric_id_lower:
        return "boolean"
    
    # Default to open text
    return "text"

def get_unresolved_metrics_names(session_state: Dict[str, Any]) -> List[str]:
    """Get names of metrics that still need assessment.
    
    Args:
        session_state: The current session state
        
    Returns:
        List of metric names
    """
    # This would normally fetch from the database, but for MVP we'll use a simpler approach
    metrics_pending = session_state.get("metrics_pending", [])
    return [m.replace("_", " ").title() for m in metrics_pending]

async def get_metrics_for_survey(survey_id: str) -> Dict[str, Any]:
    """Get all metrics for a survey, formatted as a dictionary.
    
    Args:
        survey_id: The ID of the survey
        
    Returns:
        Dictionary of metrics indexed by metric ID
    """
    # Get the metrics list from the existing function
    metrics_list = await get_survey_metrics(survey_id)
    
    # Convert to a dictionary keyed by ID
    metrics_dict = {}
    for metric in metrics_list:
        metric_id = metric.get("id")
        if metric_id:
            metrics_dict[metric_id] = metric
    
    return metrics_dict 