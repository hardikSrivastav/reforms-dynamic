from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from app.db.mongodb import get_decision_trees_collection
from app.db.redis import get_cache, set_cache
import json

async def get_decision_tree(metric_id: str) -> Optional[Dict[str, Any]]:
    """Get the decision tree for a specific metric.
    
    Args:
        metric_id: The ID of the metric
        
    Returns:
        The decision tree object or None if not found
    """
    # Try to get from cache first
    cache_key = f"decision_tree:{metric_id}"
    cached_tree = await get_cache(cache_key)
    if cached_tree:
        return cached_tree
    
    # If not in cache, get from database
    collection = get_decision_trees_collection()
    tree = await collection.find_one({"metric_id": metric_id})
    
    if tree:
        # Remove MongoDB _id for serialization
        if "_id" in tree:
            del tree["_id"]
        
        # Cache for future use (1 hour TTL)
        await set_cache(cache_key, tree, 3600)
        
    return tree

def determine_current_node(
    tree: Dict[str, Any], 
    metric_id: str, 
    session_state: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Determine the current position in the decision tree based on question history.
    
    Args:
        tree: The decision tree
        metric_id: The ID of the metric
        session_state: The current session state
        
    Returns:
        The current node in the tree or None if at end
    """
    # Get metric-specific question history
    history = session_state.get("question_history", [])
    metric_history = [q for q in history if q.get("metric_id") == metric_id]
    
    if not metric_history:
        # No questions asked yet, start at root
        return tree["nodes"].get(tree["root"])
    
    # Get the most recent question for this metric
    last_question = metric_history[-1]
    last_node_id = last_question.get("id")
    
    if last_node_id not in tree["nodes"]:
        logger.warning(f"Node {last_node_id} not found in tree for metric {metric_id}")
        return None
    
    last_node = tree["nodes"][last_node_id]
    
    # Check if we have a next node based on the response
    if "next" not in last_node:
        return None  # End of tree
        
    response = last_question.get("response_text", "")
    
    # Find the matching branch for the response
    next_node_id = select_branch(last_node, response)
    
    if not next_node_id:
        return None  # No matching branch
        
    return tree["nodes"].get(next_node_id)

def select_branch(node: Dict[str, Any], response: str) -> Optional[str]:
    """Select the appropriate branch based on the response.
    
    Args:
        node: The current node
        response: The user's response
        
    Returns:
        The ID of the next node or None if no branch matches
    """
    if "next" not in node:
        return None
        
    next_branches = node["next"]
    
    # For number type questions
    if node["type"] == "number":
        try:
            num_response = float(response)
            
            # Find the appropriate range
            for branch, next_info in next_branches.items():
                if branch.startswith("<") and num_response < float(branch[1:]):
                    return next_info.get("node")
                elif branch.startswith(">") and num_response > float(branch[1:]):
                    return next_info.get("node")
                elif "-" in branch:
                    min_val, max_val = map(float, branch.split("-"))
                    if min_val <= num_response <= max_val:
                        return next_info.get("node")
        except (ValueError, TypeError):
            # Not a number, try string matching
            pass
    
    # Direct response matching (for multiple_choice, boolean, likert)
    if response in next_branches:
        return next_branches[response].get("node")
    
    # Case-insensitive matching
    for branch, next_info in next_branches.items():
        if branch.lower() == response.lower():
            return next_info.get("node")
    
    # Check for default branch
    if "default" in next_branches:
        return next_branches["default"].get("node")
    
    # No matching branch found
    return None

def check_shortcuts(node: Dict[str, Any], session_state: Dict[str, Any]) -> Optional[Tuple[bool, Dict[str, Any]]]:
    """Check if any shortcuts apply for this node and session state.
    
    Args:
        node: The current decision tree node
        session_state: The current session state
        
    Returns:
        Tuple of (is_shortcut, shortcut_data) or None if no shortcut applies
    """
    if "shortcuts" not in node:
        return None
        
    shortcuts = node["shortcuts"]
    
    for shortcut_key, shortcut_data in shortcuts.items():
        # Parse shortcut condition
        if "_AND_" in shortcut_key:
            conditions = shortcut_key.split("_AND_")
            if all(check_condition(cond, session_state) for cond in conditions):
                return True, shortcut_data
        elif "_OR_" in shortcut_key:
            conditions = shortcut_key.split("_OR_")
            if any(check_condition(cond, session_state) for cond in conditions):
                return True, shortcut_data
        else:
            # Single condition
            if check_condition(shortcut_key, session_state):
                return True, shortcut_data
    
    return None

def check_condition(condition: str, session_state: Dict[str, Any]) -> bool:
    """Check if a condition applies to the current session state.
    
    Args:
        condition: The condition to check
        session_state: The current session state
        
    Returns:
        True if the condition applies, False otherwise
    """
    # These are simplified placeholder implementations
    # In a full system, this would be a more sophisticated condition parser
    
    # Check for comparison conditions
    if "=" in condition:
        key, value = condition.split("=", 1)
        return get_session_value(key, session_state) == value
    elif ">" in condition:
        key, value = condition.split(">", 1)
        try:
            return float(get_session_value(key, session_state)) > float(value)
        except (ValueError, TypeError):
            return False
    elif "<" in condition:
        key, value = condition.split("<", 1)
        try:
            return float(get_session_value(key, session_state)) < float(value)
        except (ValueError, TypeError):
            return False
    
    # Check for presence conditions
    if condition.startswith("has_"):
        key = condition[4:]  # Remove "has_" prefix
        return key in session_state.get("user_profile", {})
    
    # Check for specific text conditions
    if condition.endswith("_mentioned"):
        key = condition[:-10]  # Remove "_mentioned" suffix
        # Check if key was mentioned in any text response
        for q in session_state.get("question_history", []):
            if q.get("response_type") == "text" and key.lower() in q.get("response_text", "").lower():
                return True
        return False
    
    # Default to False for unknown conditions
    return False

def get_session_value(key: str, session_state: Dict[str, Any]) -> Any:
    """Get a value from the session state based on a key path.
    
    Args:
        key: The key path (e.g., "user_profile.age" or "metrics.sleep_quality.score")
        session_state: The current session state
        
    Returns:
        The value or None if not found
    """
    if "." in key:
        parts = key.split(".")
        value = session_state
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    else:
        # Direct key in session_state
        return session_state.get(key)

async def rule_based_question_selection(session_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the next question using only rule-based decision trees.
    
    Args:
        session_state: The current session state
        
    Returns:
        The selected question or None if no rule-based selection is possible
    """
    from app.core.metrics import prioritize_metrics_for_questioning, is_metric_complete
    
    metric_priorities = await prioritize_metrics_for_questioning(session_state)
    
    for metric_id in metric_priorities:
        # Skip metrics that have reached their confidence threshold
        if await is_metric_complete(metric_id, session_state):
            continue
            
        # Get the decision tree for this metric
        tree = await get_decision_tree(metric_id)
        if not tree:
            continue
            
        # Find current position in tree based on question history
        if not session_state.get("question_history"):
            # No questions asked yet, start at root
            current_node = tree["nodes"].get(tree["root"])
            if not current_node:
                continue
                
            return {
                "id": tree["root"],
                "text": current_node["text"],
                "type": current_node["type"],
                "metric_id": metric_id,
                "options": current_node.get("options"),
                "source": "rule_engine"
            }
        else:
            current_node = determine_current_node(tree, metric_id, session_state)
            
            if current_node:
                # Check for shortcuts
                shortcut = check_shortcuts(current_node, session_state)
                if shortcut:
                    is_shortcut, shortcut_data = shortcut
                    if is_shortcut and shortcut_data.get("exit", False):
                        # This is a terminal shortcut, update metric score
                        confidence = shortcut_data.get("confidence", 0.8)
                        value = shortcut_data.get("value")
                        
                        # TODO: Update metric with shortcut value
                        logger.info(f"Shortcut exit for metric {metric_id} with value {value} and confidence {confidence}")
                        
                        # Skip to next metric
                        continue
                
                return {
                    "id": current_node["id"],
                    "text": current_node["text"],
                    "type": current_node["type"],
                    "metric_id": metric_id,
                    "options": current_node.get("options"),
                    "source": "rule_engine",
                    "information_gain": current_node.get("information_gain", 0.5)
                }
    
    return None  # No rule-based decision possible

async def check_tree_shortcuts(session_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check for shortcuts across all decision trees.
    
    Args:
        session_state: The current session state
        
    Returns:
        Shortcut result or None if no shortcut applies
    """
    # For MVP, just check the current metric's tree
    current_metric = session_state.get("current_metric")
    
    if not current_metric:
        return None
        
    tree = await get_decision_tree(current_metric)
    if not tree:
        return None
    
    # Find current position in tree
    current_node = determine_current_node(tree, current_metric, session_state)
    if not current_node:
        return None
    
    # Check for shortcuts
    shortcut_result = check_shortcuts(current_node, session_state)
    if shortcut_result:
        is_shortcut, shortcut_data = shortcut_result
        if is_shortcut:
            return {
                "metric_id": current_metric,
                "confidence": shortcut_data.get("confidence", 0.8),
                "value": shortcut_data.get("value"),
                "exit": shortcut_data.get("exit", False)
            }
    
    return None 