from typing import Dict, Any, List, Optional
from loguru import logger

def calculate_certainty(response: Any, response_type: str) -> float:
    """Calculate response certainty based on response type.
    
    Args:
        response: The user's response value
        response_type: The type of response (likert, number, boolean, text)
        
    Returns:
        A certainty score between 0 and 1
    """
    if response is None:
        return 0.0
    
    if response_type == "likert":
        # Higher certainty for extreme values (Never/Always)
        likert_certainty = {
            "Never": 0.95,
            "Rarely": 0.8,
            "Sometimes": 0.6,
            "Often": 0.8,
            "Always": 0.95,
            "": 0.0  # No response
        }
        return likert_certainty.get(str(response), 0.5)
    
    elif response_type == "number":
        # Higher certainty for specific numbers vs ranges or "I don't know"
        if isinstance(response, (int, float)):
            return 0.9
        elif isinstance(response, str):
            response_lower = response.lower()
            if "about" in response_lower or "around" in response_lower:
                return 0.7
            elif "don't know" in response_lower or "not sure" in response_lower:
                return 0.3
        return 0.5
            
    elif response_type == "boolean":
        # Yes/No are certain, "Maybe" or "Sometimes" less so
        boolean_certainty = {
            "Yes": 0.95,
            "No": 0.95, 
            "True": 0.95,
            "False": 0.95,
            "Maybe": 0.5,
            "Sometimes": 0.6,
            "It depends": 0.3
        }
        return boolean_certainty.get(str(response), 0.5)
        
    elif response_type == "text":
        # Text responses assessed by length, specificity, and sentiment
        return assess_text_certainty(response)
    
    elif response_type == "multiple_choice":
        # All multiple choice answers have high certainty
        return 0.9
    
    return 0.5  # Default certainty

def assess_text_certainty(text: str) -> float:
    """Assess the certainty of a text response.
    
    Args:
        text: The text response
        
    Returns:
        A certainty score between 0 and 1
    """
    if not text:
        return 0.0
        
    # Simple heuristics for text certainty
    text_lower = text.lower()
    
    # Check for uncertain language
    uncertainty_phrases = [
        "i think", "maybe", "possibly", "not sure", "don't know", 
        "might be", "could be", "perhaps", "i guess"
    ]
    
    certainty_phrases = [
        "definitely", "absolutely", "certainly", "always", "never",
        "without a doubt", "i am sure", "i know"
    ]
    
    # Count uncertain and certain phrases
    uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in text_lower)
    certainty_count = sum(1 for phrase in certainty_phrases if phrase in text_lower)
    
    # Base certainty on response length and specific phrases
    base_certainty = min(0.7, 0.3 + (len(text) / 100))
    
    # Adjust for certainty phrases
    phrase_adjustment = 0.1 * (certainty_count - uncertainty_count)
    
    # Clamp final value
    final_certainty = max(0.2, min(0.9, base_certainty + phrase_adjustment))
    
    return final_certainty

def calculate_consistency(
    response: Any, 
    response_type: str,
    previous_responses: List[Dict[str, Any]], 
    metric: str
) -> float:
    """Calculate response consistency with previous responses.
    
    Args:
        response: The user's response value
        response_type: The type of response
        previous_responses: List of previous responses
        metric: The metric being assessed
        
    Returns:
        A consistency score between 0 and 1
    """
    if not previous_responses:
        return 0.7  # No prior responses to compare against
    
    # Get previous responses for this metric
    metric_responses = [
        r for r in previous_responses 
        if r.get("metric_id") == metric
    ]
    
    if not metric_responses:
        return 0.7  # No prior responses for this metric
    
    # TODO: Implement detailed consistency logic (this is a placeholder)
    
    # For numeric responses, check if the new response is within a reasonable range
    if response_type == "number" and isinstance(response, (int, float)):
        numeric_values = [
            r.get("response_value") for r in metric_responses 
            if isinstance(r.get("response_value"), (int, float))
        ]
        
        if numeric_values:
            avg = sum(numeric_values) / len(numeric_values)
            # Check if value is within 30% of previous average
            if abs(response - avg) <= 0.3 * avg:
                return 0.9
            elif abs(response - avg) <= 0.5 * avg:
                return 0.7
            else:
                return 0.5
    
    # For categorical responses, check if the response matches previous answers
    if response_type in ["likert", "boolean", "multiple_choice"]:
        str_response = str(response)
        matching_responses = [
            r for r in metric_responses 
            if str(r.get("response_text")) == str_response
        ]
        
        consistency = 0.5 + (0.4 * (len(matching_responses) / max(1, len(metric_responses))))
        return consistency
    
    # Default value for text and other responses
    return 0.7

def calculate_completeness(response: Any, response_type: str) -> float:
    """Calculate response completeness.
    
    Args:
        response: The user's response value
        response_type: The type of response
        
    Returns:
        A completeness score between 0 and 1
    """
    if response is None:
        return 0.0
    
    if response_type == "text":
        # Text completeness based on length and content
        if not response:
            return 0.0
        
        text_length = len(str(response))
        
        # Short responses are less complete
        if text_length < 5:
            return 0.3
        elif text_length < 20:
            return 0.6
        elif text_length < 50:
            return 0.8
        else:
            return 0.95
    
    elif response_type in ["likert", "boolean", "multiple_choice", "number"]:
        # These response types are either complete or not
        if response == "":
            return 0.0
        return 0.95
    
    # Default value
    return 0.7

def update_metric_score(
    metric: str, 
    response: Any, 
    response_type: str,
    session_state: Dict[str, Any],
    previous_responses: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """Update a metric's score based on a new response.
    
    Args:
        metric: The metric to update
        response: The user's response
        response_type: The type of response
        session_state: The current session state
        previous_responses: Previous responses (if not available in session_state)
        
    Returns:
        Dictionary with updated score components
    """
    # Get current score and question count
    metrics = session_state.get("metrics", {})
    metric_data = metrics.get(metric, {
        "score": 0.5,  # Default starting score
        "confidence": 0.0,
        "questions_asked": 0
    })
    
    current_score = metric_data.get("score", 0.5)
    questions_asked = metric_data.get("questions_asked", 0)
    
    # Use provided previous responses or get from session state
    if previous_responses is None:
        previous_responses = session_state.get("question_history", [])
    
    # Calculate response quality components
    certainty = calculate_certainty(response, response_type)
    consistency = calculate_consistency(response, response_type, previous_responses, metric)
    completeness = calculate_completeness(response, response_type)
    
    # Combine factors with weights
    response_quality = (0.4 * certainty + 0.3 * consistency + 0.3 * completeness)
    
    # Apply Bayesian update with diminishing returns
    # More weight on initial questions, less on follow-ups
    learning_rate = max(0.2, 0.8 / (1 + 0.5 * questions_asked))
    
    # Update score with learning rate
    new_score = current_score + learning_rate * (response_quality - current_score)
    new_score = min(1.0, max(0.0, new_score))  # Clamp between 0 and 1
    
    # Update confidence based on questions asked and response quality
    base_confidence = min(0.95, (questions_asked + 1) / 5)  # Max confidence after ~5 questions
    quality_factor = (response_quality + 0.5) / 1.5  # Range from 0.33 to 1
    new_confidence = base_confidence * quality_factor
    
    # Return updated components
    return {
        "score": new_score,
        "confidence": new_confidence,
        "certainty": certainty,
        "consistency": consistency,
        "completeness": completeness,
        "questions_asked": questions_asked + 1
    }

def get_confidence_threshold(metric: str, session_state: Dict[str, Any]) -> Dict[str, float]:
    """Get dynamic confidence thresholds for a metric.
    
    Args:
        metric: The metric to get thresholds for
        session_state: The current session state
        
    Returns:
        Dictionary with high, medium, and low thresholds
    """
    # Base thresholds
    base_thresholds = {
        "high": 0.75,
        "medium": 0.4,
        "low": 0.0
    }
    
    # Get metric metadata
    metrics_metadata = session_state.get("metrics_metadata", {})
    metric_metadata = metrics_metadata.get(metric, {})
    
    # Adjust based on metric importance (configured in metadata)
    importance_factor = metric_metadata.get("importance", 1.0)
    
    # Adjust based on questions asked (higher standards as we progress)
    total_questions = session_state.get("total_questions", 0)
    progression_factor = min(1.3, 1 + (total_questions / 20))
    
    # Adjust based on user patience (detected from response patterns)
    patience_factor = get_user_patience_factor(session_state)
    
    # Calculate adjusted thresholds
    adjusted = {
        level: min(0.98, base * importance_factor * progression_factor / patience_factor)
        for level, base in base_thresholds.items()
    }
    
    return adjusted

def get_user_patience_factor(session_state: Dict[str, Any]) -> float:
    """Estimate user patience factor from response patterns.
    
    Args:
        session_state: The current session state
        
    Returns:
        A patience factor (lower means less patient, higher threshold adjustments)
    """
    # This is a placeholder for more sophisticated patience detection
    # In a real implementation, examine:
    # - Response times
    # - Response length
    # - Skipped questions
    # - Overall engagement
    
    # For MVP, use a simple heuristic based on number of short responses
    history = session_state.get("question_history", [])
    
    if not history:
        return 1.0  # Default patience factor
    
    # Count short text responses as a sign of impatience
    short_responses = sum(
        1 for item in history 
        if item.get("response_type") == "text" 
        and len(str(item.get("response_text", ""))) < 10
    )
    
    short_ratio = short_responses / max(1, len(history))
    
    # Calculate patience factor (1.0 is neutral, lower means less patient)
    patience_factor = max(0.7, 1.0 - (short_ratio * 0.3))
    
    return patience_factor 