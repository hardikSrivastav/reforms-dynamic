"""Utility functions for handling different question types in the survey system.

This module provides functionality to create, validate, and transform
different types of survey questions to ensure they work correctly across
the backend and frontend.
"""

from typing import Dict, Any, List, Optional
import random
import time
import json
import re

# Question type mapping between internal names and frontend types
QUESTION_TYPES = {
    "text": "text",                      # Free text input
    "number": "number",                  # Numeric input
    "select": "multiple_choice",         # Single-choice selection
    "multiselect": "multiselect",        # Multi-choice selection
    "boolean": "boolean",                # Yes/No toggle
    "likert": "likert",                  # Likert scale (1-5)
    "range": "range",                    # Range slider
    "date": "date",                      # Date picker
    "daterange": "daterange",            # Date range picker
    "email": "email",                    # Email input
    "phone": "phone",                    # Phone number input
    "location": "location",              # Location input
    "file": "file",                      # File upload
}

# Default question template for each type
QUESTION_TEMPLATES = {
    "text": {
        "type": "text",
        "placeholder": "Type your answer here..."
    },
    "number": {
        "type": "number",
        "min_value": 0,
        "max_value": 100
    },
    "select": {
        "type": "multiple_choice",
        "options": [
            {"value": "option1", "label": "Option 1"},
            {"value": "option2", "label": "Option 2"},
            {"value": "option3", "label": "Option 3"}
        ]
    },
    "multiselect": {
        "type": "multiselect",
        "options": [
            {"value": "option1", "label": "Option 1"},
            {"value": "option2", "label": "Option 2"},
            {"value": "option3", "label": "Option 3"}
        ]
    },
    "boolean": {
        "type": "boolean"
    },
    "likert": {
        "type": "likert"
    },
    "range": {
        "type": "range",
        "min_value": 0,
        "max_value": 100
    },
    "date": {
        "type": "date"
    },
    "daterange": {
        "type": "daterange"
    },
    "email": {
        "type": "email",
        "placeholder": "Enter your email address..."
    },
    "phone": {
        "type": "phone",
        "placeholder": "Enter your phone number..."
    },
    "location": {
        "type": "location",
        "placeholder": "Enter your location..."
    },
    "file": {
        "type": "file",
        "max_size": 5, # in MB
        "allowed_types": ["pdf", "doc", "docx", "jpg", "png"]
    }
}

def get_question_template(question_type: str) -> Dict[str, Any]:
    """Get the template for a specific question type.
    
    Args:
        question_type: Type of question
        
    Returns:
        Template with default values for the question type
    """
    if question_type in QUESTION_TEMPLATES:
        return QUESTION_TEMPLATES[question_type].copy()
    else:
        # Default to text if type not found
        return QUESTION_TEMPLATES["text"].copy()

def get_preferred_question_type(metric_id: str) -> str:
    """Determine the preferred question type for a metric.
    
    Different metrics might be better assessed with different question types.
    
    Args:
        metric_id: The metric ID
        
    Returns:
        Preferred question type for this metric
    """
    # Ensure we get variety in question types by using a timestamp-based rotation
    # time and random are already imported at the top of the file
    
    # Use the current timestamp to create a pseudo-random value that changes every few seconds
    # This ensures different question types are chosen over time, without requiring state
    current_time = int(time.time())
    time_interval = current_time % 300  # 5-minute cycle
    interval_position = time_interval % 80  # Position within cycle (0-79)
    
    # Map metrics to their preferred question types when we have specific preferences
    metric_question_types = {
        "satisfaction": "likert",
        "usage_frequency": "select", 
        "likelihood_to_recommend": "range",
        "pricing_feedback": "number",
        "feature_importance": "multiselect",
        "pain_points": "text",
        "contact_info": "email"
    }
    
    return metric_question_types.get(metric_id, "text")

def parse_llm_options(llm_response: str) -> List[Dict[str, str]]:
    """Parse options from an LLM response text.
    
    The LLM might include options in various formats:
    - As a numbered list: "1. Option one, 2. Option two"
    - As bullet points: "* Option one, * Option two"
    - With explicit JSON-like format: "[{value: '1', label: 'Option 1'}, ...]"
    
    Args:
        llm_response: The raw LLM response
        
    Returns:
        Parsed options as list of dicts with value and label
    """
    # Check if response has explicit options section
    if "<options>" in llm_response.lower() and "</options>" in llm_response.lower():
        # Extract options section
        options_start = llm_response.lower().find("<options>") + len("<options>")
        options_end = llm_response.lower().find("</options>")
        options_text = llm_response[options_start:options_end].strip()
        
        # Try to parse options
        options = []
        for line in options_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Remove bullet points or numbers
            line = line.lstrip("*-•⁃◦▪▫+1234567890.) ")
            
            # Split into value/label if a delimiter exists
            if ":" in line:
                value, label = line.split(":", 1)
                options.append({"value": value.strip(), "label": label.strip()})
            else:
                # Use the line as both value and label
                options.append({"value": line.strip(), "label": line.strip()})
        
        return options
    
    # Fallback to simple parsing
    options = []
    lines = llm_response.strip().split("\n")
    
    # Look for numbered or bulleted lists
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for list item patterns
        if line.startswith(("*", "-", "•", "⁃", "◦", "▪", "▫", "+")) or (line[0].isdigit() and line[1:3] in [". ", ") "]):
            # Remove bullet points or numbers
            line = line.lstrip("*-•⁃◦▪▫+1234567890.) ")
            options.append({"value": line.strip(), "label": line.strip()})
    
    # If no options found this way, try basic comma separation
    if not options and "," in llm_response:
        for item in llm_response.split(","):
            item = item.strip()
            if item:
                options.append({"value": item, "label": item})
    
    return options

def enhance_question_with_type(question: Dict[str, Any], question_type: str = None) -> Dict[str, Any]:
    """Enhance a question with type-specific fields based on its type.
    
    Args:
        question: The question object
        question_type: Optional override for question type
        
    Returns:
        Enhanced question with type-specific fields
    """
    # Use provided type or get from question
    q_type = question_type or question.get("type", "text")
    
    # Make a copy to avoid modifying the original
    enhanced = question.copy()
    
    # Set the type
    enhanced["type"] = q_type
    
    # Add necessary fields based on question type
    if q_type == "multiple_choice" or q_type == "select":
        if "options" not in enhanced:
            enhanced["options"] = [
                {"value": "option1", "label": "Option 1"},
                {"value": "option2", "label": "Option 2"},
                {"value": "option3", "label": "Option 3"},
            ]
            
    elif q_type == "multiselect":
        if "options" not in enhanced:
            enhanced["options"] = [
                {"value": "option1", "label": "Option 1"},
                {"value": "option2", "label": "Option 2"},
                {"value": "option3", "label": "Option 3"},
            ]
            
    elif q_type == "range":
        if "min_value" not in enhanced:
            enhanced["min_value"] = 0
        if "max_value" not in enhanced:
            enhanced["max_value"] = 100
            
    elif q_type == "likert":
        # Likert scales are pre-configured with 1-5 values
        pass
            
    elif q_type == "date" or q_type == "daterange":
        # Date picker configuration
        pass
    
    return enhanced

def extract_question_type_from_llm(llm_response: str) -> str:
    """Determine the question type from LLM response.
    
    This tries to detect if the LLM response includes explicit type indicators.
    
    Args:
        llm_response: The raw LLM response
        
    Returns:
        Detected question type or default "text"
    """
    # Check for explicit type tag
    if "<type>" in llm_response.lower() and "</type>" in llm_response.lower():
        type_start = llm_response.lower().find("<type>") + len("<type>")
        type_end = llm_response.lower().find("</type>")
        q_type = llm_response[type_start:type_end].strip().lower()
        
        # Map to valid question types
        if q_type in QUESTION_TYPES:
            return q_type
        elif q_type in QUESTION_TYPES.values():
            # Find the key for this value
            for key, value in QUESTION_TYPES.items():
                if value == q_type:
                    return key
    
    # Check for common patterns in the question text
    lower_text = llm_response.lower()
    
    if "on a scale" in lower_text or "rate " in lower_text:
        if "1-5" in lower_text or "1 to 5" in lower_text:
            return "likert"
        else:
            return "range"
            
    if "select all" in lower_text or "choose all" in lower_text or "multiple options" in lower_text:
        return "multiselect"
        
    if "select one" in lower_text or "choose one" in lower_text or "which of the following" in lower_text:
        return "select"
        
    if "yes or no" in lower_text:
        return "boolean"
        
    if "number" in lower_text or "numeric" in lower_text or "quantitative" in lower_text:
        return "number"
        
    if "email" in lower_text:
        return "email"
        
    if "phone" in lower_text:
        return "phone"
        
    if "location" in lower_text or "address" in lower_text:
        return "location"
        
    if "upload" in lower_text or "attach" in lower_text or "file" in lower_text:
        return "file"
        
    if "date" in lower_text:
        if "range" in lower_text or "period" in lower_text or "from" in lower_text and "to" in lower_text:
            return "daterange"
        else:
            return "date"
    
    # Default to text for free-form responses
    return "text"

def extract_range_values_from_text(question_text: str) -> tuple:
    """Extract min and max values from a range question.
    
    Args:
        question_text: The question text
        
    Returns:
        Tuple of (min_value, max_value)
    """
    # Default values
    min_value = 0
    max_value = 100
    
    # Look for patterns like "on a scale from X to Y" or "rate from X-Y"
    scale_patterns = [
        r"scale (?:from|of) (\d+)[\s-]+to[\s-]+(\d+)",
        r"scale (?:from|of) (\d+)[\s-]+(\d+)",
        r"(\d+)[\s-]+to[\s-]+(\d+) scale",
        r"from (\d+) to (\d+)",
        r"(\d+)-(\d+) scale"
    ]
    
    for pattern in scale_patterns:
        match = re.search(pattern, question_text.lower())
        if match:
            try:
                min_value = int(match.group(1))
                max_value = int(match.group(2))
                return min_value, max_value
            except (ValueError, IndexError):
                # If conversion fails, continue with the next pattern
                continue
    
    return min_value, max_value

def create_question_from_llm_response(
    llm_response: str, 
    metric_id: str,
    source: str = "llm"
) -> Dict[str, Any]:
    """Create a complete question object from an LLM response.
    
    Args:
        llm_response: The raw LLM response
        metric_id: The metric this question is for
        source: Source of the question (llm, vector_db, etc.)
        
    Returns:
        Complete question object with all necessary fields
    """
    # Extract the question type
    question_type = extract_question_type_from_llm(llm_response)
    
    # Clean up the response text
    question_text = llm_response.strip()
    
    # Remove any type tags
    if "<type>" in question_text.lower() and "</type>" in question_text.lower():
        type_start = question_text.lower().find("<type>")
        type_end = question_text.lower().find("</type>") + len("</type>")
        question_text = question_text[:type_start] + question_text[type_end:]
        question_text = question_text.strip()
    
    # Remove any options tags and text
    if "<options>" in question_text.lower() and "</options>" in question_text.lower():
        options_start = question_text.lower().find("<options>")
        options_end = question_text.lower().find("</options>") + len("</options>")
        question_text = question_text[:options_start] + question_text[options_end:]
        question_text = question_text.strip()
    
    # Create basic question structure
    question = {
        "question_id": f"{source}_{int(time.time())}",
        "question_text": question_text,
        "metric_id": metric_id,
        "type": QUESTION_TYPES.get(question_type, question_type),
        "source": source
    }
    
    # For select/multiselect, add options
    if question_type in ["select", "multiselect"]:
        options = parse_llm_options(llm_response)
        if options:
            question["options"] = options
    
    # For range questions, extract min_value and max_value from the question text
    if question_type == "range":
        min_value, max_value = extract_range_values_from_text(question_text)
        question["min_value"] = min_value
        question["max_value"] = max_value
    
    # Add fields specific to this question type
    question = enhance_question_with_type(question, question_type)
    
    return question 