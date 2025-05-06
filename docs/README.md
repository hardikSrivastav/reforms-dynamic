# 🌲 Enhanced Decision Tree & Scoring Architecture for Adaptive Question Engine

## 🎯 Purpose

To create an ultra-responsive question engine that minimizes latency and maximizes data quality while asking the fewest possible questions. The architecture uses a tiered approach with deterministic rules first, vector-based retrieval second, and LLM reasoning only when necessary.

---

## 🧱 Core Components

### 1. Advanced Metric Scoring System

Each metric has a dynamic confidence score between 0 and 1, calculated through a Bayesian updating mechanism that accounts for:

#### Score Components:
- **Response Certainty** (40%): How definitive the user's answer is
- **Response Consistency** (30%): Agreement with previous related answers
- **Response Completeness** (30%): Depth of information provided

#### Detailed Score Update Algorithm:
```python
def update_metric_score(metric, response, session_state):
    # Current score and question count
    current_score = session_state["metrics"][metric]["score"]
    questions_asked = session_state["metrics"][metric]["questions_asked"]
    
    # Calculate response certainty (varies by response type)
    certainty = calculate_certainty(response, response_type)
    
    # Calculate consistency with previous responses
    consistency = calculate_consistency(response, session_state["responses"], metric)
    
    # Calculate completeness
    completeness = calculate_completeness(response, response_type)
    
    # Combine factors with weights
    response_quality = (0.4 * certainty + 0.3 * consistency + 0.3 * completeness)
    
    # Apply Bayesian update with diminishing returns
    # More weight on initial questions, less on follow-ups
    learning_rate = max(0.2, 0.8 / (1 + 0.5 * questions_asked))
    
    # Update score with learning rate
    new_score = current_score + learning_rate * (response_quality - current_score)
    
    return min(1.0, max(0.0, new_score))  # Clamp between 0 and 1
```

#### Response Type Handling:
```python
# Response certainty by type
def calculate_certainty(response, response_type):
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
        return likert_certainty.get(response, 0.5)
    
    elif response_type == "number":
        # Higher certainty for specific numbers vs ranges or "I don't know"
        if isinstance(response, (int, float)):
            return 0.9
        elif "about" in str(response).lower() or "around" in str(response).lower():
            return 0.7
        else:
            return 0.4
            
    elif response_type == "boolean":
        # Yes/No are certain, "Maybe" or "Sometimes" less so
        boolean_certainty = {
            "Yes": 0.95,
            "No": 0.95, 
            "Maybe": 0.5,
            "Sometimes": 0.6,
            "It depends": 0.3
        }
        return boolean_certainty.get(response, 0.5)
        
    elif response_type == "text":
        # Text responses assessed by length, specificity, and sentiment analysis
        return assess_text_certainty(response)
    
    return 0.5  # Default certainty
```

---

### 2. Adaptive Confidence Thresholds

Thresholds dynamically adjust based on:
- Question progression (tightens as more questions are asked)
- Metric importance (critical metrics require higher confidence)
- User patience factor (estimated from response patterns)

#### Dynamic Threshold Calculation:
```python
def get_confidence_threshold(metric, session_state):
    base_thresholds = {
        "high": 0.75,
        "medium": 0.4,
        "low": 0.0
    }
    
    # Adjust based on metric importance (configured in metadata)
    importance_factor = METRIC_METADATA[metric].get("importance", 1.0)
    
    # Adjust based on questions asked (higher standards as we progress)
    progression_factor = min(1.3, 1 + (session_state["total_questions"] / 20))
    
    # Adjust based on user patience (detected from response patterns)
    patience_factor = get_user_patience_factor(session_state)
    
    # Calculate adjusted thresholds
    adjusted = {
        level: min(0.98, base * importance_factor * progression_factor / patience_factor)
        for level, base in base_thresholds.items()
    }
    
    return adjusted
```

---

## 🌲 Enhanced Decision Tree Architecture

Decision trees now incorporate:
- Conditional probability paths based on user cohort detection
- Information gain calculations to prioritize high-value questions
- Short-circuit paths for rapid classification of common patterns

### Advanced Tree Structure Example (Metric: `sleep_quality`):

```json
{
  "sleep_quality": {
    "metadata": {
      "importance": 1.2,
      "min_questions": 2,
      "max_questions": 5
    },
    "root": "q1",
    "nodes": {
      "q1": {
        "text": "How many hours do you typically sleep each night?",
        "type": "number",
        "information_gain": 0.85,
        "cohort_shifts": {
          "age>60": "q1_elderly",
          "has_children": "q1_parent"
        },
        "next": {
          "<5": {"node": "q2a", "score_update": {"sleep_quality": -0.3}},
          "5-6": {"node": "q2b", "score_update": {"sleep_quality": -0.15}},
          "7-8": {"node": "q2c", "score_update": {"sleep_quality": 0.2}},
          ">8": {"node": "q2d", "score_update": {"sleep_quality": 0.1}}
        },
        "shortcuts": {
          "<4_AND_feels_tired": {"confidence": 0.85, "value": "poor", "exit": true}
        }
      },
      "q1_elderly": {
        "text": "How many hours do you sleep each night, including any naps during the day?",
        "type": "number",
        "next": {/* similar structure */}
      },
      "q2a": {
        "text": "Do you have trouble falling asleep, staying asleep, or both?",
        "type": "multiple_choice",
        "options": ["Falling asleep", "Staying asleep", "Both", "Neither"],
        "next": {
          "Falling asleep": {"node": "q3a", "score_update": {"sleep_quality": -0.1}},
          "Staying asleep": {"node": "q3b", "score_update": {"sleep_quality": -0.15}},
          "Both": {"node": "q3c", "score_update": {"sleep_quality": -0.25}},
          "Neither": {"node": "q4", "score_update": {"sleep_quality": 0.05}}
        }
      },
      /* Additional nodes */
    }
  }
}
```

---

## ⚙️ Optimized Rule Evaluation Engine

### Enhanced Logic with Predictive Path Selection:
```python
def evaluate_rules(session_state):
    # Identify candidate metrics that need assessment
    candidates = prioritize_metrics(session_state)
    
    # Check for short-circuit opportunities
    shortcut = check_shortcuts(session_state)
    if shortcut:
        return shortcut
    
    # Find best next question from decision trees
    for metric in candidates:
        tree = get_decision_tree(metric)
        if tree:
            # Calculate information gain for possible next questions
            next_question = tree.get_optimal_question(
                session_state["question_history"],
                session_state["user_profile"]
            )
            if next_question:
                return next_question
    
    return None  # No rule-based decision possible
```

### Question Selection Strategy:
```python
def prioritize_metrics(session_state):
    """Order metrics by priority for questioning."""
    metrics = session_state["metrics_pending"]
    
    # Sort based on multiple factors
    return sorted(metrics, key=lambda m: (
        # 1. Information value (higher = better)
        -get_expected_information_gain(m, session_state),
        # 2. Questions already asked (fewer = better)
        session_state["metrics"][m]["questions_asked"],
        # 3. Current confidence (lower = better)
        session_state["metrics"][m]["score"],
        # 4. Dependency satisfaction (more deps met = better)
        -count_satisfied_dependencies(m, session_state)
    ))
```

---

## 🧩 Enhanced Integration with Hybrid Engine

The three-tier system uses a graceful fallback approach with clear handoffs, prioritizing speed and reliability:

### 1. Rule-Based Layer (1-2ms latency)
- Makes deterministic decisions based on the enhanced decision trees
- Applies scoring adjustments on every response
- Handles 70-80% of question selections in the MVP phase

```python
def rule_based_question_selection(session_state):
    """Select next question using decision trees only."""
    metric_priorities = prioritize_metrics_for_questioning(session_state)
    
    for metric_id in metric_priorities:
        # Skip metrics that have reached their confidence threshold
        if is_metric_complete(metric_id, session_state):
            continue
            
        # Get the decision tree for this metric
        tree = get_decision_tree(metric_id)
        if not tree:
            continue
            
        # Find current position in tree based on question history
        current_node = determine_current_node(tree, metric_id, session_state)
        
        # Get next question from tree
        if current_node and "next" in current_node:
            # Find appropriate branch based on prior responses
            next_node_id = select_branch(current_node, session_state)
            if next_node_id:
                return {
                    "question_id": next_node_id,
                    "question_text": tree["nodes"][next_node_id]["text"],
                    "metric_id": metric_id,
                    "source": "rule_engine"
                }
    
    return None  # No rule-based decision possible
```

### 2. Vector-Based Layer (10-30ms latency)
- Uses a straightforward vector database to find similar question-response patterns
- MVP implementation focuses on simplicity and effectiveness

```python
def vector_based_question_selection(session_state):
    """Select next question using vector similarity."""
    try:
        # Create a simple feature vector from the session (MVP approach)
        session_vector = create_session_vector(session_state)
        
        # Query vector database with reasonable timeout
        with timeout(25):  # ms
            results = vector_client.search(
                collection_name="question_paths",
                query_vector=session_vector,
                limit=2,
                score_threshold=0.9  # Conservative threshold for MVP
            )
        
        if not results:
            return None
            
        # Use the top match if it's very confident
        if results[0].score > 0.95:
            next_question = extract_question(results[0].payload)
            return {
                "question_id": next_question["id"],
                "question_text": next_question["text"],
                "metric_id": next_question["metric_id"],
                "source": "vector_db",
                "confidence": results[0].score
            }
            
        return None  # No high-confidence match
            
    except (TimeoutError, ConnectionError):
        # Graceful fallback if vector search fails
        log.warning("Vector search timed out or failed")
        return None
```

#### Vector Implementation Details for MVP:
- **Vector Dimensions**: 384-dimensional vectors (text-embedding-3-small)
- **Vector Contents**: Each vector represents:
  - Previous 2-3 question/answer pairs (70% of signal)
  - Current metric scores (20% of signal)
  - Basic user context if available (10% of signal)
- **Storage**: Simple collection in Qdrant with minimal metadata:
  ```json
  {
    "vector": [0.1, 0.2, ...],  // 384 dimensions
    "payload": {
      "question_id": "sleep_q2b",
      "question_text": "Do you have trouble falling asleep?",
      "metric_id": "sleep_quality",
      "usage_count": 37,
      "success_rate": 0.86
    }
  }
  ```
- **Indexing Strategy**: HNSW index with default parameters
- **Cold Start Handling**: Pre-populated with successful paths from decision trees

### 3. LLM Layer (50-500ms latency)
- Invoked only when rules and vector-based approaches fail
- Optimized for high quality while keeping costs manageable

#### LLM Prompt Template for MVP:
```
System: You are an expert survey designer helping create the next best question for a user.

Context:
- We are conducting a survey about: {survey_topic}
- We need to assess these metrics: {unresolved_metrics}
- Current focus metric: {current_metric.name} ({current_metric.description})
- The conversation so far:
{conversation_history}

Task:
Generate exactly ONE follow-up question that will best help us assess the {current_metric.name} metric.

Requirements:
1. The question must be concise (max 20 words)
2. The question must be conversational in tone
3. The question must relate directly to {current_metric.name}
4. The question should avoid repeating information we already know
5. The question type should be {preferred_question_type}

Output format:
{
  "question": "Your question text here?",
  "expected_information_gain": 0.X,  // 0.0-1.0 scale
  "rationale": "Brief explanation of why this question is valuable"
}
```

#### LLM Configuration for MVP:
- Model: Claude 3 Opus (for highest quality) or Claude 3.5 Sonnet (for speed)
- Temperature: 0.2 (prioritize consistency)
- Max tokens: 500 (keep responses focused)
- Timeout: 1000ms (with retry at 500ms if needed)

#### LLM Response Validation:
```python
def validate_llm_response(response, session_state):
    """Validate that LLM response meets quality standards."""
    try:
        # Parse the response as JSON
        parsed = json.loads(response)
        
        # Check required fields
        if "question" not in parsed:
            return False, "Missing question field"
            
        # Check question quality
        question = parsed["question"]
        if len(question.split()) > 25:  # Allow slight buffer over requirement
            return False, "Question too long"
            
        # Check for repetition
        if is_question_repetitive(question, session_state):
            return False, "Question repeats previous information"
            
        # All checks passed
        return True, parsed
        
    except json.JSONDecodeError:
        return False, "Invalid JSON format"
```

#### MVP Communication Protocol:
```python
def get_next_question(session_state):
    """Main entry point for question selection, using the 3-tier approach."""
    # Start timer for performance tracking
    start_time = time.time()
    
    try:
        # LAYER 1: Try deterministic rules first (fastest)
        rule_result = rule_based_question_selection(session_state)
        if rule_result:
            log_performance("rule_engine", time.time() - start_time)
            return rule_result
        
        # LAYER 2: Try vector similarity search if enabled
        if config.VECTOR_SEARCH_ENABLED:
            vector_result = vector_based_question_selection(session_state)
            if vector_result:
                log_performance("vector_engine", time.time() - start_time)
                return vector_result
        
        # LAYER 3: Fall back to LLM
        # For MVP, identify the highest priority metric with lowest confidence
        target_metric = select_target_metric_for_llm(session_state)
        
        # Prepare minimal context to reduce token usage
        llm_context = {
            "survey_topic": session_state["survey_topic"],
            "current_metric": get_metric_details(target_metric),
            "unresolved_metrics": get_unresolved_metrics_names(session_state),
            "conversation_history": format_recent_conversation(session_state, max_turns=3),
            "preferred_question_type": get_preferred_question_type(target_metric)
        }
        
        # Query LLM with timeout and retry logic
        llm_result = query_llm_with_retry(llm_context)
        
        # Validate and process the LLM response
        is_valid, processed_result = validate_llm_response(llm_result, session_state)
        
        if is_valid:
            final_result = {
                "question_text": processed_result["question"],
                "metric_id": target_metric,
                "source": "llm",
                "information_gain": processed_result.get("expected_information_gain", 0.5)
            }
            
            # Store successful LLM generations for future reference
            if config.STORE_LLM_RESULTS:
                store_successful_llm_result(session_state, final_result)
                
            log_performance("llm_engine", time.time() - start_time)
            return final_result
        else:
            # If LLM fails, use a fallback question from the backup library
            log.warning(f"LLM validation failed: {processed_result}")
            return get_fallback_question(target_metric)
            
    except Exception as e:
        # Final safety net - never fail to return a question
        log.error(f"Question engine error: {str(e)}")
        return {
            "question_text": "Can you tell me more about what you're looking for in this program?",
            "metric_id": get_least_assessed_metric(session_state),
            "source": "fallback"
        }
```

#### Failure Handling:
```python
def get_fallback_question(metric_id):
    """Return a pre-defined fallback question if all else fails."""
    # Simple library of generic fallback questions per metric
    fallbacks = {
        "interest_level": "How interested are you in this fellowship program?",
        "perceived_value": "What do you think would be most valuable about this program?",
        "relevance_to_goals": "How does this program align with your future goals?",
        "program_benefits": "Which aspects of the program sound most beneficial to you?",
        "likelihood_to_recommend": "Would you recommend a program like this to your friends?",
        "engagement_level": "How engaged would you want to be in program activities?",
        "awareness": "Had you heard about our program before this conversation?"
    }
    
    return {
        "question_text": fallbacks.get(metric_id, "What aspects of this program interest you most?"),
        "metric_id": metric_id,
        "source": "fallback_library"
    }
```

---

## 🔮 Advanced Integrated Question Engine

The new integrated question engine combines the best aspects of rule-based, vector-based, and LLM-based approaches into a unified system that dynamically selects the optimal strategy for each interaction point.

### Core Components of the Integrated Approach:

```python
async def advanced_question_engine(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Integrated question selection system that intelligently combines all approaches.
    
    Args:
        session_state: Current session state with history and metrics
        
    Returns:
        The next optimal question with metadata
    """
    # 1. First analyze the session context
    session_vector = await create_session_vector(session_state)
    
    # 2. Find similar past interaction patterns via vector search
    vector_matches = await search_vectors(
        vector=session_vector,
        limit=3,
        score_threshold=0.75
    )
    
    # 3. Check if we have a high-confidence match
    high_confidence_match = None
    if vector_matches and vector_matches[0].score > 0.92:
        high_confidence_match = vector_matches[0]
    
    # 4. Use LLM to strategically determine the best approach
    strategy = await llm_determine_strategy(
        session_state, 
        vector_examples=[m.payload for m in vector_matches]
    )
    
    # 5. Select the question based on confidence and strategy
    if high_confidence_match and strategy["recommended_approach"] == "reuse_past_pattern":
        # Use the high-confidence vector match
        question = format_question_from_vector(high_confidence_match.payload)
        question["selection_method"] = "vector_exact"
        return question
        
    elif strategy["recommended_approach"] == "category_search":
        # Use the LLM-suggested category to guide vector search
        category_matches = await search_vectors_by_category(
            vector=session_vector,
            category=strategy["category"],
            limit=1
        )
        
        if category_matches and category_matches[0].score > 0.85:
            question = format_question_from_vector(category_matches[0].payload)
            question["selection_method"] = "vector_category"
            return question
    
    # 6. If no good matches or LLM recommends custom question, use RAG approach
    # Provide vector examples to the LLM as context for better question generation
    examples = [m.payload for m in vector_matches]
    
    # Generate a tailored question using the LLM with vector examples as context
    llm_question = await llm_question_with_examples(
        session_state=session_state,
        metric_id=strategy["target_metric"],
        examples=examples,
        strategy=strategy
    )
    
    llm_question["selection_method"] = "llm_rag"
    
    # 7. Store this new question pattern for future reuse
    await store_new_question_pattern(
        session_vector=session_vector,
        question=llm_question,
        strategy=strategy
    )
    
    return llm_question
```

### Continuous Learning Feedback Loop:

```python
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
    # Get session state
    session_state = await get_session(session_id)
    if not session_state:
        return
    
    # Find the question in history
    question = None
    for item in session_state.get("question_history", []):
        if item.get("id") == question_id:
            question = item
            break
    
    if not question:
        return
    
    # Calculate effectiveness score based on metrics update
    effectiveness = calculate_question_effectiveness(
        question=question,
        response=response,
        metrics_update=metrics_update,
        user_satisfaction=user_satisfaction
    )
    
    # Update the question's success stats in the vector database
    if question.get("selection_method") in ("vector_exact", "vector_category"):
        # Update existing vector
        await update_vector_success_rate(
            question_id=question_id,
            success_score=effectiveness,
            response=response
        )
    
    elif question.get("selection_method") == "llm_rag":
        # For LLM-generated questions that were effective, add to vector DB
        if effectiveness > 0.7:  # Only store effective questions
            session_vector = await create_session_vector(session_state)
            await store_successful_question(
                session_vector=session_vector,
                question=question,
                response=response,
                effectiveness=effectiveness
            )
```

### Strategy Determination by LLM:

```python
async def llm_determine_strategy(
    session_state: Dict[str, Any],
    vector_examples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Use LLM to determine the best questioning strategy.
    
    Args:
        session_state: Current session state
        vector_examples: Similar past interactions from vector search
        
    Returns:
        Strategy recommendation with approach and target metric
    """
    # Format context for LLM
    prompt = f"""
    Analyze this conversation to determine the best questioning strategy:
    
    CONVERSATION HISTORY:
    {format_conversation_history(session_state)}
    
    CURRENT METRICS STATUS:
    {format_metrics_status(session_state)}
    
    SIMILAR PAST INTERACTIONS:
    {format_vector_examples(vector_examples)}
    
    Based on this information, what is the best strategy to ask the next question?
    Consider these approaches:
    1. reuse_past_pattern: Use a question pattern from similar past interactions
    2. category_search: Focus on a specific category of questions
    3. custom_generation: Generate a completely custom question
    
    Your response should include:
    - recommended_approach: One of the approaches above
    - target_metric: Which metric to focus on
    - category: If using category_search, which category to search in
    - rationale: Brief explanation of this choice
    """
    
    # Get response from LLM
    llm_response = await query_llm(prompt=prompt, response_format="json")
    
    # Parse and validate the response
    try:
        strategy = json.loads(llm_response)
        required_fields = ["recommended_approach", "target_metric"]
        
        if all(field in strategy for field in required_fields):
            return strategy
        else:
            # Default strategy if LLM response is invalid
            return {
                "recommended_approach": "reuse_past_pattern",
                "target_metric": get_highest_priority_metric(session_state),
                "rationale": "Default strategy due to invalid LLM response"
            }
            
    except (json.JSONDecodeError, KeyError):
        # Fallback to default strategy
        return {
            "recommended_approach": "reuse_past_pattern",
            "target_metric": get_highest_priority_metric(session_state),
            "rationale": "Default strategy due to LLM parsing error"
        }
```

### Vector-Based Categories for Advanced Retrieval:

```python
async def search_vectors_by_category(
    vector: List[float],
    category: str,
    limit: int = 1
) -> List[Any]:
    """Search vectors within a specific category.
    
    Args:
        vector: The session vector
        category: The category to search within
        limit: Maximum number of results
        
    Returns:
        List of matching vectors
    """
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
    
    # Search with category filter
    results = await vector_client.search(
        collection_name="question_patterns",
        query_vector=vector,
        query_filter=filter_condition,
        limit=limit,
        score_threshold=0.65  # Lower threshold when filtering by category
    )
    
    return results
```

### Enhanced Vector Storage Schema:

```json
{
  "vector": [0.1, 0.2, ...],  // 384 dimensions
  "payload": {
    "question_id": "sleep_q2b",
    "question_text": "Do you have trouble falling asleep?",
    "metric_id": "sleep_quality",
    "category": "sleep_patterns",
    "usage_count": 37,
    "success_rate": 0.86,
    "last_used": "2025-06-01T12:34:56Z",
    "generated_by": "rule_engine",
    "effectiveness_history": [0.82, 0.85, 0.91, 0.88]
  }
}
```

### RAG-Enhanced Question Generation:

```python
async def llm_question_with_examples(
    session_state: Dict[str, Any],
    metric_id: str,
    examples: List[Dict[str, Any]],
    strategy: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a question using LLM with vector examples as context.
    
    Args:
        session_state: Current session state
        metric_id: Target metric ID
        examples: Similar question examples from vector search
        strategy: Strategy determined by LLM
        
    Returns:
        Generated question
    """
    # Format examples for LLM context
    formatted_examples = "\n\n".join([
        f"EXAMPLE {i+1}:\n"
        f"Context: {e.get('context', 'No context')}\n"
        f"Question: {e.get('question_text', 'No question')}\n"
        f"Success Rate: {e.get('success_rate', 'Unknown')}"
        for i, e in enumerate(examples)
    ])
    
    # Create prompt with examples as context
    prompt = f"""
    METRIC TO ASSESS: {metric_id} ({get_metric_description(metric_id)})
    
    USER CONVERSATION HISTORY:
    {format_recent_conversation(session_state, max_turns=3)}
    
    CURRENT METRICS STATUS:
    {format_metrics_status(session_state)}
    
    STRATEGY:
    {strategy.get('rationale', 'Generate the best question to assess the target metric')}
    
    SIMILAR SUCCESSFUL QUESTIONS:
    {formatted_examples}
    
    Based on this context, generate ONE question that will effectively assess the {metric_id} metric.
    The question should:
    1. Be conversational and engaging
    2. Build upon what we already know
    3. Not repeat previous questions
    4. Be specific and focused on the target metric
    
    QUESTION:
    """
    
    # Get response from LLM
    question_text = await query_llm(prompt=prompt, max_tokens=100)
    
    # Format the response
    return {
        "question_text": question_text.strip(),
        "metric_id": metric_id,
        "source": "llm_with_examples",
        "strategy": strategy.get("recommended_approach")
    }
```

---

## 📊 Continuous Optimization System

### Feedback Loop Architecture:
```python
def process_session_feedback(session_id, feedback_data):
    """Process end-of-session feedback to improve the system."""
    session = load_session(session_id)
    
    # Update question effectiveness scores
    for q_id, rating in feedback_data["question_ratings"].items():
        update_question_effectiveness(q_id, rating)
    
    # Analyze path efficiency
    question_count = len(session["question_history"])
    data_quality = feedback_data["data_quality"]
    
    efficiency_score = calculate_efficiency(question_count, data_quality)
    
    # Update decision trees if needed
    if efficiency_score < THRESHOLD_EFFICIENCY:
        analyze_for_tree_improvements(session)
    
    # Update vector embeddings
    if feedback_data["path_helpfulness"] > 4:  # Out of 5
        # This was a good path - prioritize it in vector search
        boost_path_in_vector_db(session)
```

---

## 🔐 Enhanced Storage Format

### Decision Tree Storage (JSON with Performance Metadata):
```json
{
  "sleep_quality": {
    "metadata": {
      "version": "2.3",
      "last_updated": "2025-04-15",
      "performance_metrics": {
        "avg_questions_needed": 2.7,
        "accuracy": 0.91,
        "latency_ms": 3.2
      },
      "importance": 1.2,
      "dependencies": ["stress_level"]
    },
    "nodes": {
      "q1": {
        "text": "How many hours do you sleep?",
        "type": "number",
        "effectiveness_score": 0.92,
        "information_gain": 0.85,
        "next": {/* branching logic */}
      },
      /* Additional nodes */
    }
  }
}
```

### Session State Structure:
```json
{
  "session_id": "user_12345_session_789",
  "user_profile": {
    "age_group": "30-45",
    "has_children": true,
    "previous_sessions": 2
  },
  "metrics": {
    "sleep_quality": {
      "score": 0.65,
      "confidence": 0.8,
      "questions_asked": 2,
      "last_updated": "2025-05-05T15:30:22Z"
    },
    "stress_level": {
      "score": 0.4,
      "confidence": 0.5,
      "questions_asked": 1,
      "last_updated": "2025-05-05T15:29:50Z"
    }
  },
  "question_history": [
    {
      "id": "stress_q1",
      "text": "On a scale of 1-10, how would you rate your stress level?",
      "response": "7",
      "timestamp": "2025-05-05T15:29:50Z",
      "decision_path": "rule_engine"
    },
    {
      "id": "sleep_q1",
      "text": "How many hours do you sleep each night?",
      "response": "5-6",
      "timestamp": "2025-05-05T15:30:10Z",
      "decision_path": "rule_engine"
    },
    {
      "id": "sleep_q2b",
      "text": "Do you have trouble falling asleep?",
      "response": "Often",
      "timestamp": "2025-05-05T15:30:22Z",
      "decision_path": "rule_engine"
    }
  ],
  "metrics_pending": ["nutrition", "exercise_frequency"],
  "total_questions": 3,
  "start_time": "2025-05-05T15:29:30Z"
}
```

---

## ⚡ Performance Optimization Features

### 1. Pre-computation and Caching
- Decision tree paths are pre-computed and cached for common user profiles
- Vector embeddings are pre-generated for frequent session patterns
- Question text templates are cached to avoid regeneration

### 2. Parallel Processing
- Multiple metric evaluations run concurrently
- Vector search runs in parallel with rule evaluation when appropriate
- Background analysis for decision tree improvements during idle times

### 3. Progressive Loading
- Initial questions use only the rule-based layer for instant response
- Vector database loads asynchronously while first questions are answered
- LLM warms up in background for potential future use

### 4. Short-Circuit Evaluation
- Common patterns can exit questioning early with high confidence
- Cascading score updates can satisfy multiple metrics with fewer questions
- Cross-metric inferences reduce total questions needed

---

## 📈 Performance Metrics and SLAs

### Response Time Targets:
- Rule-based decisions: < 5ms
- Vector-based decisions: < 25ms
- LLM-based decisions: < 500ms
- Avg question selection time: < 50ms

### Data Quality Targets:
- Min confidence threshold before completion: 0.8
- Max questions per session: 15
- Avg questions per session: 8
- Data quality score (as rated by analysts): > 4.2/5

---

## ✅ Expected Benefits

- ⚡ **Ultra-fast responses** for 80% of question paths (< 10ms)
- 📉 **Reduced question count** by 40% compared to fixed questionnaires
- 🧠 **95% reduction in LLM token usage** by leveraging rules first
- 📊 **Higher data quality** through adaptive confidence thresholds
- 🔄 **Self-improving system** via continuous feedback loops

> *"Ask the right question at the right time, never ask what you already know."*
