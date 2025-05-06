from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Question type definitions
QuestionType = Literal[
    "text", "number", "boolean", "multiple_choice", 
    "likert", "range", "date", "time"
]

# Base model with common fields
class BaseModelWithTimestamps(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Question option for multiple choice questions
class QuestionOption(BaseModel):
    value: str
    label: str
    score_update: Optional[Dict[str, float]] = None

# Node in the decision tree
class DecisionTreeNode(BaseModel):
    id: str
    text: str
    type: QuestionType
    information_gain: float = 0.5
    effectiveness_score: float = 0.5
    options: Optional[List[QuestionOption]] = None
    next: Optional[Dict[str, Dict[str, Union[str, Dict[str, float]]]]] = None
    cohort_shifts: Optional[Dict[str, str]] = None
    shortcuts: Optional[Dict[str, Dict[str, Union[bool, float, str]]]] = None

# Complete decision tree for a metric
class DecisionTree(BaseModelWithTimestamps):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_id: str
    metadata: Dict[str, Any]
    root: str
    nodes: Dict[str, DecisionTreeNode]

# Metric to be measured
class Metric(BaseModelWithTimestamps):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    survey_id: str
    name: str
    description: Optional[str] = None
    importance: float = 1.0
    min_questions: int = 2
    max_questions: int = 5
    dependencies: List[str] = []

# Question model for API requests/responses
class Question(BaseModel):
    id: str
    text: str
    type: QuestionType
    metric_id: str
    options: Optional[List[QuestionOption]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    source: str = "rule_engine"  # rule_engine, vector_db, llm
    information_gain: float = 0.5

# Response from user
class Response(BaseModelWithTimestamps):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    question_id: str
    response_text: str
    response_value: Optional[Any] = None
    response_type: str
    metric_updates: Dict[str, float] = {}
    certainty: float = 0.5
    consistency: float = 0.5
    completeness: float = 0.5
    
# Session state
class SessionState(BaseModelWithTimestamps):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_profile: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    question_history: List[Dict[str, Any]] = []
    metrics_pending: List[str] = []
    total_questions: int = 0
    
# Request to create a new session
class CreateSessionRequest(BaseModel):
    survey_id: str
    user_id: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    
# Request to get the next question
class GetNextQuestionRequest(BaseModel):
    session_id: str
    
# Response to a question
class SubmitResponseRequest(BaseModel):
    session_id: Optional[str] = None  # Made optional since it's in the URL path
    question_id: str
    response_text: str
    response_value: Optional[Any] = None
    
# Vector path for similar question sequences
class VectorPath(BaseModelWithTimestamps):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_vector: List[float]
    question_id: str
    question_text: str
    metric_id: str
    usage_count: int = 0
    success_rate: float = 0.0 