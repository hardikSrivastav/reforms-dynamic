from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, Optional
from app.models.question import CreateSessionRequest, GetNextQuestionRequest, SubmitResponseRequest
from app.services.session import (
    create_session, 
    get_session, 
    get_next_question, 
    submit_response, 
    end_session,
    process_session_feedback
)
from loguru import logger
import json

router = APIRouter()

@router.post("/sessions", response_model=Dict[str, Any])
async def api_create_session(request: CreateSessionRequest):
    """Create a new session."""
    try:
        session_data = await create_session(
            survey_id=request.survey_id,
            user_id=request.user_id,
            user_profile=request.user_profile
        )
        
        # Extract session_id from the session data and return just that
        # This maintains backward compatibility with clients expecting just the session_id
        session_id = session_data["session_id"]
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def api_get_session(session_id: str):
    """Get session state."""
    session = await get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@router.post("/sessions/{session_id}/questions/next", response_model=Dict[str, Any])
async def api_get_next_question(session_id: str):
    """Get the next question for a session."""
    question = await get_next_question(session_id)
    
    if not question:
        raise HTTPException(status_code=404, detail="Session not found or completed")
    
    return question

@router.post("/sessions/{session_id}/responses", response_model=Dict[str, Any])
async def api_submit_response(session_id: str, raw_request: Request):
    """Submit a response to a question."""
    try:
        # Get raw request body for debugging
        request_body = await raw_request.json()
        logger.info(f"Raw request body: {request_body}")
        
        # Manual validation to see exactly what's failing
        if "question_id" not in request_body:
            logger.error("Missing required field 'question_id'")
            raise HTTPException(status_code=422, detail="Missing required field 'question_id'")
            
        if "response_text" not in request_body:
            logger.error("Missing required field 'response_text'")
            raise HTTPException(status_code=422, detail="Missing required field 'response_text'")
            
        # All validation passed, manually construct the request
        question_id = request_body.get("question_id")
        response_text = request_body.get("response_text", "")
        response_value = request_body.get("response_value")
        
        logger.info(f"Processing response - Session: {session_id}, Question: {question_id}")
        logger.info(f"Response text: '{response_text}', Response value: {response_value}")
        
        # Call service layer
        result = await submit_response(
            session_id=session_id,
            question_id=question_id,
            response_text=response_text,
            response_value=response_value
        )
        
        if "error" in result:
            logger.error(f"Error in submit_response: {result['error']}")
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Log the full exception for debugging
        import traceback
        logger.error(f"Error submitting response: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/end", response_model=Dict[str, Any])
async def api_end_session(session_id: str):
    """End a session and return final metrics."""
    result = await end_session(session_id, "completed")
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@router.post("/sessions/{session_id}/feedback", response_model=Dict[str, Any])
async def api_submit_feedback(session_id: str, feedback_data: Dict[str, Any]):
    """Submit feedback for a session."""
    success = await process_session_feedback(session_id, feedback_data)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "Feedback received"} 