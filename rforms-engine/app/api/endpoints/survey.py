from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List, Optional
from app.db.postgres import get_db_conn
from app.db.redis import get_cache, set_cache
from loguru import logger
import uuid
from datetime import datetime

router = APIRouter()

@router.get("/surveys", response_model=List[Dict[str, Any]])
async def list_surveys():
    """List all active surveys."""
    # Try to get from cache first
    cache_key = "surveys:list"
    cached_surveys = await get_cache(cache_key)
    if cached_surveys:
        return cached_surveys
    
    surveys = []
    conn = await get_db_conn()
    try:
        query = """
        SELECT id, title, description, created_at, updated_at
        FROM surveys
        WHERE is_active = TRUE
        ORDER BY created_at DESC
        """
        rows = await conn.fetch(query)
        
        for row in rows:
            survey = {
                "id": row['id'],
                "title": row['title'],
                "description": row['description'],
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
            }
            surveys.append(survey)
        
        # Cache for future use (10 minutes TTL)
        await set_cache(cache_key, surveys, 600)
        
        return surveys
    except Exception as e:
        logger.error(f"Error listing surveys: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()
    
    return surveys

@router.get("/surveys/{survey_id}", response_model=Dict[str, Any])
async def get_survey(survey_id: str):
    """Get survey details."""
    # Try to get from cache first
    cache_key = f"surveys:{survey_id}"
    cached_survey = await get_cache(cache_key)
    if cached_survey:
        return cached_survey
    
    conn = await get_db_conn()
    try:
        query = """
        SELECT id, title, description, created_at, updated_at
        FROM surveys
        WHERE id = $1 AND is_active = TRUE
        """
        row = await conn.fetchrow(query, survey_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Survey not found")
        
        survey = {
            "id": row['id'],
            "title": row['title'],
            "description": row['description'],
            "created_at": row['created_at'].isoformat() if row['created_at'] else None,
            "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None
        }
        
        # Get metrics for the survey
        metrics_query = """
        SELECT id, name, description, importance, min_questions, max_questions
        FROM metrics
        WHERE survey_id = $1
        ORDER BY importance DESC
        """
        metrics_rows = await conn.fetch(metrics_query, survey_id)
        
        metrics = []
        for m_row in metrics_rows:
            metric = {
                "id": m_row['id'],
                "name": m_row['name'],
                "description": m_row['description'],
                "importance": m_row['importance'],
                "min_questions": m_row['min_questions'],
                "max_questions": m_row['max_questions']
            }
            metrics.append(metric)
        
        survey["metrics"] = metrics
        
        # Cache for future use (10 minutes TTL)
        await set_cache(cache_key, survey, 600)
        
        return survey
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting survey: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()
    
    raise HTTPException(status_code=404, detail="Survey not found")

@router.post("/surveys", response_model=Dict[str, Any])
async def create_survey(
    title: str = Body(...),
    description: str = Body(...),
    metrics: List[Dict[str, Any]] = Body(...)
):
    """Create a new survey with metrics."""
    survey_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    conn = await get_db_conn()
    try:
        # Start a transaction
        async with conn.transaction():
            # Insert survey
            survey_query = """
            INSERT INTO surveys (id, title, description, is_active, created_at, updated_at)
            VALUES ($1, $2, $3, TRUE, $4, $5)
            RETURNING id
            """
            inserted_survey_id = await conn.fetchval(
                survey_query, 
                survey_id, title, description, now, now
            )
            
            # Insert metrics
            metric_ids = []
            for metric in metrics:
                metric_id = str(uuid.uuid4())
                metric_ids.append(metric_id)
                
                metric_query = """
                INSERT INTO metrics (
                    id, survey_id, name, description, importance, 
                    min_questions, max_questions, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                await conn.execute(
                    metric_query,
                    metric_id,
                    survey_id,
                    metric.get("name"),
                    metric.get("description"),
                    metric.get("importance", 1.0),
                    metric.get("min_questions", 2),
                    metric.get("max_questions", 5),
                    now,
                    now
                )
                
                # Insert metric dependencies if any
                dependencies = metric.get("dependencies", [])
                for dep in dependencies:
                    if dep.get("metric_id") in metric_ids:
                        dep_query = """
                        INSERT INTO metric_dependencies (
                            metric_id, depends_on_metric_id, weight
                        )
                        VALUES ($1, $2, $3)
                        """
                        await conn.execute(
                            dep_query,
                            metric_id,
                            dep.get("metric_id"),
                            dep.get("weight", 1.0)
                        )
        
        # Invalidate cache
        await set_cache("surveys:list", None, 0)
        
        return {
            "id": survey_id,
            "title": title,
            "metrics": [{"id": m_id} for m_id in metric_ids]
        }
    except Exception as e:
        logger.error(f"Error creating survey: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()
    
    raise HTTPException(status_code=500, detail="Database connection failed")

@router.get("/metrics/{metric_id}", response_model=Dict[str, Any])
async def get_metric(metric_id: str):
    """Get metric details."""
    # Import here to avoid circular import
    from app.core.metrics import get_metric_details
    
    metric = await get_metric_details(metric_id)
    
    if not metric:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    return metric

@router.post("/metrics/{metric_id}/dependencies", response_model=Dict[str, Any])
async def add_metric_dependency(
    metric_id: str,
    depends_on_metric_id: str = Body(...),
    weight: float = Body(1.0)
):
    """Add a dependency between metrics."""
    conn = await get_db_conn()
    try:
        # Check if both metrics exist
        check_query = """
        SELECT COUNT(*) FROM metrics WHERE id IN ($1, $2)
        """
        count = await conn.fetchval(check_query, metric_id, depends_on_metric_id)
        
        if count != 2:
            raise HTTPException(status_code=404, detail="One or both metrics not found")
        
        # Check if dependency already exists
        exists_query = """
        SELECT COUNT(*) FROM metric_dependencies
        WHERE metric_id = $1 AND depends_on_metric_id = $2
        """
        exists = await conn.fetchval(exists_query, metric_id, depends_on_metric_id)
        
        if exists > 0:
            # Update existing dependency
            update_query = """
            UPDATE metric_dependencies
            SET weight = $1
            WHERE metric_id = $2 AND depends_on_metric_id = $3
            """
            await conn.execute(update_query, weight, metric_id, depends_on_metric_id)
        else:
            # Create new dependency
            insert_query = """
            INSERT INTO metric_dependencies (metric_id, depends_on_metric_id, weight)
            VALUES ($1, $2, $3)
            """
            await conn.execute(insert_query, metric_id, depends_on_metric_id, weight)
        
        # Invalidate cache
        await set_cache(f"metric:{metric_id}", None, 0)
        
        return {
            "metric_id": metric_id,
            "depends_on_metric_id": depends_on_metric_id,
            "weight": weight
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding metric dependency: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()
    
    raise HTTPException(status_code=500, detail="Database connection failed") 