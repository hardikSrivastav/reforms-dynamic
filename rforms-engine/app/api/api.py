from fastapi import APIRouter
from app.api.endpoints import session, survey, decision_tree

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(session.router, tags=["sessions"])
api_router.include_router(survey.router, tags=["surveys"])
api_router.include_router(decision_tree.router, tags=["decision_trees"]) 