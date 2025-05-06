from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
import json
from loguru import logger
import sys
import uuid

from app.core.config import settings
from app.api.api import api_router
from app.db import init_db, close_db

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level=logging.DEBUG if settings.API_DEBUG else logging.INFO,
    serialize=False
)

# Create FastAPI app
app = FastAPI(
    title="RForms Adaptive Question Engine",
    description="API for managing adaptive question flows with decision trees, vector search, and LLM fallback",
    version="0.1.0",
    docs_url="/docs" if settings.API_DEBUG else None,
    redoc_url="/redoc" if settings.API_DEBUG else None,
    openapi_url="/openapi.json" if settings.API_DEBUG else None,
)

# Add CORS middleware
origins = settings.CORS_ORIGINS
if settings.API_ENV == "development" and "*" not in origins:
    # Add wildcard origin in development for easier debugging
    origins = origins + ["*"]

logger.info(f"Setting up CORS with origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=["X-Request-ID", "X-Process-Time"],
    max_age=3600,
)

# Add request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request time if not docs request
    path = request.url.path
    if not path.startswith(("/docs", "/redoc", "/openapi.json")):
        logger.info(f"Request to {request.method} {path} took {process_time:.4f}s")
    
    return response

# Add API router
app.include_router(api_router, prefix="/api")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    logger.info("Starting up database connections")
    try:
        await init_db()
        logger.info("Database connections initialized successfully")
    except Exception as e:
        # Log the error but allow the application to start
        logger.error(f"Failed to initialize database connections: {e}")
        logger.warning("Application starting with limited functionality due to database initialization errors")

@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down database connections")
    try:
        await close_db()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "app": "RForms Adaptive Question Engine",
        "status": "running",
        "version": "0.1.0",
        "docs_url": "/docs" if settings.API_DEBUG else None
    }
