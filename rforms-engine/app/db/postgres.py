import asyncpg
from typing import AsyncGenerator
from app.core.config import settings
from loguru import logger
import asyncio
import uuid
from datetime import datetime

# Create a connection pool
pg_pool = None

async def init_postgres():
    """Initialize the PostgreSQL connection pool."""
    global pg_pool
    
    try:
        # Create a connection pool with asyncpg
        pg_pool = await asyncpg.create_pool(
            dsn=settings.POSTGRES_URI,
            min_size=5,
            max_size=20
        )
        
        # Test connection
        async with pg_pool.acquire() as conn:
            await conn.execute('SELECT 1')
        
        logger.info("PostgreSQL connection initialized successfully")
        
        # Create tables if they don't exist
        await create_tables()
        
        # Add sample data
        await add_sample_data()
        
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL connection: {e}")
        raise

async def close_postgres():
    """Close the PostgreSQL connection pool."""
    global pg_pool
    
    if pg_pool:
        await pg_pool.close()
        logger.info("PostgreSQL connection closed")

async def get_db_conn():
    """Get a database connection from the pool."""
    global pg_pool
    
    if pg_pool is None:
        # Initialize pool if not already done
        pg_pool = await asyncpg.create_pool(
            dsn=settings.POSTGRES_URI,
            min_size=5,
            max_size=20
        )
    
    return await pg_pool.acquire()

async def create_tables():
    """Create necessary tables if they don't exist."""
    tables = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(255) PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS surveys (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_by VARCHAR(255) REFERENCES users(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id VARCHAR(255) PRIMARY KEY,
            survey_id VARCHAR(255) REFERENCES surveys(id),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            importance FLOAT DEFAULT 1.0,
            min_questions INTEGER DEFAULT 2,
            max_questions INTEGER DEFAULT 5,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (survey_id, name)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS metric_dependencies (
            id SERIAL PRIMARY KEY,
            metric_id VARCHAR(255) REFERENCES metrics(id),
            depends_on_metric_id VARCHAR(255) REFERENCES metrics(id),
            weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (metric_id, depends_on_metric_id)
        )
        """
    ]
    
    try:
        async with pg_pool.acquire() as conn:
            for table_sql in tables:
                await conn.execute(table_sql)
            logger.info("PostgreSQL tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL tables: {e}")
        raise

async def add_sample_data():
    """Add sample data to the database for development."""
    try:
        async with pg_pool.acquire() as conn:
            # Check if we already have sample data
            survey_count = await conn.fetchval("SELECT COUNT(*) FROM surveys")
            if survey_count > 0:
                logger.info("Sample data already exists, skipping insertion")
                return
                
            # Create a sample admin user
            admin_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO users (id, email, hashed_password, is_active)
                VALUES ($1, $2, $3, true)
                """,
                admin_id, 
                "admin@example.com", 
                "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # "password"
            )
            
            # Create sample surveys
            now = datetime.utcnow()
            
            # Fellowship Survey
            fellowship_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO surveys (id, title, description, is_active, created_by, created_at, updated_at)
                VALUES ($1, $2, $3, true, $4, $5, $5)
                """,
                fellowship_id,
                "Fellowship Program Interest Survey",
                "A survey to gauge interest and fit for our fellowship program.",
                admin_id,
                now
            )
            
            # Product Feedback Survey
            product_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO surveys (id, title, description, is_active, created_by, created_at, updated_at)
                VALUES ($1, $2, $3, true, $4, $5, $5)
                """,
                product_id,
                "Product Feedback Survey",
                "Help us improve our product by sharing your thoughts and experiences.",
                admin_id,
                now
            )
            
            # Health Assessment
            health_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO surveys (id, title, description, is_active, created_by, created_at, updated_at)
                VALUES ($1, $2, $3, true, $4, $5, $5)
                """,
                health_id,
                "Health and Wellness Assessment",
                "Evaluate your current health status and receive personalized recommendations.",
                admin_id,
                now
            )
            
            # Add metrics for the Fellowship survey
            fellowship_metrics = [
                (str(uuid.uuid4()), fellowship_id, "interest_level", "Level of interest in the fellowship program", 1.5, 2, 5),
                (str(uuid.uuid4()), fellowship_id, "background_fit", "How well the candidate's background fits the program", 1.2, 2, 4),
                (str(uuid.uuid4()), fellowship_id, "goals_alignment", "Alignment between candidate goals and program outcomes", 1.3, 2, 4),
                (str(uuid.uuid4()), fellowship_id, "time_commitment", "Ability to commit time to the program", 1.0, 1, 3),
                (str(uuid.uuid4()), fellowship_id, "technical_skills", "Technical skill level relevant to the program", 1.1, 2, 4)
            ]
            
            for metric in fellowship_metrics:
                await conn.execute(
                    """
                    INSERT INTO metrics (id, survey_id, name, description, importance, min_questions, max_questions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    *metric
                )
            
            # Add metrics for the Product survey
            product_metrics = [
                (str(uuid.uuid4()), product_id, "ease_of_use", "How easy the product is to use", 1.4, 2, 5),
                (str(uuid.uuid4()), product_id, "feature_satisfaction", "Satisfaction with product features", 1.3, 2, 5),
                (str(uuid.uuid4()), product_id, "performance", "Performance and reliability of the product", 1.2, 2, 4),
                (str(uuid.uuid4()), product_id, "value_perception", "Perceived value for the price", 1.1, 1, 3),
                (str(uuid.uuid4()), product_id, "recommendation_likelihood", "Likelihood to recommend to others", 1.5, 1, 3)
            ]
            
            for metric in product_metrics:
                await conn.execute(
                    """
                    INSERT INTO metrics (id, survey_id, name, description, importance, min_questions, max_questions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    *metric
                )
            
            # Add metrics for the Health assessment
            health_metrics = [
                (str(uuid.uuid4()), health_id, "sleep_quality", "Quality of sleep and rest", 1.4, 2, 5),
                (str(uuid.uuid4()), health_id, "stress_level", "Current level of stress", 1.3, 2, 4),
                (str(uuid.uuid4()), health_id, "physical_activity", "Amount of regular physical activity", 1.2, 2, 4),
                (str(uuid.uuid4()), health_id, "nutrition", "Quality of dietary habits", 1.3, 2, 5),
                (str(uuid.uuid4()), health_id, "mental_wellbeing", "Overall mental and emotional wellbeing", 1.5, 2, 5)
            ]
            
            for metric in health_metrics:
                await conn.execute(
                    """
                    INSERT INTO metrics (id, survey_id, name, description, importance, min_questions, max_questions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    *metric
                )
            
            logger.info("Sample data inserted successfully")
    except Exception as e:
        logger.error(f"Failed to add sample data: {e}")
        # Don't raise here, just log the error so the application can still start 