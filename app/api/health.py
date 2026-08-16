from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models.database import get_db_session
from app.models.schemas import HealthStatus
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus)
async def health_check(db: Session = Depends(get_db_session)):
    """Check health of all services."""
    
    # Check database
    db_status = "healthy"
    try:
        db.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check vector database
    vector_db_status = "healthy"
    try:
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        vs.client.get_collections()
    except Exception as e:
        vector_db_status = f"unhealthy: {str(e)}"
    
    # Check LLM API
    llm_api_status = "healthy"
    try:
        from app.services.agent import AgentService
        agent = AgentService()
        # Just check initialization, don't make actual call
    except Exception as e:
        llm_api_status = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if all([
        db_status == "healthy",
        vector_db_status == "healthy",
        llm_api_status == "healthy"
    ]) else "degraded"
    
    return HealthStatus(
        status=overall_status,
        version=settings.APP_VERSION,
        database=db_status,
        vector_db=vector_db_status,
        llm_api=llm_api_status
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Enterprise Agentic AI Platform API",
        "docs": "/docs",
        "health": "/health"
    }
