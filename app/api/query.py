from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.models.database import get_db_session, User
from app.models.schemas import (
    QueryRequest, QueryResponse, SourceChunk,
    EvaluationRequest, EvaluationResponse, MetricResult, MetricsSummary
)
from app.utils.deps import get_current_user
from app.services.agent import AgentService
from app.services.document_service import QueryHistoryService, EvaluationService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Submit a query to the AI agent with RAG."""
    
    agent_service = AgentService()
    
    # Process query
    result = agent_service.query_with_rag(
        query=request.query,
        top_k=request.top_k,
        include_sources=request.include_sources,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # Log query to history
    query_history_service = QueryHistoryService(db)
    query_record = query_history_service.log_query(
        user_id=current_user.id,
        query_text=request.query,
        response_text=result["response"],
        retrieved_chunks=result.get("sources", []),
        tokens_used=result.get("tokens_used", 0),
        latency_ms=result.get("latency_ms", 0),
        cost_usd=result.get("cost_usd", 0),
        success=result.get("success", False),
        error_message=result.get("error")
    )
    
    # Format sources
    sources = None
    if request.include_sources and result.get("sources"):
        sources = [
            SourceChunk(
                id=chunk.get("id", ""),
                content=chunk.get("content", ""),
                metadata=chunk.get("metadata", {}),
                score=chunk.get("score", 0.0)
            )
            for chunk in result["sources"]
        ]
    
    return QueryResponse(
        id=query_record.id,
        query=request.query,
        response=result["response"],
        sources=sources,
        tokens_used=result.get("tokens_used", 0),
        latency_ms=result.get("latency_ms", 0),
        cost_usd=result.get("cost_usd", 0),
        success=result.get("success", False),
        created_at=datetime.utcnow()
    )


@router.get("/history")
async def get_query_history(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get query history for the current user."""
    query_history_service = QueryHistoryService(db)
    
    history = query_history_service.get_query_history(
        user_id=current_user.id,
        limit=limit,
        offset=offset
    )
    
    return {
        "queries": [
            {
                "id": q.id,
                "query": q.query_text,
                "response": q.response_text,
                "success": q.success,
                "latency_ms": q.latency_ms,
                "created_at": q.created_at
            }
            for q in history
        ],
        "total": len(history)
    }


@router.post("/feedback/{query_id}")
async def submit_feedback(
    query_id: str,
    feedback_score: int,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Submit feedback for a query response."""
    if feedback_score < 1 or feedback_score > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback score must be between 1 and 5"
        )
    
    query_history_service = QueryHistoryService(db)
    success = query_history_service.update_feedback(query_id, feedback_score)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    return {"message": "Feedback recorded successfully"}
