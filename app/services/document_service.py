from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from app.models.database import Document, DocumentChunk, QueryHistory, EvaluationMetric
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Service for document management."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_document(
        self,
        user_id: str,
        filename: str,
        file_path: str,
        file_size: int,
        file_type: str,
        metadata: Optional[Dict] = None
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            id=str(uuid.uuid4()),
            user_id=user_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            status="pending",
            chunk_count=0,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        
        logger.info("document_created", document_id=doc.id, filename=filename)
        return doc
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def list_documents(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """List documents with optional filters."""
        query = self.db.query(Document)
        
        if user_id:
            query = query.filter(Document.user_id == user_id)
        if status:
            query = query.filter(Document.status == status)
        
        return query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    
    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None
    ) -> bool:
        """Update document processing status."""
        doc = self.get_document(document_id)
        if not doc:
            return False
        
        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        doc.updated_at = datetime.utcnow()
        
        self.db.commit()
        logger.info("document_status_updated", document_id=document_id, status=status)
        return True
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        doc = self.get_document(document_id)
        if not doc:
            return False
        
        self.db.delete(doc)
        self.db.commit()
        logger.info("document_deleted", document_id=document_id)
        return True
    
    def add_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """Add chunks to a document."""
        doc = self.get_document(document_id)
        if not doc:
            return 0
        
        chunk_objects = []
        for i, chunk_data in enumerate(chunks):
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk_data.get("chunk_index", i),
                content=chunk_data["content"],
                embedding=chunk_data.get("embedding"),
                metadata=chunk_data.get("metadata", {})
            )
            chunk_objects.append(chunk)
        
        self.db.add_all(chunk_objects)
        doc.chunk_count = len(chunks)
        doc.status = "completed"
        doc.updated_at = datetime.utcnow()
        
        self.db.commit()
        logger.info("chunks_added", document_id=document_id, count=len(chunks))
        return len(chunks)


class QueryHistoryService:
    """Service for query history tracking."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_query(
        self,
        user_id: str,
        query_text: str,
        response_text: str,
        retrieved_chunks: List[Dict],
        tokens_used: int,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> QueryHistory:
        """Log a query to history."""
        query_record = QueryHistory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            query_text=query_text,
            response_text=response_text,
            retrieved_chunks=[
                {
                    "id": chunk.get("id"),
                    "content": chunk.get("content")[:500],  # Truncate for storage
                    "score": chunk.get("score"),
                    "metadata": chunk.get("metadata")
                }
                for chunk in retrieved_chunks
            ],
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=success,
            error_message=error_message,
            created_at=datetime.utcnow()
        )
        
        self.db.add(query_record)
        self.db.commit()
        self.db.refresh(query_record)
        
        logger.info("query_logged", query_id=query_record.id, success=success)
        return query_record
    
    def get_query_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[QueryHistory]:
        """Get query history with optional filters."""
        query = self.db.query(QueryHistory)
        
        if user_id:
            query = query.filter(QueryHistory.user_id == user_id)
        
        return query.order_by(QueryHistory.created_at.desc()).offset(offset).limit(limit).all()
    
    def update_feedback(
        self,
        query_id: str,
        feedback_score: int
    ) -> bool:
        """Update feedback score for a query."""
        query_record = self.db.query(QueryHistory).filter(
            QueryHistory.id == query_id
        ).first()
        
        if not query_record:
            return False
        
        query_record.feedback_score = feedback_score
        self.db.commit()
        return True


class EvaluationService:
    """Service for evaluation metrics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_type: str,
        test_set_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> EvaluationMetric:
        """Record an evaluation metric."""
        metric = EvaluationMetric(
            id=str(uuid.uuid4()),
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            test_set_id=test_set_id,
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        logger.info("metric_recorded", name=metric_name, value=metric_value)
        return metric
    
    def get_metrics_summary(
        self,
        test_set_id: Optional[str] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """Get summary of evaluation metrics."""
        query = self.db.query(EvaluationMetric)
        if test_set_id:
            query = query.filter(EvaluationMetric.test_set_id == test_set_id)
        
        metrics = query.order_by(EvaluationMetric.created_at.desc()).limit(limit).all()
        
        # Calculate summaries by type
        summaries = {}
        for metric_type in ["retrieval", "faithfulness", "latency", "cost", "success"]:
            type_metrics = [m for m in metrics if m.metric_type == metric_type]
            if type_metrics:
                values = [m.metric_value for m in type_metrics]
                summaries[f"{metric_type}_avg"] = sum(values) / len(values)
                summaries[f"{metric_type}_count"] = len(values)
        
        return {
            "total_metrics": len(metrics),
            "summaries": summaries
        }
