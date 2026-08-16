from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import aiofiles
from datetime import datetime

from app.models.database import get_db_session
from app.models.schemas import (
    DocumentResponse, DocumentList, DocumentUpload
)
from app.utils.deps import get_current_user
from app.models.database import User
from app.services.document_service import DocumentService
from app.services.vector_store import DocumentChunker, EmbeddingService, VectorStoreService
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Upload and process a document."""
    
    # Validate file type
    file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Read file content
    try:
        file_content = await file.read()
        file_size = len(file_content)
        
        # Check file size
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Save file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join("data", "documents", safe_filename)
        
        async with aiofiles.open(file_path, "wb") as out_file:
            await out_file.write(file_content)
        
        # Create document record
        doc_service = DocumentService(db)
        document = doc_service.create_document(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_ext,
            metadata={"original_filename": file.filename}
        )
        
        # Process document (chunking and embedding)
        try:
            text_content = file_content.decode("utf-8")
            
            # Chunk document
            chunker = DocumentChunker()
            chunks = chunker.chunk_text(
                text_content,
                metadata={
                    "document_id": document.id,
                    "filename": file.filename,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            # Generate embeddings
            embedding_service = EmbeddingService()
            texts = [chunk["content"] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            # Store in vector database
            vector_store = VectorStoreService()
            vector_store.upsert_vectors(chunks, embeddings)
            
            # Update document record with chunks
            doc_service.add_chunks(document.id, chunks)
            
            logger.info(
                "document_processed",
                document_id=document.id,
                chunks_count=len(chunks)
            )
            
        except Exception as e:
            logger.error("document_processing_failed", document_id=document.id, error=str(e))
            doc_service.update_document_status(document.id, "failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document processing failed: {str(e)}"
            )
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("document_upload_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("", response_model=DocumentList)
async def list_documents(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """List all documents for the current user."""
    doc_service = DocumentService(db)
    
    documents = doc_service.list_documents(
        user_id=current_user.id,
        status=status,
        limit=limit,
        offset=offset
    )
    
    total = len(documents)
    
    return DocumentList(
        documents=[
            DocumentResponse(
                id=doc.id,
                user_id=doc.user_id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                status=doc.status,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
            for doc in documents
        ],
        total=total
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Get a specific document by ID."""
    doc_service = DocumentService(db)
    document = doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check ownership
    if document.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return DocumentResponse(
        id=document.id,
        user_id=document.user_id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        status=document.status,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """Delete a document and its vectors."""
    doc_service = DocumentService(db)
    document = doc_service.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check ownership
    if document.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Delete from vector store
    try:
        vector_store = VectorStoreService()
        vector_store.delete_by_document_id(document_id)
    except Exception as e:
        logger.warning("vector_deletion_failed", document_id=document_id, error=str(e))
    
    # Delete from database
    doc_service.delete_document(document_id)
    
    return {"message": "Document deleted successfully"}
