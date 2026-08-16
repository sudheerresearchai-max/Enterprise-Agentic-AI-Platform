from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.dimension = 1536  # Default for text-embedding-3-small
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error("batch_embedding_generation_failed", error=str(e))
            raise


class DocumentChunker:
    """Service for chunking documents."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": self._generate_chunk_id(text, i),
                    "content": chunk,
                    "chunk_index": i,
                    "metadata": metadata or {}
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info("document_chunked", num_chunks=len(chunks))
            return chunk_dicts
        except Exception as e:
            logger.error("document_chunking_failed", error=str(e))
            raise
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate unique ID for a chunk."""
        hash_input = f"{text[:100]}-{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class VectorStoreService:
    """Service for vector storage and retrieval using Qdrant."""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embedding_service = EmbeddingService()
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("vector_collection_created", collection=self.collection_name)
        except Exception as e:
            logger.error("vector_collection_check_failed", error=str(e))
            raise
    
    def upsert_vectors(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> bool:
        """Upsert vectors and their metadata to Qdrant."""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=i,  # Will be replaced by Qdrant with actual ID
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "document_id": chunk["metadata"].get("document_id"),
                        "filename": chunk["metadata"].get("filename"),
                        "created_at": chunk["metadata"].get("created_at")
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info("vectors_upserted", count=len(points))
            return True
        except Exception as e:
            logger.error("vector_upsert_failed", error=str(e))
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=self._build_filter(filter_dict) if filter_dict else None
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": str(result.id),
                    "content": result.payload.get("content"),
                    "score": result.score,
                    "metadata": {
                        "document_id": result.payload.get("document_id"),
                        "filename": result.payload.get("filename"),
                        "chunk_index": result.payload.get("chunk_index")
                    }
                })
            
            logger.info("vector_search_completed", results_count=len(formatted_results))
            return formatted_results
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []
    
    def _build_filter(self, filter_dict: Dict) -> Any:
        """Build Qdrant filter from dictionary."""
        from qdrant_client.http import models
        
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                models.FieldCondition(
                    key=f"payload.{key}",
                    match=models.MatchValue(value=value)
                )
            )
        
        return models.Filter(must=conditions) if conditions else None
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """Delete vectors associated with a document."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="payload.document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info("document_vectors_deleted", document_id=document_id)
            return True
        except Exception as e:
            logger.error("document_vector_deletion_failed", error=str(e))
            return False
