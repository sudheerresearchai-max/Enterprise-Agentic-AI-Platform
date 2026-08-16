from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Dict, Any, Optional
import time
import json
from datetime import datetime

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.vector_store import EmbeddingService, VectorStoreService

logger = get_logger(__name__)


class AgentService:
    """Service for LLM-powered agent with tool calling and RAG."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        
        # Cost tracking (approximate per 1K tokens)
        self.cost_per_1k_input = 0.01  # $0.01 per 1K input tokens
        self.cost_per_1k_output = 0.03  # $0.03 per 1K output tokens
    
    def query_with_rag(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a query using RAG (Retrieval Augmented Generation)."""
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Step 3: Build context from retrieved chunks
            context = self._build_context(retrieved_chunks)
            
            # Step 4: Generate response with LLM
            response, tokens_used = self._generate_response(
                query=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            cost_usd = self._calculate_cost(tokens_used)
            
            # Build response
            result = {
                "response": response,
                "sources": retrieved_chunks if include_sources else [],
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "success": True
            }
            
            logger.info(
                "query_processed",
                query_length=len(query),
                chunks_retrieved=len(retrieved_chunks),
                tokens_used=tokens_used,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost_usd, 6)
            )
            
            return result
            
        except Exception as e:
            logger.error("query_processing_failed", error=str(e))
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": [],
                "tokens_used": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_usd": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return "No relevant documents found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}]: {chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[str, int]:
        """Generate response using LLM with RAG context."""
        
        system_template = """You are an intelligent assistant for an enterprise AI platform. 
Your task is to answer questions based ONLY on the provided context from retrieved documents.

IMPORTANT RULES:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information, say so clearly
3. Cite your sources when possible (e.g., "According to Source 1...")
4. Be concise but thorough
5. Do not make up information or hallucinate

Context:
{context}

Remember: Always ground your answers in the provided context."""

        human_template = """Question: {query}

Please provide a helpful, accurate answer based on the context above."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Create chain
        chain = prompt | self.llm
        
        # Generate response
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        
        response_message = chain.invoke({"query": query, "context": context}, **llm_kwargs)
        
        # Extract token usage (if available)
        tokens_used = 0
        if hasattr(response_message, 'response_metadata'):
            usage = response_message.response_metadata.get('token_usage', {})
            tokens_used = usage.get('total_tokens', 0)
        
        # Estimate tokens if not provided
        if tokens_used == 0:
            tokens_used = self._estimate_tokens(context) + self._estimate_tokens(response_message.content)
        
        return response_message.content, tokens_used
    
    def _calculate_cost(self, tokens_used: int) -> float:
        """Calculate approximate cost based on token usage."""
        # Rough estimate: assume 80% input, 20% output
        input_tokens = int(tokens_used * 0.8)
        output_tokens = int(tokens_used * 0.2)
        
        cost = (input_tokens / 1000) * self.cost_per_1k_input + \
               (output_tokens / 1000) * self.cost_per_1k_output
        
        return cost
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    async def process_batch_queries(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        for query in queries:
            result = self.query_with_rag(query, top_k=top_k)
            results.append(result)
        return results
