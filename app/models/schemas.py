from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============ Authentication Schemas ============

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None


# ============ Document Schemas ============

class DocumentBase(BaseModel):
    filename: str
    file_type: str


class DocumentUpload(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: str
    user_id: str
    file_size: int
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ============ Query Schemas ============

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=100, le=8192)


class SourceChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    id: str
    query: str
    response: str
    sources: Optional[List[SourceChunk]] = None
    tokens_used: int
    latency_ms: float
    cost_usd: float
    success: bool
    created_at: datetime


# ============ Evaluation Schemas ============

class EvaluationRequest(BaseModel):
    test_set_id: Optional[str] = None
    metrics: List[str] = ["retrieval", "faithfulness", "latency", "cost", "success"]


class MetricResult(BaseModel):
    name: str
    value: float
    type: str
    description: str
    metadata: Dict[str, Any] = {}


class EvaluationResponse(BaseModel):
    evaluation_id: str
    test_set_id: str
    metrics: List[MetricResult]
    overall_score: float
    completed_at: datetime


class MetricsSummary(BaseModel):
    retrieval_precision: float
    retrieval_recall: float
    faithfulness_score: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_usd: float
    success_rate: float
    total_queries: int


# ============ Health Check ============

class HealthStatus(BaseModel):
    status: str
    version: str
    database: str
    vector_db: str
    llm_api: str
