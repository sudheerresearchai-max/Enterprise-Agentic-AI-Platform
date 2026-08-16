# Enterprise Agentic AI Platform

## Overview
An end-to-end agentic AI platform built with Python, FastAPI, PostgreSQL, and Vector Database for enterprise use cases.

## Features
- **Document Ingestion**: Upload and process documents with intelligent chunking
- **Vector Search**: Semantic search using embeddings and vector database
- **Agentic AI**: LLM-powered agents with tool calling capabilities
- **Authentication**: Ready-to-use API authentication boundaries
- **Evaluation**: Comprehensive metrics for retrieval quality, faithfulness, latency, and cost
- **Enterprise Ready**: Logging, failure recovery, and production-style APIs

## Tech Stack
- **Backend**: Python 3.11+, FastAPI
- **Database**: PostgreSQL with pgvector
- **Vector DB**: Qdrant / pgvector
- **LLM**: OpenAI API compatible
- **Containerization**: Docker & Docker Compose

## Project Structure
```
├── app/
│   ├── api/          # API endpoints
│   ├── core/         # Core configuration
│   ├── models/       # Pydantic & SQLAlchemy models
│   ├── services/     # Business logic services
│   └── utils/        # Utility functions
├── data/
│   ├── documents/    # Uploaded documents
│   └── embeddings/   # Vector storage
├── tests/            # Test suite
├── scripts/          # Utility scripts
└── configs/          # Configuration files
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- OpenAI API key (or compatible LLM provider)

### Installation

1. Clone the repository
2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Start services with Docker:
```bash
docker-compose up -d
```

4. Run the application:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/register` - User registration

### Documents
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete document

### Query
- `POST /api/v1/query` - Submit query to agent
- `GET /api/v1/query/{id}` - Get query results

### Evaluation
- `POST /api/v1/evaluate` - Run evaluation suite
- `GET /api/v1/metrics` - Get performance metrics

## Evaluation Metrics
- **Retrieval Quality**: Precision@K, Recall@K, MRR
- **Answer Faithfulness**: Groundedness score
- **Latency**: P50, P95, P99 response times
- **Token Cost**: Cost per query analysis
- **Task Success Rate**: End-to-end success metrics

## License
MIT License
