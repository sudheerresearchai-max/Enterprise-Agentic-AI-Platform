import pytest
import asyncio
from httpx import AsyncClient
from datetime import datetime

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_check():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


@pytest.mark.anyio
async def test_root_endpoint():
    """Test root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


@pytest.mark.anyio
async def test_register_user():
    """Test user registration."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        user_data = {
            "email": f"test_{datetime.utcnow().timestamp()}@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
        response = await ac.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["email"] == user_data["email"]


@pytest.mark.anyio
async def test_login():
    """Test user login."""
    email = f"login_test_{datetime.utcnow().timestamp()}@example.com"
    
    # First register a user
    async with AsyncClient(app=app, base_url="http://test") as ac:
        register_data = {
            "email": email,
            "password": "testpassword123",
            "full_name": "Test User"
        }
        await ac.post("/api/v1/auth/register", json=register_data)
        
        # Then login
        login_data = {
            "email": email,
            "password": "testpassword123"
        }
        response = await ac.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


@pytest.mark.anyio
async def test_invalid_login():
    """Test login with invalid credentials."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        response = await ac.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401


@pytest.mark.anyio
async def test_documents_list_requires_auth():
    """Test that document listing requires authentication."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/documents")
        assert response.status_code == 401


@pytest.mark.anyio
async def test_query_requires_auth():
    """Test that query endpoint requires authentication."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        query_data = {"query": "test question"}
        response = await ac.post("/api/v1/query", json=query_data)
        assert response.status_code == 401


# Integration tests would require actual database and vector store
# These are example placeholders for more comprehensive testing

@pytest.mark.skip(reason="Requires database setup")
@pytest.mark.anyio
async def test_document_upload_flow():
    """Test complete document upload flow."""
    # This would test the full flow of:
    # 1. Register user
    # 2. Login
    # 3. Upload document
    # 4. Verify processing
    pass


@pytest.mark.skip(reason="Requires LLM API key")
@pytest.mark.anyio
async def test_query_with_rag():
    """Test query with RAG functionality."""
    # This would test:
    # 1. Upload documents
    # 2. Submit query
    # 3. Verify response with sources
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
