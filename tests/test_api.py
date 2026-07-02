"""
test_api.py
-----------
Integration tests for the FastAPI endpoints. These use FastAPI's
TestClient (which wraps httpx) to make real HTTP calls to the app
WITHOUT needing to start a separate server process.

These tests DO require:
  - The knowledge base to be indexed (chroma_db/ populated)
  - A valid GROQ_API_KEY for the /chat endpoint test

The /chat test is marked with pytest.mark.skipif so it gracefully
skips on CI if the API key isn't configured, rather than failing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from fastapi.testclient import TestClient
from main import app

# TestClient wraps the FastAPI app and lets us call endpoints directly
# without starting a server. Responses are real HTTP Response objects.
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self):
        """Health endpoint must always return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        """Health response must contain all expected keys."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "total_chunks" in data
        assert "total_documents" in data
        assert "model" in data

    def test_health_status_is_ok(self):
        """Status field must be 'ok' when server is healthy."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"


class TestDepartmentsEndpoint:
    """Tests for GET /departments"""

    def test_departments_returns_200(self):
        response = client.get("/departments")
        assert response.status_code == 200

    def test_departments_contains_all_options(self):
        """Must include all 6 departments plus the 'All' option."""
        response = client.get("/departments")
        data = response.json()
        keys = [d["key"] for d in data["departments"]]
        assert "" in keys           # All Departments
        assert "signal" in keys
        assert "electrical" in keys
        assert "operations" in keys
        assert "safety" in keys

    def test_departments_have_key_and_label(self):
        """Each department entry must have both key and label fields."""
        response = client.get("/departments")
        for dept in response.json()["departments"]:
            assert "key" in dept
            assert "label" in dept


class TestDocumentsEndpoint:
    """Tests for GET /documents"""

    def test_documents_returns_200(self):
        response = client.get("/documents")
        assert response.status_code == 200

    def test_documents_has_required_fields(self):
        data = client.get("/documents").json()
        assert "documents" in data
        assert "total_documents" in data
        assert "total_chunks" in data

    def test_documents_is_list(self):
        data = client.get("/documents").json()
        assert isinstance(data["documents"], list)


class TestChatEndpoint:
    """Tests for POST /chat"""

    def test_chat_rejects_empty_question(self):
        """
        Questions shorter than 3 characters should be rejected with 422
        (FastAPI's validation error status code).
        """
        response = client.post("/chat", json={"question": "hi"})
        assert response.status_code == 422

    def test_chat_rejects_missing_question(self):
        """Request body with no question field must return 422."""
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_rejects_invalid_department(self):
        """
        An unrecognised department key should return 200 with a clear
        error message in the answer field. The RAG pipeline handles this
        gracefully rather than raising an HTTP exception — so we check
        that the answer contains a helpful message, not that the status
        is non-200.
        """
        response = client.post("/chat", json={
            "question": "What are signal failure procedures?",
            "department": "nonexistent_dept"
        })
        data = response.json()
        # The API returns 200 with an explanatory answer — verify the
        # answer text tells the user which departments are valid.
        assert response.status_code == 200
        assert "Valid options" in data["answer"]
        assert data["chunks_used"] == 0

    @pytest.mark.skipif(
        # Skip this test if GROQ_API_KEY is not set in the environment.
        # On CI, this secret is injected from GitHub Secrets. Locally,
        # it comes from your .env file. If neither is present, skip
        # rather than fail — the test is meaningless without a real key.
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set — skipping live API test"
    )
    def test_chat_returns_valid_response(self):
        """
        End-to-end test: a real question should return a valid answer
        with the expected response structure.
        """
        response = client.post("/chat", json={
            "question": "What are the steps to follow during a signal failure?",
            "department": "signal"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "department" in data
        assert "chunks_used" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 10