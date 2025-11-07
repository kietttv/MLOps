"""Smoke tests for the Flask application factory."""

from __future__ import annotations

from app.app import create_app


def test_create_app() -> None:
    """Ensure the Flask application factory returns a configured app."""
    app = create_app()
    assert app.testing is False
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


