"""Minimal Flask application serving the MLOps dashboard."""

from __future__ import annotations

from flask import Flask, render_template


def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        """Render the landing page for the application."""
        return render_template("index.html")

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=8000, debug=True)


