"""Smoke tests for Databricks model serving endpoints."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

FOUNDATION_MODEL = "databricks-meta-llama-3-1-8b-instruct"
SENTIMENT_ENDPOINT = "distilbert-sentiment"


class TestFoundationModelServing:
    """Smoke tests against a foundation model endpoint."""

    def test_endpoint_exists_and_ready(self, workspace_client: WorkspaceClient):
        """The foundation model endpoint is listed and READY."""
        endpoints = workspace_client.serving_endpoints.list()
        names = {e.name for e in endpoints}
        assert FOUNDATION_MODEL in names

    def test_query_foundation_model(self, workspace_client: WorkspaceClient):
        """Can send a simple prompt and get a valid chat completion response."""
        response = workspace_client.serving_endpoints.query(
            name=FOUNDATION_MODEL,
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Say hello in exactly one word.")],
            max_tokens=10,
        )
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content.strip()) > 0

    def test_query_returns_usage_metadata(self, workspace_client: WorkspaceClient):
        """Response includes token usage metadata."""
        response = workspace_client.serving_endpoints.query(
            name=FOUNDATION_MODEL,
            messages=[ChatMessage(role=ChatMessageRole.USER, content="Reply with 'ok'.")],
            max_tokens=5,
        )
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


class TestSentimentEndpoint:
    """Smoke tests against the deployed distilbert-sentiment pyfunc endpoint."""

    def test_endpoint_exists(self, workspace_client: WorkspaceClient):
        """The sentiment model endpoint is listed."""
        endpoints = workspace_client.serving_endpoints.list()
        names = {e.name for e in endpoints}
        assert SENTIMENT_ENDPOINT in names, (
            f"Endpoint '{SENTIMENT_ENDPOINT}' not found. Deploy with: uv run python register_model.py --deploy"
        )

    def test_endpoint_is_ready(self, workspace_client: WorkspaceClient):
        """The sentiment endpoint is in READY state."""
        ep = workspace_client.serving_endpoints.get(name=SENTIMENT_ENDPOINT)
        state = ep.state.ready.value if ep.state and ep.state.ready else "UNKNOWN"
        assert state == "READY", f"Endpoint state is {state}, expected READY"

    def test_positive_sentiment(self, workspace_client: WorkspaceClient):
        """Positive text returns POSITIVE label with high confidence."""
        response = workspace_client.serving_endpoints.query(
            name=SENTIMENT_ENDPOINT,
            dataframe_records=[{"text": "I absolutely love this product, it's fantastic!"}],
        )
        predictions = response.predictions
        assert predictions is not None
        assert len(predictions) > 0
        pred = predictions[0]
        assert pred["label"] == "POSITIVE"
        assert pred["score"] > 0.9

    def test_negative_sentiment(self, workspace_client: WorkspaceClient):
        """Negative text returns NEGATIVE label with high confidence."""
        response = workspace_client.serving_endpoints.query(
            name=SENTIMENT_ENDPOINT,
            dataframe_records=[{"text": "This is terrible and I hate it."}],
        )
        predictions = response.predictions
        assert predictions is not None
        assert len(predictions) > 0
        pred = predictions[0]
        assert pred["label"] == "NEGATIVE"
        assert pred["score"] > 0.9

    def test_batch_prediction(self, workspace_client: WorkspaceClient):
        """Multiple texts return predictions for each input."""
        texts = [
            {"text": "Great experience, highly recommend!"},
            {"text": "Worst purchase I ever made."},
            {"text": "It was okay, nothing special."},
        ]
        response = workspace_client.serving_endpoints.query(
            name=SENTIMENT_ENDPOINT,
            dataframe_records=texts,
        )
        predictions = response.predictions
        assert predictions is not None
        assert len(predictions) == 3
        # First should be positive, second negative
        assert predictions[0]["label"] == "POSITIVE"
        assert predictions[1]["label"] == "NEGATIVE"
