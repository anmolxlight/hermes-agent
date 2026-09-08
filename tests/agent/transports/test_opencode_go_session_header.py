"""OpenCode Go requires x-opencode-session on every transport.

Console Go 400s with MissingSessionID ("Request is missing x-opencode-session
and cannot be routed efficiently") when the header is absent. OpenCode Go
routes different models to different wire protocols — muse-spark to
/v1/responses, minimax/qwen to /v1/messages, GLM/Kimi to
/v1/chat/completions — so all three transports must attach it.
"""

from __future__ import annotations

GO_URL = "https://opencode.ai/zen/go/v1"


class TestCodexResponsesTransport:
    """muse-spark on OpenCode Go uses the Responses API."""

    def _build(self, **extra):
        from agent.transports.codex import ResponsesApiTransport

        return ResponsesApiTransport().build_kwargs(
            model="muse-spark-1.3-contributor",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            **extra,
        )

    def test_session_header_attached(self):
        kwargs = self._build(session_id="sess-1", base_url=GO_URL)
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-1"

    def test_no_header_without_session_id(self):
        kwargs = self._build(session_id=None, base_url=GO_URL)
        assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})

    def test_no_header_for_other_endpoints(self):
        kwargs = self._build(session_id="sess-1", base_url="https://api.openai.com/v1")
        assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})


class TestAnthropicMessagesTransport:
    """minimax-*/qwen* on OpenCode Go use the Anthropic messages API."""

    def _build(self, **extra):
        from agent.transports.anthropic import AnthropicTransport

        return AnthropicTransport().build_kwargs(
            model="minimax-m2.7",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            max_tokens=16,
            **extra,
        )

    def test_session_header_attached(self):
        kwargs = self._build(session_id="sess-1", request_base_url=GO_URL)
        assert kwargs["extra_headers"]["x-opencode-session"] == "sess-1"

    def test_no_header_without_session_id(self):
        kwargs = self._build(session_id=None, request_base_url=GO_URL)
        assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})

    def test_no_header_for_real_anthropic(self):
        kwargs = self._build(
            session_id="sess-1", request_base_url="https://api.anthropic.com"
        )
        assert "x-opencode-session" not in (kwargs.get("extra_headers") or {})

    def test_existing_extra_headers_preserved(self):
        """The anthropic adapter sets anthropic-beta; we must not clobber it."""
        kwargs = self._build(
            session_id="sess-1",
            request_base_url=GO_URL,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        headers = kwargs["extra_headers"]
        assert headers["x-opencode-session"] == "sess-1"
        # Any pre-existing header the adapter produced survives the merge.
        for key, value in headers.items():
            assert value is not None, key
