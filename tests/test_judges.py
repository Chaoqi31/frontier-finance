import pytest

from frontier_finance.judges import AnthropicJudge, Judge

_BUILD_KWARGS = {"max_tokens": 100, "timeout": 5.0, "max_retries": 0}


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        Judge.build("llama-3", **_BUILD_KWARGS)


@pytest.mark.parametrize(
    "model, rejects",
    [
        ("claude-opus-4-8", True),  # opus >= 4.7 rejects sampling params
        ("claude-opus-4-7", True),
        ("claude-opus-4-6", False),  # older opus accepts temperature=0
        ("claude-sonnet-5", True),  # sonnet >= 5 rejects
        ("claude-sonnet-4-6", False),
        ("claude-fable-1", True),
        ("claude-mythos-1", True),
        ("claude-haiku-4-5", False),
        ("claude-haiku-5-5", True),  # haiku >= 5 rejects
        ("claude-3-5-sonnet", False),  # unversioned-by-this-scheme -> accepts
    ],
)
def test_claude_rejects_sampling_params(model, rejects):
    assert AnthropicJudge._claude_rejects_sampling_params(model) is rejects


def test_build_routes_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    judge = Judge.build("claude-opus-4-8", **_BUILD_KWARGS)
    assert type(judge).__name__ == "AnthropicJudge"
    assert judge.model == "claude-opus-4-8"


def test_anthropic_judge_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Judge.build("claude-opus-4-8", **_BUILD_KWARGS)


class _Recorder:
    def __init__(self, reply):
        self.calls, self._reply = [], reply

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply(kwargs)


@pytest.mark.parametrize(
    "model, has_temperature",
    [
        ("gpt-5.4", True),
        ("gpt-5.6-luna", True),
        ("gpt-oss-120b", True),
        ("gpt-6-luna", False),
        ("o3", False),
    ],
)
def test_openai_temperature(monkeypatch, model, has_temperature):
    from types import SimpleNamespace

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    judge = Judge.build(model, **_BUILD_KWARGS)
    rec = _Recorder(
        lambda _: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )
    )
    monkeypatch.setattr(judge._client.chat.completions, "create", rec)
    judge.complete("s", "u")
    assert ("temperature" in rec.calls[0]) is has_temperature


def test_gemini_minimal_falls_back_to_low(monkeypatch):
    from types import SimpleNamespace

    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    judge = Judge.build("gemini-3.8-flash", **_BUILD_KWARGS)

    def reply(kwargs):
        if kwargs["config"].thinking_config.thinking_level.value == "MINIMAL":
            raise errors.ClientError(
                400,
                {
                    "error": {
                        "message": "Thinking level is unsupported: THINKING_LEVEL_MINIMAL"
                    }
                },
            )
        return SimpleNamespace(text="{}")

    rec = _Recorder(reply)
    monkeypatch.setattr(judge._client.models, "generate_content", rec)
    judge.complete("s", "u")
    judge.complete("s", "u")
    levels = [c["config"].thinking_config.thinking_level.value for c in rec.calls]
    assert levels == ["MINIMAL", "LOW", "LOW"]  # learns once, then skips MINIMAL
