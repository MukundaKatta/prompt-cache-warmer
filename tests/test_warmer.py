"""Tests for prompt_cache_warmer.Warmer."""

from __future__ import annotations

import pytest

from prompt_cache_warmer import (
    DEFAULT_PRICES,
    WarmResult,
    Warmer,
    add_cache_breakpoints,
    to_system_blocks,
)


# ---- block helpers ---------------------------------------------------------


def test_to_system_blocks_from_string():
    assert to_system_blocks("hi") == [{"type": "text", "text": "hi"}]


def test_to_system_blocks_passthrough_copies():
    blocks = [{"type": "text", "text": "x"}]
    out = to_system_blocks(blocks)
    assert out == blocks
    assert out is not blocks
    assert out[0] is not blocks[0]


def test_add_cache_breakpoints_zero_is_noop():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert add_cache_breakpoints(blocks, breakpoints=0) == blocks


def test_add_cache_breakpoints_default_marks_last_only():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = add_cache_breakpoints(blocks, breakpoints=1)
    assert "cache_control" not in out[0]
    assert out[1]["cache_control"] == {"type": "ephemeral"}


def test_add_cache_breakpoints_caps_at_four():
    blocks = [{"type": "text", "text": f"b{i}"} for i in range(10)]
    out = add_cache_breakpoints(blocks, breakpoints=99)
    marked = [b for b in out if "cache_control" in b]
    assert len(marked) == 4


def test_add_cache_breakpoints_preserves_existing_marker():
    blocks = [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "b"},
    ]
    out = add_cache_breakpoints(blocks, breakpoints=1)
    # both ends up cached: the first because we preserved, the last because
    # the breakpoint position landed on it.
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[1]["cache_control"] == {"type": "ephemeral"}


# ---- fake client -----------------------------------------------------------


class _FakeUsage:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeResp:
    def __init__(self, usage: _FakeUsage):
        self.usage = usage


class _FakeMessages:
    def __init__(self, scripts: list[_FakeResp]):
        self._scripts = list(scripts)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripts.pop(0)


class _FakeClient:
    def __init__(self, scripts: list[_FakeResp]):
        self.messages = _FakeMessages(scripts)


# ---- warm() ----------------------------------------------------------------


def test_warm_returns_warm_result_with_token_counts():
    client = _FakeClient(
        [
            _FakeResp(
                _FakeUsage(
                    input_tokens=10,
                    output_tokens=4,
                    cache_creation_input_tokens=12000,
                    cache_read_input_tokens=0,
                )
            ),
        ]
    )
    w = Warmer(client)
    out = w.warm(
        model="claude-opus-4-7",
        system="long system text",
    )
    assert isinstance(out, WarmResult)
    assert out.cache_creation_input_tokens == 12000
    assert out.cache_read_input_tokens == 0
    assert out.input_tokens == 10
    assert out.output_tokens == 4
    assert out.verified_hit_tokens is None
    assert out.latency_ms_warm >= 0


def test_warm_inserts_cache_control_on_last_block():
    client = _FakeClient([_FakeResp(_FakeUsage())])
    w = Warmer(client)
    w.warm(model="claude-opus-4-7", system="abc")
    sent = client.messages.calls[0]
    assert sent["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_warm_default_ping_user_message():
    client = _FakeClient([_FakeResp(_FakeUsage())])
    w = Warmer(client)
    w.warm(model="claude-opus-4-7", system="x")
    sent = client.messages.calls[0]
    assert sent["messages"] == [{"role": "user", "content": "ok"}]
    assert sent["max_tokens"] == 8


def test_warm_uses_supplied_messages_and_tools():
    client = _FakeClient([_FakeResp(_FakeUsage())])
    w = Warmer(client)
    w.warm(
        model="claude-opus-4-7",
        system="x",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "t", "input_schema": {"type": "object"}}],
    )
    sent = client.messages.calls[0]
    assert sent["messages"][0]["content"] == "hello"
    assert sent["tools"][0]["name"] == "t"


def test_warm_verify_returns_cache_read_tokens():
    client = _FakeClient(
        [
            _FakeResp(_FakeUsage(cache_creation_input_tokens=5000)),
            _FakeResp(_FakeUsage(cache_read_input_tokens=5000)),
        ]
    )
    w = Warmer(client)
    out = w.warm(model="claude-opus-4-7", system="x", verify=True)
    assert out.verified_hit_tokens == 5000
    assert out.latency_ms_verify is not None
    assert len(client.messages.calls) == 2


def test_warm_cost_estimate_with_default_prices():
    client = _FakeClient(
        [
            _FakeResp(
                _FakeUsage(
                    input_tokens=0,
                    output_tokens=10,
                    cache_creation_input_tokens=1_000_000,
                )
            ),
        ]
    )
    w = Warmer(client)
    out = w.warm(model="claude-opus-4-7", system="x")
    # 1M write tokens * $15/M * 1.25 = $18.75, + 10 out tokens * $75/M ~ 0.00075
    assert out.cost_usd is not None
    assert 18.7 <= out.cost_usd <= 18.8


def test_warm_cost_is_none_for_unknown_model():
    client = _FakeClient([_FakeResp(_FakeUsage(input_tokens=10))])
    w = Warmer(client)
    out = w.warm(model="some-unknown-model", system="x")
    assert out.cost_usd is None


def test_warmer_accepts_plain_callable():
    captured: list[dict] = []

    def fake_call(kwargs: dict):
        captured.append(kwargs)
        return _FakeResp(_FakeUsage(cache_creation_input_tokens=42))

    w = Warmer(fake_call)
    out = w.warm(model="claude-opus-4-7", system="x")
    assert captured and captured[0]["model"] == "claude-opus-4-7"
    assert out.cache_creation_input_tokens == 42


def test_warmer_rejects_invalid_client():
    with pytest.raises(TypeError):
        Warmer(object())


def test_default_prices_contains_known_models():
    assert "claude-opus-4-7" in DEFAULT_PRICES
    assert "claude-sonnet-4-6" in DEFAULT_PRICES
    assert "claude-haiku-4-5" in DEFAULT_PRICES


def test_warm_handles_dict_response_shape():
    # some clients return dicts instead of objects
    class _DictClient:
        class messages:
            calls: list[dict] = []

            @classmethod
            def create(cls, **kwargs):
                cls.calls.append(kwargs)
                return {
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                }

    w = Warmer(_DictClient())
    out = w.warm(model="claude-opus-4-7", system="x")
    assert out.input_tokens == 11
    assert out.cache_creation_input_tokens == 100


def test_add_cache_breakpoints_empty_is_noop():
    assert add_cache_breakpoints([], breakpoints=3) == []


def test_add_cache_breakpoints_marks_all_when_breakpoints_exceed_blocks():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = add_cache_breakpoints(blocks, breakpoints=2)
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[1]["cache_control"] == {"type": "ephemeral"}


def test_warm_handles_response_without_usage():
    class _NoUsageResp:
        pass

    def fake_call(kwargs: dict):
        return _NoUsageResp()

    w = Warmer(fake_call)
    out = w.warm(model="claude-opus-4-7", system="x")
    # missing usage degrades gracefully to zeros, not an error.
    assert out.input_tokens == 0
    assert out.cache_creation_input_tokens == 0
    assert out.cache_read_input_tokens == 0
    assert out.output_tokens == 0
    # cost is still computable (all zeros -> 0.0) for a known model.
    assert out.cost_usd == 0.0


def test_warm_respects_custom_price_table():
    client = _FakeClient([_FakeResp(_FakeUsage(input_tokens=1_000_000))])
    w = Warmer(client, prices={"claude-opus-4-7": {"input": 30.0, "output": 60.0}})
    out = w.warm(model="claude-opus-4-7", system="x")
    # 1M input tokens * $30/M = $30.00
    assert out.cost_usd is not None
    assert abs(out.cost_usd - 30.0) < 1e-6
