"""prompt-cache-warmer - pre-warm Anthropic prompt cache before user traffic.

Anthropic charges 25% more on the first request that creates a cache entry
and 10% as much on subsequent reads. If user requests are slow or expensive
on the first hit of a new system prompt, you want that first hit to be a
cheap synthetic warmup, not a real user.

This library:

  1. Takes your long system prompt (string or block list) and a model name.
  2. Inserts up to N `cache_control` breakpoints in the right places.
  3. Fires a tiny warmup call (max_tokens=8 by default).
  4. Optionally fires a second verification call and asserts
     `cache_read_input_tokens > 0`.
  5. Returns a `WarmResult` with timings, token counts, and estimated cost.

Example:

    from prompt_cache_warmer import Warmer
    import anthropic

    client = anthropic.Anthropic()
    warmer = Warmer(client)

    result = warmer.warm(
        model="claude-opus-4-7",
        system=long_system_prompt,
        tools=my_tools,
        verify=True,
    )
    assert result.verified_hit_tokens and result.verified_hit_tokens > 0

The default `client` argument expects the Anthropic SDK shape
(`client.messages.create(**kwargs)`), but `Warmer` also accepts any
`Callable[[dict], object]` so you can plug in a fake client or a wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "Warmer",
    "WarmResult",
    "PriceTable",
    "DEFAULT_PRICES",
    "add_cache_breakpoints",
    "to_system_blocks",
]


# ---- pricing (USD per 1M tokens, write/read multipliers applied) -----------


PriceTable = dict[str, dict[str, float]]


DEFAULT_PRICES: PriceTable = {
    # rough public list prices as of 2026-04; users can override
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


# ---- result type -----------------------------------------------------------


@dataclass(frozen=True)
class WarmResult:
    model: str
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    input_tokens: int
    output_tokens: int
    latency_ms_warm: int
    verified_hit_tokens: int | None
    latency_ms_verify: int | None
    cost_usd: float | None
    raw: list[Any] = field(default_factory=list)


# ---- block helpers ---------------------------------------------------------


def to_system_blocks(system: str | list[dict]) -> list[dict]:
    """Coerce a system arg into the Anthropic block list shape."""
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    return [dict(b) for b in system]


def add_cache_breakpoints(
    blocks: list[dict],
    breakpoints: int = 1,
) -> list[dict]:
    """Add up to `breakpoints` ephemeral cache_control markers.

    Markers are placed at evenly-spaced indexes ending with the last block.
    Anthropic caps the number of breakpoints at 4 per request; we cap at 4
    here too. Blocks that already have a `cache_control` key are preserved.
    """
    if breakpoints <= 0 or not blocks:
        return [dict(b) for b in blocks]

    breakpoints = min(breakpoints, 4)
    n = len(blocks)
    if breakpoints >= n:
        positions = set(range(n))
    else:
        step = n / breakpoints
        positions = {int((i + 1) * step) - 1 for i in range(breakpoints)}

    out: list[dict] = []
    for i, b in enumerate(blocks):
        nb = dict(b)
        if i in positions and "cache_control" not in nb:
            nb["cache_control"] = {"type": "ephemeral"}
        out.append(nb)
    return out


# ---- callable normalizer ---------------------------------------------------


def _normalize_client(client_or_callable: Any) -> Callable[[dict], Any]:
    """Return a callable that takes kwargs as a dict and returns a Message."""
    if callable(client_or_callable) and not hasattr(
        client_or_callable, "messages"
    ):
        # already a (kwargs)->resp callable
        return client_or_callable  # type: ignore[return-value]

    messages = getattr(client_or_callable, "messages", None)
    if messages is None or not hasattr(messages, "create"):
        raise TypeError(
            "Warmer requires an Anthropic client (with .messages.create) or "
            "a Callable[[dict], object]; got %r" % (client_or_callable,)
        )

    def _call(kwargs: dict) -> Any:
        return messages.create(**kwargs)

    return _call


def _usage_get(usage: Any, key: str, default: int = 0) -> int:
    if usage is None:
        return default
    if isinstance(usage, dict):
        v = usage.get(key, default)
    else:
        v = getattr(usage, key, default)
    return int(v) if v is not None else default


# ---- main class ------------------------------------------------------------


class Warmer:
    """Pre-warm an Anthropic prompt cache.

    Args:
        client: an `anthropic.Anthropic()` instance, or any callable
            `(kwargs: dict) -> Message`.
        prices: optional `PriceTable` to override `DEFAULT_PRICES`.
    """

    def __init__(
        self,
        client: Any,
        prices: PriceTable | None = None,
    ) -> None:
        self._call = _normalize_client(client)
        self._prices: PriceTable = dict(prices) if prices else dict(DEFAULT_PRICES)

    def warm(
        self,
        *,
        model: str,
        system: str | list[dict],
        messages: list[dict] | None = None,
        tools: Iterable[dict] | None = None,
        max_tokens: int = 8,
        breakpoints: int = 1,
        verify: bool = False,
        ping_text: str = "ok",
    ) -> WarmResult:
        """Send a warmup call. Optionally verify with a second call.

        Returns a `WarmResult` with token counts, latency, and estimated cost.
        """
        system_blocks = add_cache_breakpoints(
            to_system_blocks(system), breakpoints=breakpoints
        )
        msgs = list(messages) if messages else [
            {"role": "user", "content": ping_text}
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_blocks,
            "messages": msgs,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)

        t0 = perf_counter()
        resp1 = self._call(kwargs)
        t1 = perf_counter()
        usage1 = _usage_get_usage(resp1)
        warm_ms = int((t1 - t0) * 1000)

        verified: int | None = None
        verify_ms: int | None = None
        raws: list[Any] = [resp1]
        if verify:
            t2 = perf_counter()
            resp2 = self._call(kwargs)
            t3 = perf_counter()
            verify_ms = int((t3 - t2) * 1000)
            usage2 = _usage_get_usage(resp2)
            verified = _usage_get(usage2, "cache_read_input_tokens", 0)
            raws.append(resp2)

        return WarmResult(
            model=model,
            cache_creation_input_tokens=_usage_get(
                usage1, "cache_creation_input_tokens", 0
            ),
            cache_read_input_tokens=_usage_get(
                usage1, "cache_read_input_tokens", 0
            ),
            input_tokens=_usage_get(usage1, "input_tokens", 0),
            output_tokens=_usage_get(usage1, "output_tokens", 0),
            latency_ms_warm=warm_ms,
            verified_hit_tokens=verified,
            latency_ms_verify=verify_ms,
            cost_usd=self._estimate_cost(model, usage1),
            raw=raws,
        )

    def _estimate_cost(self, model: str, usage: Any) -> float | None:
        price = self._prices.get(model)
        if not price:
            return None
        per_in = price["input"] / 1_000_000
        per_out = price["output"] / 1_000_000
        regular_in = _usage_get(usage, "input_tokens", 0)
        write_in = _usage_get(usage, "cache_creation_input_tokens", 0)
        read_in = _usage_get(usage, "cache_read_input_tokens", 0)
        out = _usage_get(usage, "output_tokens", 0)
        return (
            regular_in * per_in
            + write_in * per_in * CACHE_WRITE_MULTIPLIER
            + read_in * per_in * CACHE_READ_MULTIPLIER
            + out * per_out
        )


def _usage_get_usage(resp: Any) -> Any:
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp.get("usage")
    return getattr(resp, "usage", None)
