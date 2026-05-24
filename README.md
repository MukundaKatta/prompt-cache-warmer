# prompt-cache-warmer

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-cache-warmer.svg)](https://pypi.org/project/prompt-cache-warmer/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Pre-warm Anthropic prompt cache before user traffic hits.**

Anthropic charges 25% more on the first request that creates a cache entry and only 10% on subsequent reads. If you have a long system prompt or a tool list, you want the first cache write to be a cheap synthetic warmup at deploy time, not a real user paying for it (and waiting for it).

```python
from prompt_cache_warmer import Warmer
import anthropic

client = anthropic.Anthropic()
warmer = Warmer(client)

result = warmer.warm(
    model="claude-opus-4-7",
    system=LONG_SYSTEM_PROMPT,
    tools=MY_TOOLS,
    verify=True,
)

print(f"cache write: {result.cache_creation_input_tokens} tokens")
print(f"verified read: {result.verified_hit_tokens} tokens")
print(f"warm cost: ${result.cost_usd:.4f}")
```

That's it. `Warmer` injects `cache_control` breakpoints into your system blocks, fires a tiny `max_tokens=8` call, and optionally fires a second call to assert the cache actually read back.

## Install

```bash
pip install prompt-cache-warmer
pip install "prompt-cache-warmer[anthropic]"   # if you don't already have the SDK
```

## Why

The first request that creates a cache entry costs 1.25x the regular input price. Every subsequent request that reads it costs 0.10x. If your system prompt is 50k tokens, that first hit is the difference between a snappy response and a user waiting two seconds for the prompt to be processed from scratch.

Run `Warmer` from a deploy hook or a cron and the first user request hits a warm cache.

## API

```python
warmer = Warmer(client_or_callable, prices=None)

result = warmer.warm(
    *,
    model: str,
    system: str | list[dict],
    messages: list[dict] | None = None,   # default: a single user "ok"
    tools: Iterable[dict] | None = None,
    max_tokens: int = 8,
    breakpoints: int = 1,                 # number of cache_control markers
    verify: bool = False,                 # second call to confirm cache_read > 0
    ping_text: str = "ok",
) -> WarmResult
```

`client` accepts either an `anthropic.Anthropic()` instance or any `Callable[[dict], object]` so you can plug in a fake client or wrap a custom transport.

```python
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
    raw: list                              # the underlying responses
```

## Companion libraries

`prompt-cache-warmer` warms the cache. [`cachebench`](https://github.com/MukundaKatta/cachebench) measures the resulting hit ratio over time. Run the warmer once at deploy, then have `cachebench` keep score.

For runtime cost tracking, pair with [`claude-cost`](https://github.com/MukundaKatta/claude-cost) (Rust) or wire the same multipliers into your own logging.

## License

MIT
