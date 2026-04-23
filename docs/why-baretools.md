---
title: Why Baretools? — Zero Dependencies, Minimal Attack Surface
---

# Why Baretools?

[← Home](.) &nbsp;·&nbsp; [Get Started](getting-started) &nbsp;·&nbsp; [Concepts](concepts) &nbsp;·&nbsp; [Providers](providers) &nbsp;·&nbsp; [API Reference](api-reference)

---

## The Hidden Cost of Heavy Frameworks

When you `pip install langchain`, you're not installing one package. You're pulling in a dependency tree that includes 50–100+ transitive packages: Pydantic, httpx, tenacity, SQLAlchemy, aiohttp, tiktoken, numpy, and more — many of which you'll never use.

Each of those packages is:

- A potential source of **CVEs** you need to track and patch
- A package whose maintainer could push a **compromised release** (see: `event-stream`, `xz-utils`, `polyfill.io`)
- A version constraint that **blocks your other deps** from upgrading
- Megabytes of code that **slows container builds** and expands your attack surface

### Baretools' Approach

```
$ pip install baretools-ai
$ pip show baretools-ai | grep Requires
Requires:
```

Nothing. Baretools uses only the Python standard library:

| Feature | Standard library module |
|---|---|
| Type hints and schema reflection | `inspect`, `typing` |
| Async execution | `asyncio` |
| Parallel execution | `concurrent.futures` |
| Structured types | `dataclasses`, `typing.TypedDict` |
| Logging / events | `logging` |
| JSON parsing | `json` |

## Dependency Comparison

The table below compares runtime dependency counts for popular tool-calling solutions. Figures are approximate as of April 2026 — install in a fresh venv with `pip list` to verify.

| Library | Runtime deps (approx.) | Install size (approx.) | Notes |
|---|---|---|---|
| **baretools-ai** | **0** | **~25 KB** | stdlib only |
| `langchain-core` | ~15 | ~5 MB | pydantic, httpx, tenacity, yaml... |
| `langchain` + `langchain-openai` | ~50 | ~30 MB | adds tiktoken, openai SDK, etc. |
| `llama-index-core` | ~30 | ~15 MB | pydantic, httpx, nltk, numpy... |
| `crewai` | ~40+ | ~20 MB | langchain, pydantic, litellm... |

## What We Deliberately Exclude

**Pydantic** — a great library, but not everyone needs it. If your tools accept `BaseModel` parameters, install it explicitly: `pip install "baretools-ai[pydantic]"`. Otherwise, standard-library `@dataclass` types work out-of-the-box with zero extra installs.

**httpx / requests** — Baretools never makes network calls. HTTP is your application's responsibility.

**tiktoken / tokenizers** — Token counting and context management are orchestration concerns that vary by application. They belong in your code.

**LangChain / LangGraph** — Baretools is not an alternative to the orchestration parts of those libraries. It replaces only the tool-wiring plumbing, which you can now handle yourself without pulling in the rest.

## Supply-Chain Risk in Practice

The [Python Package Index](https://pypi.org) has seen a sustained rise in typosquatting, dependency confusion, and maintainer account compromise attacks. The fewer packages in your dependency tree, the smaller your blast radius.

A zero-dependency library means:

- **No upstream compromise can reach your tool execution layer** via a transitive package
- **`pip-audit` and `safety` scans complete instantly** — there is nothing to scan
- **Renovate / Dependabot produce zero noise** for the baretools install itself
- **Security reviews are fast**: read `src/baretools/core.py` (~900 lines of pure Python) and you're done

## Philosophy

Baretools was designed around one constraint: if a feature requires a non-stdlib dependency, it does not go in core.

The corollary is that Baretools is intentionally small. It will never include:

- An LLM client (use the provider's official SDK)
- Prompt templates (use f-strings or your own engine)
- Agent orchestration (your loop, your rules)
- Vector stores, memory, or retrieval

If you need those things, excellent libraries exist for each. Install only what you need.
