# Contributing to OpenJarvis

Thanks for your interest! OpenJarvis is in early alpha — every contribution counts.

## Where to start

| You want to... | File / area |
|---|---|
| Add a new LLM provider | `openjarvis/llm/providers/` — implement `BaseProvider` |
| Add a new tool | `openjarvis/tools/builtin/` or expose via MCP |
| Improve wake word detection | `openjarvis/wake/` |
| Improve ASR streaming | `openjarvis/asr/` |
| Fix a bug | open an issue first if it is non-trivial |

## Dev setup

```bash
git clone https://github.com/YOUR_USERNAME/OpenJarvis.git
cd OpenJarvis
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"
```

## Code style

- `ruff format .` before commit
- `ruff check .` must pass
- `mypy openjarvis` should pass (strict mode)
- All new code requires at least one unit test

## Architecture rule

**Modules must not import each other directly.** All cross-module communication goes through the Redis event bus. This keeps modules independently testable and swappable.

## Pull requests

- Branch from `main`
- Reference an issue in the description
- Keep PRs small and focused
- Add a line to `CHANGELOG.md` under `Unreleased`
