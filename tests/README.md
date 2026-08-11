# Verification Profiles

- `tests/unit/`: dependency-light schema, configuration, path, and utility tests.
- `tests/contracts/`: dataset, radar tensor, model I/O, and artifact contracts.
- `tests/integration/`: two-batch CPU/GPU workflows using explicit fixtures.

The canonical verification suite runs inside the locked UV environment:

```bash
uv run ruff check src/mmprism tests
uv run mypy
uv run pytest
```
