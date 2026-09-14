# Repository instructions

- Use jj for version control. Use git for submodule operations.
- Manage `ext/duckdb` with picky.
- Build with `uv sync --all-groups --reinstall-package=ducky`.
- Test with `uv run --no-sync pytest -q`.
- Run `uvx prek run --all-files` before finishing.
- Put tests in `tests/`.
- Use C-style casts in C++.

See [DEVELOPMENT.md](DEVELOPMENT.md) for sanitizer commands.
