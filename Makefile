.PHONY: help lint lint-check test verify check-claude-md update-claude-md

help:
	@echo "Available targets:"
	@echo "  make lint             - Auto-fix lint + format with ruff"
	@echo "  make lint-check       - Check lint + format without making changes"
	@echo "  make test             - Run the test suite"
	@echo "  make check-claude-md  - Verify CLAUDE.md managed section matches the collection SSOT"
	@echo "  make update-claude-md - Rewrite CLAUDE.md managed section from the collection SSOT"
	@echo "  make verify           - Run all checks (lint-check + test + check-claude-md)"

lint:
	uv run ruff check --fix .; uv run ruff format .

lint-check:
	uv run ruff format --check . && uv run ruff check .

test:
	uv run python -m unittest discover -s tests

verify: lint-check test check-claude-md

# Sync/check the managed top section of CLAUDE.md against the collection SSOT.
# Pass FROM=<path> to compare against a local checkout instead of GitHub (offline / pre-push).
check-claude-md:
	python3 tools/claude_md.py --check $(if $(FROM),--from $(FROM))

update-claude-md:
	python3 tools/claude_md.py --update $(if $(FROM),--from $(FROM))
