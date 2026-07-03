#!/usr/bin/env python3
"""Keep the managed top section of this repo's CLAUDE.md in sync with the collection SSOT.

The single source of truth is
  https://github.com/c4ffein/c4ffein/blob/main/guidelines/python-cli-utils/CLAUDE.md
whose **last line** is a marker containing the token `c4ffein:end-managed`. That SSOT (marker
included) is copied verbatim into each tool repo's CLAUDE.md as the "managed" section; whatever
the repo adds BELOW the marker is its own notes and is never touched.

  python3 tools/claude_md.py --check                 # CI: fail if managed section is stale
  python3 tools/claude_md.py --update                # rewrite managed section from source
  python3 tools/claude_md.py --check  --from PATH     # compare against a local SSOT copy
  python3 tools/claude_md.py --update --from PATH     # (offline / pre-push bootstrap)

This script is itself vendored into each repo; edit the canonical copy in
c4ffein/c4ffein/guidelines/tools/ and re-vendor if it ever changes (it rarely does).
"""

import sys
from pathlib import Path
from urllib.request import urlopen

# The managed section always ends with the marker line; we locate the split by this HTML-comment
# prefix (not the bare token) so the guidance can mention `c4ffein:end-managed` in prose freely.
MARKER_MATCH = "<!-- c4ffein:end-managed"
SOURCE_URL = "https://raw.githubusercontent.com/c4ffein/c4ffein/main/guidelines/python-cli-utils/CLAUDE.md"
STARTER_TAIL = "\n# Project-specific notes\n\n_None yet._\n"

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"


def normalize(text: str) -> str:
    """Unify line endings and trailing whitespace so comparisons are byte-stable."""
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


def fetch_source(local_path: str | None) -> str:
    if local_path:
        return Path(local_path).read_text(encoding="utf-8")
    with urlopen(SOURCE_URL, timeout=30) as response:  # fixed https URL, not user input
        return response.read().decode("utf-8")


def split_after_marker(text: str):
    """Return (managed_including_marker, tail) or (None, None) if the marker is absent."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith(MARKER_MATCH):
            return "".join(lines[: i + 1]), "".join(lines[i + 1 :])
    return None, None


def main(argv: list[str]) -> int:
    do_check, do_update = "--check" in argv, "--update" in argv
    if do_check == do_update:
        print("usage: claude_md.py (--check | --update) [--from PATH]", file=sys.stderr)
        return 2
    local_path = argv[argv.index("--from") + 1] if "--from" in argv else None

    source = normalize(fetch_source(local_path))
    if not source.rstrip("\n").endswith("-->") or MARKER_MATCH not in source:
        print("✗ source SSOT does not end with the c4ffein:end-managed marker", file=sys.stderr)
        return 1

    if not CLAUDE_MD.exists():
        if do_check:
            print(f"✗ {CLAUDE_MD.name} is missing", file=sys.stderr)
            return 1
        CLAUDE_MD.write_text(source + STARTER_TAIL, encoding="utf-8")
        print(f"✓ created {CLAUDE_MD.name} from source")
        return 0

    managed, tail = split_after_marker(CLAUDE_MD.read_text(encoding="utf-8"))
    if managed is None:
        print(f"✗ no `{MARKER_MATCH}…` marker line in {CLAUDE_MD.name}", file=sys.stderr)
        return 1

    if do_check:
        if normalize(managed) == source:
            print(f"✓ {CLAUDE_MD.name} managed section is up to date")
            return 0
        print(
            f"✗ {CLAUDE_MD.name} managed section is STALE — run `make update-claude-md`",
            file=sys.stderr,
        )
        return 1

    CLAUDE_MD.write_text(source + tail, encoding="utf-8")
    print(f"✓ updated managed section of {CLAUDE_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
