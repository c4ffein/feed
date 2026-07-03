# CLAUDE.md — Python CLI utils collection

Shared guidance for working on c4ffein's collection of small, single-file Python CLI tools.
**This file is the single source of truth.** It is vendored — copied verbatim — into the
`CLAUDE.md` of every tool repo, above a marker line (see "How this file is distributed" at the
bottom). Edit it **here, in `c4ffein/c4ffein`**, never in a downstream copy.

- **Project catalog:** [`../../repos/repos-python-cli-utils.md`](../../repos/repos-python-cli-utils.md)
  (published: https://github.com/c4ffein/c4ffein/blob/main/repos/repos-python-cli-utils.md)

Each tool is its own git repository. This file captures only what is **shared**; each repo's
own project-specific notes live **below the marker line** in that repo's `CLAUDE.md`.

## The collection

Paths are `repo/entrypoint`; inside a solo checkout the repo prefix is implicit.

| Project | Entrypoint | Purpose |
|---|---|---|
| **fv** | `fv/fv.py` | End-to-end encrypted file storage via GPG (per-file passphrase, UUID index) |
| **ai** | `ai/ai.py` | Zero-dependency Claude API bridge to the terminal |
| **2fa** | `2fa/2fa.py` | TOTP/HOTP generator, `rsc/2fa`-compatible `~/.2fa` format |
| **bank** | `bank/bank.py` | Qonto banking client (transactions, invoice upload/verify) |
| **sm** | `sm/sm.py` | IMAP/SMTP mail client + reader UI + `git send-email`-style patch sending |
| **crypto** | `crypto/crypto.py` | Manual TLS certificate-chain verifier (educational) |
| **pass** | `pass/pass.py` | Single-file reimplementation of `password-store` |
| **capucintype** | `capucintype/capucintype.py` | Monkeytype-style CLI typing test |
| **qrcode** | `qrcode/qrcode.py` | Standalone QR generator, AST-bundled from `python-qrcode` via `bundler.py` |
| **companion** | `companion/src/companion.py` | Trusted-LAN file-sharing server + CLI + web UI (built to root `companion.py`) |
| **feed** | `feed/feed.py` | Concurrent RSS/Atom reader, HTML→ANSI rendering, persistent cache |
| **containerctl** | `containerctl/containerctl.py` | Docker manager CLI + hand-rolled TUI, TOML projects/services |
| **cloud** | `cloud/cloud.py` | OVH + Scaleway personal-infra manager (stdlib async over urllib) |

## Shared philosophy (KISS)

- **Single file, standard library only.** No runtime dependencies unless unavoidable
  (`crypto` needs `cryptography`; everything else is pure stdlib). Do not add a
  dependency to solve something the stdlib already does.
- **Personal proofs-of-concept.** Every entrypoint carries an honest "I don't recommend
  using this as-is" header. Preserve that framing; don't overstate maturity.
- **Enhance when needed, not upfront.** Standards vary by project — some are polished
  (`sm`, `companion`), some are throwaway (`capucintype`). Match the project you're in.

## House style (follow when editing any project)

- **Atomic persistence.** Every disk write goes through a temp file + `os.replace`/`rename`
  (add `fsync` where durability matters). Never write directly to a live index/config/state
  file. Use a lock file (`open(..., "x")` or `fcntl`/`msvcrt` flock) for single-instance
  or concurrent-write paths.
- **TLS certificate pinning.** The network clients (`ai`, `bank`, `cloud`, `sm`, `companion`)
  share `make_pinned_ssl_context`, which subclasses `SSLSocket.do_handshake` to verify a
  SHA-256 pin. It reaches into private stdlib internals (`_ssl`, `_ASN1Object`) — when you
  touch it, keep the shared shape identical across projects and re-test on the target
  Python versions. Pin **in addition to** normal CA + hostname verification, never instead.
- **Config vs state split.** Read-only config lives under `~/.config/<tool>/`; mutable,
  tool-owned state under `~/.local/state/<tool>/` or `~/.cache/<tool>/` (XDG). Don't write
  back into the config file at runtime.
- **Errors flow through one channel.** Where a project defines an error sink
  (e.g. `ctx.record_error` in `sm`, the `_errors` registry in `containerctl`), record errors
  there and let the UI surface them — do not `print` to stderr ad hoc or swallow with a
  bare `except:`.
- **Validate untrusted input with allowlists.** Charset-validate anything that reaches a
  URL, a subprocess argv, or a filesystem path before use (`is_valid_id`, `_validate_*`,
  path-traversal guards). Enforce at both the parser and the call site (defense in depth).
- **Keep the README in sync via a test.** The `--help`/usage output is asserted against the
  README code block (`test-readme`, `test_help`, etc.). If you change usage text, update the
  README in the same change or the test fails — this is intentional.

## Before declaring work "ready"

Run the project's read-only gate, then stop — **the user does the commits, never auto-commit.**

- `make verify` — the standard read-only "is this branch ready?" gate in **every** project
  (lint-check + format-check + tests, where each applies). A few are aliases: `companion`'s
  `verify` runs `check` (which also does icon-sync + format/lint on `src/`); `pass` and
  `qrcode` `verify` run the test suite; `ai` `verify` runs `lint-check` + `test-readme`
  (its `test` target has no `test.py` yet).
- `make lint-check` — `ruff check --no-fix` (verify, don't mutate).
- `make lint` / `make format` — auto-fix during iteration (mutates files).
- `make test` — tests only. Integration suites that stand up a real TLS fake server
  (`sm`, `companion`) need `openssl` and skip cleanly without it.
- **`companion` is special:** after editing `src/companion.py` you must also run
  `make build` (inlines `index.html` + PDF.js into the root `companion.py`); both the
  source and built file are committed. Never hand-edit the built `companion.py`.
- **`qrcode` is special:** never hand-edit `qrcode.py` — it is generated. Edit the
  upstream submodule or `bundler.py` and run `make generate` (CI enforces the bundle is
  reproducible via a regen `git diff --exit-code`).

**Prefer ephemeral tool runs over a managed venv.** No venv to activate, no lockfile to
babysit — that ceremony is the thing this collection avoids. Ruff is pinned per project
(mostly `0.5.1`; newer in `cloud`/`feed`), so run it pinned and throwaway with `uvx`:

```
uvx ruff@0.5.1 check --no-fix .     # use the project's own pinned version
uvx ruff@0.5.1 format --check .
```

Stdlib-only test suites just run under bare `python3 -m unittest`. For the few projects that
need a third-party package, pull it in for that single run instead of building an environment:
`uv run --with cryptography ...` (`crypto`), `uv run --with hypothesis --with pytest ...`
(`fv`). Reach for `uv venv` + activate only when you genuinely need a persistent environment —
for these single-file tools you normally don't.

## Recurring pitfalls — check these when touching a project

These issues recur across the collection; watch for and prefer to fix them:

- **Python floor is `>=3.11` across the collection** (`crypto` is `>=3.12` for its own
  reasons). Keep `requires-python`, ruff `target-version`, ty `python-version`, and the CI
  matrix all aligned to that floor when you touch a project — they have drifted before.
- **CI tests a single Python version** (`ai`, `fv`, `sm`, `crypto`, `qrcode`) despite broader
  support claims. Prefer a matrix (`3.11`–`3.13`) that matches `requires-python`.
- **Use `secrets`, not `random`, for anything security-sensitive** (passphrases, tokens).
  `random` is Mersenne-Twister and predictable.
- **Plaintext secrets at rest.** Config files hold API keys / passwords unencrypted. At
  minimum create them `0o600` up front (via `os.open`), not `chmod` after writing.
- **Private stdlib internals** (`_ssl`, `_ASN1Object`) can break across Python releases —
  the pinning code is the main exposure; re-test on upgrades.
- **Author email** should be consistent (`c4ffein.work@gmail.com`); some `pyproject.toml`
  files still say `c4ffein@gmail.com`.

## How this file is distributed

This section, and everything above it — **including the marker line at the very bottom of this
file** — is **managed content**, copied verbatim into each tool repo's `CLAUDE.md`. The marker
line contains the token `c4ffein:end-managed`; everything a tool repo adds **below** it is that
project's own guidance and is never touched by the sync.

- `make update-claude-md` — rewrite the managed section of the local `CLAUDE.md` from this
  source (fetched from `raw.githubusercontent.com/c4ffein/c4ffein/main/...`), preserving the
  marker line and everything below it.
- `make check-claude-md` — CI gate: fail if the managed section drifts from this source.
  Runs `git diff`-style enforcement, the same idea as `qrcode`'s bundle regen check.

To change the shared guidance, edit **this file** here in `c4ffein/c4ffein`, push, then run
`make update-claude-md` in each tool repo (their `check-claude-md` CI will go red until you do).

<!-- c4ffein:end-managed — everything ABOVE is synced from c4ffein/c4ffein/guidelines/python-cli-utils/CLAUDE.md via `make update-claude-md`; edit only BELOW this line. -->

# Project-specific notes

_None yet._
