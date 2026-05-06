#!/usr/bin/env bash
# Local CI — runs before `git push` via .githooks/pre-push.
# Bypass: `git push --no-verify`. Manual run: `scripts/ci.sh`.
set -euo pipefail

cd "$(dirname "$0")/.."

# Auto-activate .venv if not already inside one (maturin + pytest need it).
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

step "ruff check"
uvx ruff check .

step "ruff format --check"
uvx ruff format --check .

step "cargo fmt --check"
cargo fmt --check

step "cargo clippy (-D warnings)"
cargo clippy --all-targets --all-features -- -D warnings

step "cargo test --lib"
cargo test --lib --quiet

# Rebuild python extension if rust files differ from upstream OR have
# uncommitted changes in the working tree.
RUST_DIRTY=0
if [[ -n "$(git status --porcelain -- src/ Cargo.toml Cargo.lock 2>/dev/null)" ]]; then
    RUST_DIRTY=1
elif git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git diff --quiet '@{u}...HEAD' -- src/ Cargo.toml Cargo.lock || RUST_DIRTY=1
else
    # No upstream tracking — be safe, rebuild.
    RUST_DIRTY=1
fi

if [[ $RUST_DIRTY -eq 1 ]]; then
    step "maturin develop (rust changes detected)"
    maturin develop
else
    step "skip maturin (no rust changes vs @{u} or working tree)"
fi

step "pytest"
pytest -q tests/

printf '\n\033[32mall checks passed\033[0m\n'
