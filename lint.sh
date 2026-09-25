#!/usr/bin/env bash
# Run the linters and type checkers of PTtools.
#
# All checks are run even if some of them fail.
# Exit code:
# - 0 if all checks pass
# - the exit code of the failed check if exactly one check fails
# - 100 if multiple checks fail
# - 64 for invalid command-line arguments
set -u

MULTIPLE_FAILED=100

usage() {
  cat <<EOF
Usage: $0 [-f|--fast] [-h|--help]

Run pyrefly check, pyrefly coverage check, ruff check and the documentation lint (pttools.docs.lint).

Options:
  -f, --fast  Skip the documentation lint, which is significantly slower than the other checks.
  -h, --help  Show this help.

Exit code: 0 if all checks pass, the exit code of the failed check if exactly one check fails,
and ${MULTIPLE_FAILED} if multiple checks fail.
EOF
}

FAST=false
for arg in "$@"; do
  case "${arg}" in
    -f|--fast) FAST=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; usage >&2; exit 64 ;;
  esac
done

cd "$(dirname "$0")" || exit 1

FAILED=()
LAST_CODE=0

run() {
  echo "=== $* ==="
  "$@"
  local code=$?
  if [ "${code}" -ne 0 ]; then
    echo "=== FAILED with exit code ${code}: $* ===" >&2
    FAILED+=("$* (exit code ${code})")
    LAST_CODE=${code}
  fi
  echo
}

run uv run pyrefly check
run uv run pyrefly coverage check
run uv run ruff check
if [ "${FAST}" = false ]; then
  run uv run python -m pttools.docs.lint
fi

if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "All checks passed."
  exit 0
fi

echo "Failed checks:" >&2
for failed in "${FAILED[@]}"; do
  echo "- ${failed}" >&2
done
if [ "${#FAILED[@]}" -eq 1 ]; then
  exit "${LAST_CODE}"
fi
exit "${MULTIPLE_FAILED}"
