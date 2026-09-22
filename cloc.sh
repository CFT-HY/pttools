#!/usr/bin/env sh
# Count lines of code with cloc.
#
# Usage:
#   ./cloc.sh [-l | --library] [CLOC_OPTIONS...]
#
# By default, the entire repository is counted.
# With -l or --library, only the PTtools library in ./pttools is counted.
# Any further arguments are passed to cloc.
#
# Only the files tracked by Git are counted, and the files matching the patterns in .clocignore are excluded.
# The patterns have the same syntax as in .gitignore.
# The file list is built with Git instead of the --exclude-list-file option of cloc,
# as that does not support relative paths in older versions of cloc.
# https://github.com/AlDanial/cloc/issues/784
set -eu

cd "$(dirname "$0")"

path="."
case "${1:-}" in
  -l|--library)
    path="pttools"
    shift
    ;;
  -h|--help)
    sed -n '2,/^set -eu$/{ /^set -eu$/!s/^# \{0,1\}//p; }' "$0"
    exit 0
    ;;
esac

ignored="$(mktemp)"
trap 'rm -f "$ignored"' EXIT

# The files in the index that match the ignore patterns
git ls-files --cached --ignored --exclude-from=.clocignore -- "$path" > "$ignored"
# The other files in the index are counted.
# grep -v prints all lines if the pattern file is empty.
git ls-files --cached -- "$path" \
  | grep --invert-match --line-regexp --fixed-strings --file="$ignored" \
  | cloc --list-file=- --by-file-by-lang "$@"
