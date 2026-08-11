#!/usr/bin/env sh
set -eu

# Count lines of code in the project

# https://github.com/AlDanial/cloc/issues/784
# cloc --by-file-by-lang --exclude-list-file=.clocignore .
# https://stackoverflow.com/a/26679008
# cloc --by-file-by-lang --exclude-dir=$(tr '\n' ',' < .clocignore) .
# This uses .gitignore
cloc --by-file-by-lang --vcs=git .
