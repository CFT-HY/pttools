#!/usr/bin/env sh
set -e

# Count lines of code in the project

cloc --by-file-by-lang --exclude-list-file=.clocignore .
