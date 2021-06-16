#!/usr/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CALL_GRAPH_DIR="${SCRIPT_DIR}/call_graphs"
mkdir -p "$CALL_GRAPH_DIR"

NAME="$1"
shift

# -a = --annotated
# -c = --colored
# -e = --nested-groups
pyan3 "$@" -a -c -e --dot > "${CALL_GRAPH_DIR}/call_graph_${NAME}.dot"
pyan3 "$@" -a -c -e --html > "${CALL_GRAPH_DIR}/call_graph_${NAME}.html"
pyan3 "$@" -a -c -e --svg > "${CALL_GRAPH_DIR}/call_graph_${NAME}.svg"
