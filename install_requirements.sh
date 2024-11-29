#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pip install --upgrade -r "${SCRIPT_DIR}/requirements.txt" -r "${SCRIPT_DIR}/requirements-dev.txt" -r "${SCRIPT_DIR}/docs/requirements.txt"
