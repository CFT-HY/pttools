#!/usr/bin/env sh
set -eu

rm -rf \
  ./**/*.nbc \
  ./**/*.nbi \
  ./*.egg-info \
  ./.coverage \
  ./.coverage.*.*.* \
  ./.pytest_cache \
  ./.test_durations \
  ./call_graphs \
  ./coverage.json \
  ./coverage.xml \
  ./dist \
  ./htmlcov \
  ./logs \
  ./pyrefly.txt \
  ./output.html \
  ./ruff.txt \
  ./test-results \
  ./tests/test_data/*.hdf5
