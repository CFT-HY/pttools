#!/bin/sh
set -eu

rm -rf \
  ./**/*.nbc \
  ./**/*.nbi \
  ./**/.coverage \
  ./**/.coverage.*.*.* \
  ./**/.pytest_cache \
  ./**/.test_durations \
  ./**/coverage.json \
  ./**/coverage.xml \
  ./**/logs \
  ./**/htmlcov \
  ./**/output.html \
  ./**/pylint.txt \
  ./**/test-results \
