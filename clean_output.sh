#!/bin/sh

rm -r ./**/.pytest_cache
rm -r ./**/logs
rm -r ./**/htmlcov
rm -r ./**/test-results
rm ./**/.coverage
rm ./**/.coverage.*.*.*
rm ./**/coverage.xml
rm ./**/output.html
rm ./**/pylint.txt

rm -r ./.pytest_cache
rm -r ./logs
rm -r ./htmlcov
rm -r ./test-results
rm ./.coverage
rm ./.coverage.*.*.*
rm ./coverage.xml
rm ./output.html
rm pylint.txt
