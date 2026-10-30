#!/usr/bin/env bash
# Compile the project into ./out (requires JDK 17+).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf out && mkdir -p out
javac -d out $(find src -name '*.java')
echo "Built into ./out"
