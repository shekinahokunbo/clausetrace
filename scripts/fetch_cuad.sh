#!/usr/bin/env bash
# Download CUAD v1 (CC BY 4.0, The Atticus Project) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/TheAtticusProject/cuad "$tmp/cuad"
unzip -o -q "$tmp/cuad/data.zip" -d "$tmp/extracted"
cp "$tmp/extracted/CUADv1.json" "$tmp/extracted/test.json" data/
cp "$tmp/cuad/category_descriptions.csv" data/
rm -rf "$tmp"
echo "CUAD ready in data/"
