#!/usr/bin/env bash
set -e

python infer.py --config paviau.yaml --checkpoint best.pth
