#!/bin/bash
# Richtet Voili Server ein: venv mit PyTorch und Inworld Voice Isolator, Swift-Helfer nach ./bin.
# Sets up Voili Server: venv with PyTorch and Inworld Voice Isolator, Swift helpers into ./bin.
set -e
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
PY=$(command -v python3.12 || command -v python3)
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
mkdir -p bin
swiftc -O swift/ausgabe.swift -o bin/vi-ausgabe
swiftc -O swift/klang.swift -o bin/vi-klang
system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole" || echo "Hint: install BlackHole 2ch for system audio: brew install blackhole-2ch"
echo "Done. Start with ./start.sh (model weights, ~0.5 GB, download on first start)."
