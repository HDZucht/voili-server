#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 · Copyright 2026 Hans-Dieter Zucht
# Startet Voili Server und öffnet das Bedienfeld. / Starts Voili Server and opens the control panel.
# Optional: Ausgabegerät als Argument (Teilstring). / Optional: output device as argument (substring).
D="$(cd "$(dirname "$0")" && pwd)"
PY="$D/.venv/bin/python"; [ -x "$PY" ] || PY=~/.venvs/voice-isolator/bin/python
mkdir -p ~/Library/Logs
if curl -s -m 1 http://127.0.0.1:8770/status >/dev/null; then open http://127.0.0.1:8770; exit 0; fi
nohup "$PY" -I "$D/voili_server.py" "$@" >> ~/Library/Logs/voice-isolator-live.log 2>&1 &
for i in $(seq 1 90); do curl -s -m 1 http://127.0.0.1:8770/status >/dev/null && break; sleep 0.5; done
open http://127.0.0.1:8770
echo "Voili Server: http://127.0.0.1:8770"
