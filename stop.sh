#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 · Copyright 2026 Hans-Dieter Zucht
# Beendet Voili Server und stellt den Systemton zurück. / Quits Voili Server and restores system audio.
D="$(cd "$(dirname "$0")" && pwd)"
A="$D/bin/vi-ausgabe"; [ -x "$A" ] || A=~/.local/bin/vi-ausgabe
curl -s -m 2 http://127.0.0.1:8770/ende >/dev/null
sleep 1
pkill -f "voili_server.py|voice_isolator_live.py" 2>/dev/null
if "$A" | grep -q -i -E "blackhole|loopback"; then
  # Notfall-Rückweg, falls das Programm hing: eingebaute Lautsprecher bevorzugen. / Fallback: prefer built-in speakers.
  Z=$("$A" liste | grep -i -E "lautsprecher|speakers|kopfhörer|headphones" | head -1)
  [ -n "$Z" ] || Z=$("$A" liste | grep -v -i -E "blackhole|loopback|teams|nomachine|multi|apple tv|airplay" | head -1)
  "$A" "$Z" >/dev/null
fi
echo "System output: $("$A")"
