# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hans-Dieter Zucht. Redistributions must keep this notice and the NOTICE file (author and sources).
"""Echtzeitfaktor des Inworld Voice Isolator auf diesem Mac messen.
Streaming (Echtzeitmodell) mit verschiedenen Stückgrößen, Offline-Modell auf kurzen Fenstern.
RTF = Rechenzeit / Audiodauer; < 1 heißt schneller als Echtzeit."""
import sys, time
import numpy as np, torch
from inworld.voice_isolation import VoiceIsolator

SR = 48_000
rng = np.random.default_rng(0)
t = np.arange(SR * 12) / SR
audio = (0.1 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)

for dev in sys.argv[1:] or ["mps", "cpu"]:
    for prec in (["fp32", "fp16"] if dev == "mps" else ["fp32"]):
        try:
            rt = VoiceIsolator.load("inworld/voice-isolator", mode="streaming", lookahead_frames=2, device=dev, precision=prec)
        except Exception as e:
            print(dev, prec, "Laden scheitert:", e); continue
        for piece_ms in (20, 50, 100):
            n = SR * piece_ms // 1000
            s = rt.stream(block_frames=max(4, piece_ms // 10))
            s.push(audio[: SR * 2])  # Aufwärmen
            if dev == "mps": torch.mps.synchronize()
            t0 = time.perf_counter(); worst = 0.0
            for i in range(SR * 2, SR * 12, n):
                a = time.perf_counter(); s.push(audio[i:i + n])
                if dev == "mps": torch.mps.synchronize()
                worst = max(worst, (time.perf_counter() - a) / (n / SR))
            el = time.perf_counter() - t0
            print(f"streaming {dev} {prec} Stück {piece_ms} ms: RTF {el / 10:.2f}, schlechtestes Stück {worst:.2f}", flush=True)
    off = VoiceIsolator.load("inworld/voice-isolator", mode="offline", device=dev)
    for win in (2.5, 4.0):
        seg = audio[: int(SR * win)]
        off.isolate(seg, SR)
        t0 = time.perf_counter()
        for _ in range(3): off.isolate(seg, SR)
        el = (time.perf_counter() - t0) / 3
        print(f"offline {dev} Fenster {win} s: {el:.2f} s Rechenzeit (RTF {el / win:.2f})", flush=True)
