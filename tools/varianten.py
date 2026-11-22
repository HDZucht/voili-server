"""Eine Datei durch alle Varianten schicken: Echtzeitmodell mit Vorschau 0/20/40 ms (100-ms-Stücke wie live),
Offline-Modell ganz (8-s-Blöcke) und in den Live-Fenstern (0,5 s · 1 s · 0,25 s). Schreibt WAVs (16 bit) und
druckt je Variante den Pegel gegenüber dem Original, gesamt und im SSB-Band 300–2700 Hz.
Aufruf: python -I varianten.py <eingang.wav> <ausgabeordner>"""
import os, sys
import numpy as np, soundfile as sf
from inworld.voice_isolation import VoiceIsolator

SR = 48_000
x, sr = sf.read(sys.argv[1], dtype="float32"); assert sr == SR
if x.ndim > 1: x = x.mean(1)
out = sys.argv[2]; os.makedirs(out, exist_ok=True)

def band_db(y, lo=300, hi=2700):
    Y = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR)
    return 10 * np.log10(Y[(f >= lo) & (f <= hi)].sum() + 1e-12)

def rms_db(y): return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)

ergebnis = {}
rt = VoiceIsolator.load("inworld/voice-isolator", mode="streaming", device="mps")
for la in rt.lookahead_choices:
    rt.lookahead_frames = la
    s = rt.stream(block_frames=10); teile = [s.push(x[i:i + 4800]) for i in range(0, len(x), 4800)]
    teile.append(s.flush()); y = np.concatenate(teile)[:len(x)]
    ergebnis[f"echtzeit_vorschau{la * 10}ms"] = y
off = VoiceIsolator.load("inworld/voice-isolator", mode="offline", device="mps")
ergebnis["offline_ganz"] = off.isolate(x, SR)
P, E, F, X = SR // 2, SR, SR // 4, SR // 100
pad = np.concatenate([np.zeros(P, np.float32), x, np.zeros(E + F, np.float32)])
y = np.zeros(len(x) + E, np.float32); schwanz = None
for p in range(0, len(x), E):
    w = off.isolate(pad[p:p + P + E + F], SR); teil = w[P:P + E].copy()
    if schwanz is not None:
        r = np.linspace(0, 1, X, dtype=np.float32); teil[:X] = schwanz * (1 - r) + teil[:X] * r
    schwanz = w[P + E:P + E + X]; y[p:p + E] = teil
ergebnis["offline_livefenster"] = y[:len(x)]

sf.write(os.path.join(out, "0_original.wav"), x, SR, subtype="PCM_16")
print(f"{'Variante':28s} {'Pegel':>8s} {'SSB-Band':>9s}")
for i, (k, y) in enumerate(ergebnis.items(), 1):
    sf.write(os.path.join(out, f"{i}_{k}.wav"), np.clip(y, -1, 1), SR, subtype="PCM_16")
    print(f"{k:28s} {rms_db(y) - rms_db(x):+7.1f} dB {band_db(y) - band_db(x):+7.1f} dB")
