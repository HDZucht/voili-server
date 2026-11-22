# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hans-Dieter Zucht. Redistributions must keep this notice and the NOTICE file (author and sources).
"""Demo Ein/Aus — einen Voili-Mitschnitt so nachrechnen, wie er live klang, und den Rauschboden messen.

Ends:   HDZ 2026-11-22: „nimm mal folgendes am Radio auf, um zu zeigen, wie gut das Modell arbeitet … ich schalte
        mal den Filter ein und aus“. Voili gibt direkt auf den Lautsprecher aus; der Ausgang ist nicht abgreifbar,
        ohne den laufenden Server neu zu starten. Also: Rohton (BlackHole) + Zustandsprotokoll (/status alle 0,2 s)
        mitschneiden und den Ausgang hier nachrechnen.
Ways:   Wie der Arbeiter im Offline-Modus: Fenster 0,5 s Vergangenheit + 1 s Ausgabe + 0,25 s Zukunft, 10 ms
        Überblendung, Bandgrenze aus dem Protokoll, Verstärkung aus dem Protokoll. Mix wie aus_cb: Filter aus → 100 %
        Original, an → Mix-Wert; gleitet 0,02 je 10 ms. Schaltzeit im Rohton = Wanduhr − Verzögerung (verzoegerung_ms),
        denn was HDZ zur Zeit t hörte, kam um die Verzögerung früher herein.
Means:  Schreibt 0_original.wav, 1_isoliert.wav, 2_wie_gehoert.wav (16 bit, mono, 48 kHz) und druckt den Pegel im
        SSB-Band 300–2700 Hz: Rauschboden (10. Perzentil der 50-ms-Blöcke) und Sprachspitzen (95. Perzentil).
Expects: Ordner mit roh_blackhole.wav, zustand.log, start_epoch.txt; Python mit inworld-Paket (wie voili_server.py).
Fails:  Das Zustandsprotokoll startet wenige ms nach ffmpeg; Schaltpunkte sind auf ±0,2 s genau.
Aufruf: python -I tools/demo_ein_aus.py <aufnahmeordner>   (VI_GERAET=cpu, solange jemand live hört: GPU teilen = Aussetzer)
Tags:   voice-isolator, demo, messung
"""
import json, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bandgrenze
from inworld.voice_isolation import VoiceIsolator

SR = 48_000
P, E, F, X = SR // 2, SR, SR // 4, SR // 100
d = sys.argv[1]
x, sr = sf.read(os.path.join(d, "roh_blackhole.wav"), dtype="float32"); assert sr == SR
x = x.mean(1) if x.ndim > 1 else x
t0 = float(open(os.path.join(d, "start_epoch.txt")).read())
log = []
for zeile in open(os.path.join(d, "zustand.log")):
    ts, _, j = zeile.partition(" ")
    try: log.append((float(ts) - t0, json.loads(j)))
    except Exception: pass
z0 = log[0][1]

off = VoiceIsolator.load("inworld/voice-isolator", mode="offline", device=os.environ.get("VI_GERAET", "mps"))
bandgrenze.Bandgrenze(off.model).setze(z0["khz"])
xp = np.concatenate([np.zeros(P, np.float32), x, np.zeros(E + F, np.float32)])
y = np.zeros(len(x) + E, np.float32); schwanz = None
for p in range(0, len(x), E):
    fy = off.isolate(xp[p:p + P + E + F], SR)
    teil = fy[P:P + E].astype(np.float32).copy()
    if schwanz is not None:
        r = np.linspace(0, 1, X, dtype=np.float32); teil[:X] = schwanz * (1 - r) + teil[:X] * r
    schwanz = fy[P + E:P + E + X].astype(np.float32); y[p:p + E] = teil
y = y[:len(x)]

# Mix-Soll je 10-ms-Block (Index im Rohton), dann gleiten wie aus_cb
B = SR // 100; nb = len(x) // B + 1
soll = np.ones(nb, np.float32); verz = z0["verzoegerung_ms"] / 1000
for t, z in log:
    i = int(max(0, (t - verz)) * 100)
    soll[i:] = (z["mix"] / 100) if z["an"] else 1.0
ist = np.empty(nb, np.float32); m = soll[0]
for i in range(nb):
    m += max(-0.02, min(0.02, soll[i] - m)); ist[i] = m
mix = np.interp(np.arange(len(x)), np.arange(nb) * B, ist).astype(np.float32)
g = 10 ** (z0["gain_db"] / 20)
gehoert = np.clip((y * (1 - mix) + x * mix) * g, -1, 1)
sf.write(os.path.join(d, "0_original.wav"), np.clip(x * g, -1, 1), SR, subtype="PCM_16")
sf.write(os.path.join(d, "1_isoliert.wav"), np.clip(y * g, -1, 1), SR, subtype="PCM_16")
sf.write(os.path.join(d, "2_wie_gehoert.wav"), gehoert, SR, subtype="PCM_16")

def band_bloecke(s, lo=300, hi=2700, n=SR // 20):
    k = len(s) // n; S = np.abs(np.fft.rfft(s[:k * n].reshape(k, n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR); return 10 * np.log10(S[:, (f >= lo) & (f <= hi)].sum(1) / n + 1e-12)
a, b = band_bloecke(x), band_bloecke(y)
print(f"Dauer {len(x)/SR:.1f} s · Bandgrenze {z0['khz']} kHz · +{z0['gain_db']} dB")
print(f"Rauschboden (P10)  Original {np.percentile(a,10):6.1f} dB · isoliert {np.percentile(b,10):6.1f} dB · Δ {np.percentile(b,10)-np.percentile(a,10):+.1f} dB")
print(f"Sprachspitzen (P95) Original {np.percentile(a,95):6.1f} dB · isoliert {np.percentile(b,95):6.1f} dB · Δ {np.percentile(b,95)-np.percentile(a,95):+.1f} dB")
wechsel = [(round(max(0, t - verz), 1), z["an"]) for (t, z), (_, z1) in zip(log[1:], log) if z["an"] != z1["an"]]
print("Schaltpunkte im Rohton (s, an):", wechsel)
