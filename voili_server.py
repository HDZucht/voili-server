# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hans-Dieter Zucht. Redistributions must keep this notice and the NOTICE file (author and sources).
"""Voili Server (voice-isolator-live) — Inworld Voice Isolator als NR-Equalizer im Systemton.

Ends:   HDZ 2026-11-22: „es würde mir in Stufe null reichen, wenn ich es akustisch selber bewerten kann,
        und aktivieren im Systempfad, dann würde ich irgendein KiwiSDR anhören und diesen eigenen NR-Equalizer
        dazuschalten … mit zwei Stufen 100 ms Verzögerung und meinetwegen 1 s Verzögerung“.
        HDZ 2026-11-22 (v0.1.1): „einen Mix-Schieberegler von 0 bis 100 %, dass man den Originalton dazumischen
        könnte“ · „die Links zur Quelle als Literaturhinweis rein mit GitHub-Link, dann ist das ja schon ein
        Miniprodukt“ · „Eingang und Ausgang Audio-Interfaces wählbar mit Hinweis install BlackHole“.
        HDZ 2026-11-22 (v0.1.2): „ja, Bandbreite auf voll, 12 kHz, 8 kHz, 6 kHz einbauen“ (Rechenlast ∝ Bänder,
        siehe bandgrenze.py).
        HDZ 2026-11-22 (v0.1.3): „Musik-Switcher, zwei Regler Sprachannahme (5 %) und Musikannahme 15 % z. B. als
        Default, beide Regler wählbar“ → klang.swift (Apples SoundAnalysis) meldet Musik/Sprache, Hysterese,
        der Mix gleitet in 0,5 s auf den Wert der erkannten Art.
        HDZ 2026-11-22 (v0.1.4): „übersetze Voili Server in Englisch, wenn Sprache nicht Deutsch ist … als Titel
        Voili Server … stilisiere den Volleyball … damit neues Bild rechtefrei wird, und mach da ein Logo rein“.
        Sprache folgt der Browsersprache (de → Deutsch, sonst Englisch); Fehler kommen als Codes (E_…).
        HDZ 2026-11-22 (v0.1.7): „nenne die Modelle einfach Schnell und Quality“ → Knöpfe Schnell/Qualität bzw.
        Fast/Quality; intern bleiben die Modi „echtzeit“ und „offline“.
        HDZ 2026-11-22 (v0.1.6): „ja, mache Sprachumschalter“ → DE | EN oben rechts, auch ?lang=de|en, Wahl bleibt.
Ways:   Eingang (BlackHole 2ch für Systemton, oder jedes andere Eingabegerät) → dieses Programm → Ausgabegerät.
        Ist der Eingang ein Loopback-Gerät (BlackHole, Loopback), wird die Standard-Ausgabe des Systems dorthin
        umgestellt und beim Wechsel oder Ende zurückgestellt.
        Stufe „Echtzeit“: kausales Modell (58 M), Stücke à 100 ms, Vorschau 20 ms.
        Stufe „Offline“: zweiseitiges Modell (64 M) auf gleitenden Fenstern (0,5 s Vergangenheit, 1 s Ausgabe,
        0,25 s Zukunft), Übergänge mit 10 ms Überblendung.
        Original und Isoliert laufen zeitgleich ausgerichtet nebeneinander; Schalter und Mix wirken beim
        Abspielen, also sofort und ohne Zeitsprung.
Means:  PyTorch auf MPS (gemessen 22.11. auf einem MacBook Air M4: Echtzeit RTF 0,53 bei 100-ms-Stücken, Spitzen bis 2,9;
        Offline RTF 0,32; CPU RTF 7,5 → unbrauchbar). sounddevice für Ein-/Ausgabe, http.server für das Bedienfeld.
Provides: Bedienfeld http://127.0.0.1:8770 (An/Aus, Stufe, Mix, Puffer, Lautstärke, Geräte, Zustand, Quellen).
Expects: Geräte bei 48 kHz (sonst wandelt Core Audio, falls es kann); Hilfsprogramm ~/.local/bin/vi-ausgabe.
Ensures: Beim Beenden (Knopf, SIGTERM, Strg-C) wird die vorige Standard-Ausgabe wiederhergestellt;
        Einstellungen bleiben in ~/Library/Application Support/voice-isolator-live/einstellungen.json.
Fails:  Unterläufe → kurze Stille, gezählt; Gerätefehler und Ausnahmen → Bedienfeld „Fehler“ und
        ~/Library/Logs/voice-isolator-live.log.
"""
import atexit, collections, json, os, signal, subprocess, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import sounddevice as sd
import torch
from inworld.voice_isolation import VoiceIsolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bandgrenze  # noqa: E402

VERSION = "0.1.8"
SR = 48_000
PORT = int(os.environ.get("VI_PORT", 8770))
HIER = os.path.dirname(os.path.abspath(__file__))


def hilfsprogramm(name):
    # Why: im Repo baut install.sh die Swift-Helfer nach ./bin; im Vault liegen sie in ~/.local/bin.
    for d in (os.path.join(HIER, "bin"), os.path.expanduser("~/.local/bin")):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return os.path.join(HIER, "bin", name)


AUSGABE = hilfsprogramm("vi-ausgabe")
KLANG = hilfsprogramm("vi-klang")
LOG = os.path.expanduser("~/Library/Logs/voice-isolator-live.log")
EINST = os.environ.get("VI_EINST") or os.path.expanduser("~/Library/Application Support/voice-isolator-live/einstellungen.json")
LOOPBACK = ("BlackHole", "Loopback", "Soundflower")
STUECK = SR // 10                 # Echtzeit: 100 ms je push (gemessen: kleiner wird RTF > 0,6)
P, E, F, X = SR // 2, SR, SR // 4, SR // 100   # Offline: Vergangenheit, Ausgabe, Zukunft, Überblendung


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def ausgabe(*args):
    r = subprocess.run([AUSGABE, *args], capture_output=True, text=True)
    if r.returncode:
        log(f"vi-ausgabe {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def ist_loopback(name):
    return any(k.lower() in name.lower() for k in LOOPBACK)


class Zustand:
    FELDER = ("an", "modus", "puffer_ms", "gain_db", "mix", "eingang", "ausgang", "khz", "mix_musik", "musik_an")

    def __init__(self):
        self.an = True
        self.modus = "offline"            # HDZ 22.11.: am SSB-Funk trägt nur das Offline-Modell
        self.puffer_ms = {"echtzeit": 300, "offline": 1300}
        self.gain_db = 0.0
        self.mix = 5                      # Originalanteil bei Sprache in %, 0 = nur isoliert (HDZ: 5 %)
        self.mix_musik = 15               # Originalanteil bei erkannter Musik (HDZ: 15 %)
        self.musik_an = True              # Musik-Umschalter
        self.musik_p = self.sprache_p = 0.0
        self.musik_aktiv = False
        self.m_ist = 0.05
        self.khz = 0                      # Bandgrenze, 0 = voll
        self.baender = 46
        self.eingang = "BlackHole 2ch"
        self.ausgang = ""
        self.rtf = 0.0
        self.unterlaeufe = 0
        self.verworfen = 0
        self.fehler = ""
        self.neu = threading.Event()
        self.ende = threading.Event()

    def lade(self):
        try:
            with open(EINST) as f:
                for k, v in json.load(f).items():
                    if k in self.FELDER:
                        setattr(self, k, v)
        except FileNotFoundError:
            pass
        except Exception as e:
            log(f"Einstellungen lesen: {e}")

    def sichere(self):
        try:
            os.makedirs(os.path.dirname(EINST), exist_ok=True)
            with open(EINST, "w") as f:
                json.dump({k: getattr(self, k) for k in self.FELDER}, f, ensure_ascii=False, indent=1)
        except Exception as e:
            log(f"Einstellungen schreiben: {e}")


Z = Zustand()


class Ring:
    """Zwei gleich lange Spuren (isoliert, original), zeitgleich geschrieben und gelesen."""

    def __init__(self):
        self.lock = threading.Lock()
        self.nass = collections.deque()
        self.trocken = collections.deque()
        self.n = 0
        self.fuellen = True

    def schreibe(self, nass, trocken):
        with self.lock:
            self.nass.append(nass); self.trocken.append(trocken); self.n += len(nass)

    def leere(self):
        with self.lock:
            self.nass.clear(); self.trocken.clear(); self.n = 0; self.fuellen = True

    def lies(self, k):
        with self.lock:
            ziel = Z.puffer_ms[Z.modus] * SR // 1000
            if self.fuellen:
                if self.n < ziel:
                    return None
                self.fuellen = False
            if self.n > ziel + SR // 2:          # Uhrendrift Eingang ↔ Ausgang: Überschuss abwerfen
                weg = self.n - ziel
                Z.verworfen += weg
                self._nimm(weg)
            if self.n < k:
                Z.unterlaeufe += 1
                self.fuellen = True
                return None
            return self._nimm(k)

    def _nimm(self, k):
        a, b, got = [], [], 0
        while got < k:
            n, t = self.nass[0], self.trocken[0]
            take = min(k - got, len(n))
            a.append(n[:take]); b.append(t[:take])
            if take == len(n):
                self.nass.popleft(); self.trocken.popleft()
            else:
                self.nass[0], self.trocken[0] = n[take:], t[take:]
            got += take
        self.n -= k
        return np.concatenate(a), np.concatenate(b)


RING = Ring()
EIN = collections.deque()
EIN_LOCK = threading.Condition()


def ein_cb(indata, frames, t, status):
    with EIN_LOCK:
        EIN.append(indata.mean(axis=1).astype(np.float32))
        EIN_LOCK.notify()


def mix_soll():
    if not Z.an:
        return 1.0
    return (Z.mix_musik if Z.musik_an and Z.musik_aktiv else Z.mix) / 100


def aus_cb(outdata, frames, t, status):
    r = RING.lies(frames)
    if r is None:
        outdata.fill(0); return
    nass, trocken = r
    # Why: ein Sprung im Mix knackt; der Wert gleitet in 0,5 s (0,02 je 10-ms-Block) auf das Soll.
    soll, ist = mix_soll(), Z.m_ist
    neu = ist + max(-0.02, min(0.02, soll - ist)) if Z.an else soll
    m = np.linspace(ist, neu, frames, dtype=np.float32); Z.m_ist = neu
    x = np.clip((nass * (1 - m) + trocken * m) * 10 ** (Z.gain_db / 20), -1, 1)
    outdata[:] = x[:, None]


class Klang:
    """Musik/Sprache-Erkennung im Nebenprozess vi-klang (SoundAnalysis); Hysterese wie ein Schmitt-Trigger."""
    EIN, AUS, N_EIN, N_AUS = 0.5, 0.3, 4, 6    # ≈ 2 s über 0,5 schaltet ein, ≈ 3 s unter 0,3 schaltet aus

    def __init__(self):
        self.q = collections.deque(maxlen=50); self.p = None; self.zaehler = 0; self.letzt = 0.0
        threading.Thread(target=self._schreiber, daemon=True).start()

    def _starte(self):
        if time.time() - self.letzt < 10:
            return
        self.letzt = time.time()
        try:
            self.p = subprocess.Popen([KLANG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(LOG, "a"))
            threading.Thread(target=self._leser, args=(self.p,), daemon=True).start()
            log("vi-klang gestartet")
        except Exception as e:
            self.p = None; log(f"vi-klang: {e}")

    def fuettere(self, x):
        self.q.append(x)

    def _schreiber(self):
        while not Z.ende.is_set():
            if not self.q:
                time.sleep(0.05); continue
            x = self.q.popleft()
            if self.p is None or self.p.poll() is not None:
                self._starte()
                if self.p is None: continue
            try:
                self.p.stdin.write(x.tobytes()); self.p.stdin.flush()
            except Exception as e:
                log(f"vi-klang schreiben: {e}"); self.p = None
        if self.p: self.p.kill()

    def _leser(self, p):
        for zeile in p.stdout:
            try:
                d = json.loads(zeile)
            except ValueError:
                continue
            Z.musik_p, Z.sprache_p = d["musik"], d["sprache"]
            if Z.musik_aktiv:
                self.zaehler = self.zaehler + 1 if Z.musik_p < self.AUS else 0
                if self.zaehler >= self.N_AUS: Z.musik_aktiv, self.zaehler = False, 0
            else:
                self.zaehler = self.zaehler + 1 if Z.musik_p > self.EIN else 0
                if self.zaehler >= self.N_EIN: Z.musik_aktiv, self.zaehler = True, 0


KLANGERKENNUNG = None


class Verlauf:
    """Eingangsverlauf mit absolutem Index, damit Original und Isoliert zeitgleich bleiben."""

    def __init__(self):
        self.buf = np.zeros(0, np.float32); self.basis = 0

    def dazu(self, x):
        self.buf = np.concatenate([self.buf, x])
        ueber = len(self.buf) - 10 * SR
        if ueber > 0:
            self.buf = self.buf[ueber:]; self.basis += ueber

    @property
    def ende(self):
        return self.basis + len(self.buf)

    def hole(self, start, n):
        out = np.zeros(n, np.float32)
        a, b = max(start, self.basis), min(start + n, self.ende)
        if b > a:
            out[a - start:b - start] = self.buf[a - self.basis:b - self.basis]
        return out


def arbeiter(rt, off):
    grenzen = (bandgrenze.Bandgrenze(rt.model), bandgrenze.Bandgrenze(off.model))
    while not Z.ende.is_set():
        Z.neu.clear(); RING.leere()
        try:
            Z.baender = [g.setze(Z.khz) for g in grenzen][0]
        except Exception as e:
            Z.fehler = f"E_BAND|{e}"; log("Bandgrenze: " + traceback.format_exc())
        with EIN_LOCK:
            EIN.clear()
        v = Verlauf()
        modus = Z.modus
        rtf = collections.deque(maxlen=20)
        if modus == "echtzeit":
            s = rt.stream(block_frames=10)
            erzeugt = gesammelt = 0
        else:
            p = 0
            schwanz = None
        while not (Z.neu.is_set() or Z.ende.is_set()):
            with EIN_LOCK:
                while not EIN and not Z.neu.is_set() and not Z.ende.is_set():
                    EIN_LOCK.wait(0.2)
                stuecke = list(EIN); EIN.clear()
            for x in stuecke:
                v.dazu(x)
            if KLANGERKENNUNG is not None and stuecke:
                KLANGERKENNUNG.fuettere(np.concatenate(stuecke))
            try:
                if modus == "echtzeit":
                    while v.ende - gesammelt >= STUECK:
                        x = v.hole(gesammelt, STUECK); gesammelt += STUECK
                        t0 = time.perf_counter()
                        y = s.push(x)
                        torch.mps.synchronize()
                        rtf.append((time.perf_counter() - t0) / (STUECK / SR))
                        if len(y):
                            RING.schreibe(y.astype(np.float32), v.hole(erzeugt, len(y))); erzeugt += len(y)
                else:
                    while v.ende >= p + E + F:
                        fenster = v.hole(p - P, P + E + F)
                        t0 = time.perf_counter()
                        y = off.isolate(fenster, SR)
                        rtf.append((time.perf_counter() - t0) / (E / SR))
                        teil = y[P:P + E].astype(np.float32).copy()
                        if schwanz is not None:
                            r = np.linspace(0, 1, X, dtype=np.float32)
                            teil[:X] = schwanz * (1 - r) + teil[:X] * r
                        schwanz = y[P + E:P + E + X].astype(np.float32)
                        RING.schreibe(teil, v.hole(p, E)); p += E
                Z.rtf = float(np.mean(rtf)) if rtf else 0.0
            except Exception as e:
                Z.fehler = f"E_RUN|{type(e).__name__}: {e}"
                log("Arbeiter: " + traceback.format_exc())
                time.sleep(0.5)


class Audio:
    """Ein- und Ausgabestrom; Gerätewechsel zur Laufzeit; Systemton-Umleitung bei Loopback-Eingang."""

    def __init__(self, vorher):
        self.vorher = vorher              # Standard-Ausgabe beim Start, Ziel der Rückstellung
        self.ein = self.aus = None
        self.lock = threading.Lock()

    @staticmethod
    def geraete():
        ins, outs = [], []
        for d in sd.query_devices():
            if d["max_input_channels"] > 0: ins.append(d["name"])
            if d["max_output_channels"] > 0: outs.append(d["name"])
        return ins, outs

    def zurueck(self):
        # Startete das Programm schon auf BlackHole (Rest eines Absturzes), geht der Ton auf den gewählten Ausgang.
        ziel = self.vorher if self.vorher and not ist_loopback(self.vorher) else Z.ausgang
        if ziel and not ist_loopback(ziel) and ist_loopback(ausgabe()):
            ausgabe(ziel); log(f"Systemton zurück auf {ziel!r}")

    def starte(self, eingang, ausgang):
        with self.lock:
            for s in (self.ein, self.aus):
                if s is not None:
                    try: s.stop(); s.close()
                    except Exception as e: log(f"Strom schließen: {e}")
            self.ein = self.aus = None
            ins, outs = self.geraete()
            if eingang not in ins:
                Z.fehler = f"E_IN|{eingang}"; return
            if not ausgang or ausgang not in outs or ausgang == eingang:
                ausgang = self.vorher if self.vorher in outs and not ist_loopback(self.vorher) else \
                    next((o for o in outs if not ist_loopback(o)), "")
            if ausgang == eingang or ist_loopback(ausgang):
                Z.fehler = "E_LOOP"; return
            try:
                ci = min(2, sd.query_devices(eingang)["max_input_channels"])
                co = min(2, sd.query_devices(ausgang)["max_output_channels"])
                self.ein = sd.InputStream(device=eingang, channels=ci, samplerate=SR, blocksize=480, dtype="float32", callback=ein_cb)
                self.aus = sd.OutputStream(device=ausgang, channels=co, samplerate=SR, blocksize=480, dtype="float32", callback=aus_cb)
                self.ein.start(); self.aus.start()
            except Exception as e:
                Z.fehler = f"E_DEV|{e}"; log("Audio: " + traceback.format_exc()); return
            Z.eingang, Z.ausgang, Z.fehler = eingang, ausgang, ""
            if ist_loopback(eingang):
                ausgabe(eingang); log(f"Systemton → {eingang}")
            else:
                self.zurueck()
            Z.neu.set()
            with EIN_LOCK: EIN_LOCK.notify()
            log(f"Audio {eingang!r} → {ausgang!r}")

    def stoppe(self):
        with self.lock:
            for s in (self.ein, self.aus):
                if s is not None:
                    try: s.stop(); s.close()
                    except Exception: pass
        self.zurueck()


LOGO = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voili-logo.svg"), encoding="utf-8").read()

QUELLEN = [
    ("Inworld Voice Isolator v1 (GitHub, Apache-2.0)", "https://github.com/inworld-ai/voice-isolator"),
    ("Hugging Face: inworld/voice-isolator (Apache-2.0)", "https://huggingface.co/inworld/voice-isolator"),
    ("Inworld: Introducing Voice Isolator (2026-11-18)", "https://inworld.ai/blog/introducing-voice-isolator"),
    ("Inworld Open Source", "https://inworld.ai/open-source#voice-isolator"),
    ("BlackHole (GitHub, GPL-3.0)", "https://github.com/ExistentialAudio/BlackHole"),
]

SEITE = """<!doctype html><meta charset=utf-8><title>Voili Server</title>
<link rel=icon href="/logo.svg" type="image/svg+xml">
<style>body{font:15px -apple-system,sans-serif;max-width:580px;margin:20px auto;color:#222}
header{display:flex;align-items:center;gap:12px;margin-bottom:8px}header img{width:84px}
header h1{font-size:26px;margin:0}header .sub{font-size:13px;color:#555}
.sprache{margin-left:auto;font-size:14px}.sprache a{color:#1f6f5c;text-decoration:none;padding:2px 4px}.sprache a.ein{font-weight:bold;text-decoration:underline}
button{font-size:15px;padding:8px 14px;margin:3px;border-radius:7px;border:1px solid #888;background:#f4f4f4}
button.ein{background:#1f6f5c;color:#fff;border-color:#1f6f5c}#an{font-size:20px;padding:14px 30px}
table{border-collapse:collapse;margin-top:10px}td{padding:2px 10px 2px 0;vertical-align:top}
input[type=range]{width:220px}select{font-size:14px;max-width:300px}
.hinweis{background:#fff6d8;border:1px solid #e3c96b;padding:8px 10px;border-radius:6px;font-size:14px}
.klein{font-size:13px;color:#555}h4{margin:22px 0 6px}</style>
<header><img src="/logo.svg" alt="Voili"><div><h1>Voili Server</h1>
<div class=sub><span data-t=untertitel></span> · v__VERSION__</div></div>
<div class=sprache><a id=l_de href="?lang=de">DE</a>|<a id=l_en href="?lang=en">EN</a></div></header>
<button id=an onclick="s('an',Z.an?0:1)">…</button><br>
<button id=offline onclick="s('modus','offline')" data-t=b_offline></button>
<button id=echtzeit onclick="s('modus','echtzeit')" data-t=b_echtzeit></button><br>
<span data-t=bandbreite></span> <button id=k0 onclick="s('khz',0)" data-t=voll></button><button id=k12 onclick="s('khz',12)">12 kHz</button><button id=k8 onclick="s('khz',8)">8 kHz</button><button id=k6 onclick="s('khz',6)">6 kHz</button>
<table>
<tr><td data-t=mix_sprache></td><td><input type=range id=mix min=0 max=100 step=5 onchange="s('mix',this.value)"> <span id=mv></span> %</td></tr>
<tr><td data-t=mix_musik></td><td><input type=range id=mixm min=0 max=100 step=5 onchange="s('mix_musik',this.value)"> <span id=mmv></span> % <button id=mu onclick="s('musik_an',Z.musik_an?0:1)">…</button></td></tr>
<tr><td data-t=puffer></td><td><input type=range id=puffer min=100 max=3000 step=50 onchange="s('puffer',this.value)"> <span id=pv></span> ms</td></tr>
<tr><td data-t=lautstaerke></td><td><input type=range id=gain min=-24 max=12 step=1 onchange="s('gain',this.value)"> <span id=gv></span> dB</td></tr>
<tr><td data-t=eingang></td><td><select id=eingang onchange="s('eingang',encodeURIComponent(this.value))"></select></td></tr>
<tr><td data-t=ausgang></td><td><select id=ausgang onchange="s('ausgang',encodeURIComponent(this.value))"></select></td></tr>
</table>
<p id=bh class=hinweis style=display:none data-th=blackhole></p>
<p class=klein data-t=erklaerung></p>
<table id=t></table>
<p><button onclick="if(confirm(L.beenden_frage))fetch('/ende')" data-t=beenden></button></p>
<h4 data-t=quellen></h4><ul class=klein>__QUELLEN__</ul>
<p class=klein data-t=fuss></p>
<script>
const TX={de:{untertitel:'Stimme isolieren im Systemton',b_offline:'Qualität (≈ 2,5 s, für Funk)',b_echtzeit:'Schnell (≈ 0,5 s)',
bandbreite:'Bandbreite',voll:'voll',mix_sprache:'Mix bei Sprache',mix_musik:'Mix bei Musik',puffer:'Puffer',lautstaerke:'Lautstärke',
eingang:'Eingang',ausgang:'Ausgang',an:'Isolator AN',aus:'Isolator AUS (Original)',um_an:'Umschalter AN',um_aus:'Umschalter AUS',
blackhole:'Für den Systemton (Browser, KiwiSDR, YouTube) braucht es das virtuelle Gerät <b>BlackHole 2ch</b>: <code>brew install blackhole-2ch</code> oder Installer von <a href="https://github.com/ExistentialAudio/BlackHole" target=_blank>github.com/ExistentialAudio/BlackHole</a>. Danach hier als Eingang wählen; der Systemton wird dann automatisch dorthin umgeleitet und beim Beenden zurückgestellt.',
erklaerung:'Musik-Umschalter: Apples Klangerkennung (SoundAnalysis) meldet Musik; nach ≈ 2 s gilt „Mix bei Musik“, nach ≈ 3 s ohne Musik wieder „Mix bei Sprache“, der Übergang gleitet. Bandbreite: das Modell rechnet nur die Bänder unter der Grenze (weniger Rechenlast), darüber kommt aus dem Isolator nichts. Mix 0 % = nur Stimme, 100 % = Original. Ist BlackHole der Eingang, wird die Systemausgabe dorthin umgestellt; die Lautstärketasten wirken dann auf BlackHole, die Lautstärke regelt hier der Regler.',
erkannt:'Erkannt',musik:'Musik',sprache:'Sprache',verzoegerung:'Verzögerung ca.',baender:'Bänder gerechnet',von:'von',rtf:'Rechenlast (RTF)',zu_hoch:' ⚠️ zu hoch',
unterlaeufe:'Unterläufe',verworfen:'verworfen (Drift)',systemton:'Systemton',fehler:'Fehler',beenden:'Beenden',beenden_frage:'Beenden und Systemton zurückstellen?',quellen:'Quellen',
fuss:'Modelle: Inworld Voice Isolator v1, „Schnell“ = Echtzeitmodell (58 M, kausal), „Qualität“ = Offline-Modell (64 M), Apache-2.0. Werkzeug: Hans-Dieter Zucht mit Aletheia (Claude Opus 5.5), 2026. Logo: eigene Zeichnung.',
E_IN:'Eingang „{0}“ fehlt – BlackHole installieren oder anderen Eingang wählen',E_LOOP:'Ausgang darf kein Loopback-Gerät sein (Rückkopplung)',E_DEV:'Gerät: {0}',E_BAND:'Bandgrenze: {0}',E_RUN:'Verarbeitung: {0}'},
en:{untertitel:'Voice isolation for system audio',b_offline:'Quality (≈ 2.5 s, for radio)',b_echtzeit:'Fast (≈ 0.5 s)',
bandbreite:'Bandwidth',voll:'full',mix_sprache:'Mix for speech',mix_musik:'Mix for music',puffer:'Buffer',lautstaerke:'Volume',
eingang:'Input',ausgang:'Output',an:'Isolator ON',aus:'Isolator OFF (original)',um_an:'Switch ON',um_aus:'Switch OFF',
blackhole:'System audio (browser, KiwiSDR, YouTube) needs the virtual device <b>BlackHole 2ch</b>: <code>brew install blackhole-2ch</code> or the installer from <a href="https://github.com/ExistentialAudio/BlackHole" target=_blank>github.com/ExistentialAudio/BlackHole</a>. Then choose it as input here; system audio is routed to it automatically and restored when you quit.',
erklaerung:'Music switch: Apple sound classification (SoundAnalysis) detects music; after ≈ 2 s “Mix for music” applies, after ≈ 3 s without music “Mix for speech” again, with a smooth transition. Bandwidth: the model computes only the bands below the limit (less load); above it the isolator outputs nothing. Mix 0 % = voice only, 100 % = original. With BlackHole as input the system output is switched to it; the volume keys then act on BlackHole, use the slider here for volume.',
erkannt:'Detected',musik:'music',sprache:'speech',verzoegerung:'Delay approx.',baender:'Bands computed',von:'of',rtf:'Load (RTF)',zu_hoch:' ⚠️ too high',
unterlaeufe:'Underruns',verworfen:'dropped (drift)',systemton:'System output',fehler:'Error',beenden:'Quit',beenden_frage:'Quit and restore system audio?',quellen:'Sources',
fuss:'Models: Inworld Voice Isolator v1, “Fast” = real-time model (58 M, causal), “Quality” = offline model (64 M), Apache-2.0. Tool: Hans-Dieter Zucht with Aletheia (Claude Opus 5.5), 2026. Logo: original drawing.',
E_IN:'Input “{0}” missing – install BlackHole or choose another input',E_LOOP:'Output must not be a loopback device (feedback)',E_DEV:'Device: {0}',E_BAND:'Band limit: {0}',E_RUN:'Processing: {0}'}};
// Sprache: ?lang= (merkt sich die Wahl) > gemerkte Wahl > Browsersprache. HDZ 22.11.: „ja, mache Sprachumschalter“.
const qs=new URLSearchParams(location.search).get('lang');if(qs==='de'||qs==='en')localStorage.setItem('voili_lang',qs);
const SPR=localStorage.getItem('voili_lang')||((navigator.language||'en').toLowerCase().startsWith('de')?'de':'en');
const L=TX[SPR];document.documentElement.lang=SPR;document.getElementById('l_'+SPR).className='ein';
document.querySelectorAll('[data-t]').forEach(e=>e.textContent=L[e.dataset.t]);document.querySelectorAll('[data-th]').forEach(e=>e.innerHTML=L[e.dataset.th]);
function fx(f){if(!f)return '–';const [k,...a]=f.split('|');return L[k]?L[k].replace('{0}',a.join('|')):f}
const zahl=(v,n)=>L===TX.de?v.toFixed(n).replace('.',','):v.toFixed(n);
let Z={},G=null;
function s(k,v){fetch('/set?'+k+'='+v).then(u)}
function fuelle(sel,liste,wert){if(sel.dataset.l===liste.join('|')&&document.activeElement===sel)return;
sel.innerHTML=liste.map(n=>'<option'+(n===wert?' selected':'')+'>'+n+'</option>').join('');sel.dataset.l=liste.join('|')}
function g(){fetch('/geraete').then(r=>r.json()).then(x=>{G=x;bh.style.display=x.blackhole?'none':'block'})}
function u(){fetch('/status').then(r=>r.json()).then(z=>{Z=z;
an.textContent=z.an?L.an:L.aus;an.className=z.an?'ein':'';
echtzeit.className=z.modus=='echtzeit'?'ein':'';offline.className=z.modus=='offline'?'ein':'';
for(const k of [0,12,8,6])document.getElementById('k'+k).className=z.khz==k?'ein':'';
mu.textContent=z.musik_an?L.um_an:L.um_aus;mu.className=z.musik_an?'ein':'';
for(const [el,v,w] of [[mix,mv,z.mix],[mixm,mmv,z.mix_musik],[puffer,pv,z.puffer_ms],[gain,gv,z.gain_db]]){if(document.activeElement!==el)el.value=w;v.textContent=w}
if(G){fuelle(eingang,G.eingaenge,z.eingang);fuelle(ausgang,G.ausgaenge.filter(n=>!G.loopback.includes(n)),z.ausgang)}
t.innerHTML=[[L.erkannt,(z.musik_aktiv?'🎵 '+L.musik:'🗣 '+L.sprache)+' · '+L.musik+' '+zahl(z.musik_p,2)+' · '+L.sprache+' '+zahl(z.sprache_p,2)+' → Mix '+Math.round(z.m_ist*100)+' %'],
[L.verzoegerung,z.verzoegerung_ms+' ms'],[L.baender,z.baender+' '+L.von+' 46'],[L.rtf,zahl(z.rtf,2)+(z.rtf>0.9?L.zu_hoch:'')],[L.unterlaeufe,z.unterlaeufe],
[L.verworfen,zahl(z.verworfen/48000,1)+' s'],[L.systemton,z.systemton],[L.fehler,fx(z.fehler)]]
.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('')})}
g();u();setInterval(u,1000);setInterval(g,5000)</script>"""


def handler_fabrik(audio):
    seite = SEITE.replace("__VERSION__", VERSION).replace(
        "__QUELLEN__", "".join(f'<li><a href="{u}" target=_blank>{t}</a></li>' for t, u in QUELLEN))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def antwort(self, body, typ="application/json"):
            b = body.encode(); self.send_response(200)
            self.send_header("Content-Type", typ + "; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self.antwort(seite, "text/html")
            if u.path == "/logo.svg":
                return self.antwort(LOGO, "image/svg+xml")
            if u.path == "/geraete":
                ins, outs = audio.geraete()
                return self.antwort(json.dumps(dict(eingaenge=ins, ausgaenge=outs,
                                                    loopback=[o for o in outs if ist_loopback(o)],
                                                    blackhole=any("BlackHole" in n for n in ins))))
            if u.path == "/set":
                if "an" in q: Z.an = q["an"] == "1"
                if "gain" in q: Z.gain_db = float(q["gain"])
                if "mix" in q: Z.mix = max(0, min(100, int(q["mix"])))
                if "mix_musik" in q: Z.mix_musik = max(0, min(100, int(q["mix_musik"])))
                if "musik_an" in q: Z.musik_an = q["musik_an"] == "1"
                if "puffer" in q: Z.puffer_ms[Z.modus] = int(q["puffer"])
                if "khz" in q and int(q["khz"]) in bandgrenze.GRENZEN_KHZ and int(q["khz"]) != Z.khz:
                    Z.khz = int(q["khz"]); Z.neu.set()
                    with EIN_LOCK: EIN_LOCK.notify()
                if q.get("modus") in ("echtzeit", "offline") and q["modus"] != Z.modus:
                    Z.modus = q["modus"]; Z.neu.set()
                    with EIN_LOCK: EIN_LOCK.notify()
                if "eingang" in q or "ausgang" in q:
                    threading.Thread(target=audio.starte, args=(q.get("eingang", Z.eingang), q.get("ausgang", Z.ausgang)),
                                     daemon=True).start()
                Z.sichere()
            if u.path == "/ende":
                Z.ende.set()
            puffer = Z.puffer_ms[Z.modus]
            grund = 100 + 63 if Z.modus == "echtzeit" else (E + F) * 1000 // SR
            return self.antwort(json.dumps(dict(an=Z.an, modus=Z.modus, puffer_ms=puffer, gain_db=Z.gain_db, mix=Z.mix,
                                                rtf=Z.rtf, unterlaeufe=Z.unterlaeufe, verworfen=Z.verworfen,
                                                verzoegerung_ms=grund + puffer, fehler=Z.fehler, khz=Z.khz, baender=Z.baender, mix_musik=Z.mix_musik,
                                                musik_an=Z.musik_an, musik_aktiv=Z.musik_aktiv, musik_p=Z.musik_p,
                                                sprache_p=Z.sprache_p, m_ist=Z.m_ist,
                                                eingang=Z.eingang, ausgang=Z.ausgang, systemton=ausgabe())))
    return H


def main():
    log(f"— Start voice-isolator-live {VERSION}, torch {torch.__version__}")
    Z.lade()
    vorher = ausgabe()
    if len(sys.argv) > 1:
        Z.ausgang = sys.argv[1]
    log(f"Systemton vorher {vorher!r}")
    print("Modelle laden …", flush=True)
    rt = VoiceIsolator.load("inworld/voice-isolator", mode="streaming", lookahead_frames=2, device="mps")
    off = VoiceIsolator.load("inworld/voice-isolator", mode="offline", device="mps")
    audio = Audio(vorher)
    atexit.register(audio.zurueck)
    signal.signal(signal.SIGTERM, lambda *a: Z.ende.set())
    global KLANGERKENNUNG
    KLANGERKENNUNG = Klang()
    threading.Thread(target=arbeiter, args=(rt, off), daemon=True).start()
    audio.starte(Z.eingang, Z.ausgang)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler_fabrik(audio))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"läuft: http://127.0.0.1:{PORT}  ({Z.eingang} → {Z.ausgang})", flush=True)
    try:
        while not Z.ende.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    Z.sichere()
    audio.stoppe(); srv.shutdown()
    atexit.unregister(audio.zurueck)
    log("— Ende")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("main: " + traceback.format_exc()); raise
