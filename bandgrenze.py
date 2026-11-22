"""Bandgrenze — das Netz nur über die Bänder unterhalb einer Frequenz rechnen lassen.

Ends:   HDZ 2026-11-22: „die Frequenz begrenzen auf 12 kHz, sollte doch immer als Schalter gehen?“ ·
        „ja, Bandbreite auf voll, 12 kHz, 8 kHz, 6 kHz einbauen und dann CoreML“.
Ways:   Das Modell teilt das Spektrum in 46 Bänder, je Band ein Token; die Rechenlast hängt fast linear an der
        Zahl der Token. Ein Schnitt behält die ersten k Bänder: Encoder, Band-Einbettung und Decoder-Köpfe werden
        auf k gekürzt, die Blöcke (Attention, Mamba) bleiben, sie sind über die Bänder geteilt. Oberhalb des
        Schnitts setzt die Korrektur −Spektrum ein, der Ausgang ist dort still (Tiefpass); den Originalton dort
        liefert der Mix-Regler.
Means:  Umschalten an Ort und Stelle (Attribute tauschen), umkehrbar; die gepackten Bänder des Streams
        (`_PackedBands`) bauen sich beim nächsten Stream neu, weil sich die Parameter geändert haben.
Expects: Umschalten nur zwischen zwei Streams bzw. zwei Fenstern (der Arbeiter setzt bei jedem Wechsel zurück).
Fails:  Das Modell hat nie unvollständige Spektren gesehen: Attention und Mamba über die Bänder laufen in beide
        Richtungen, die tiefen Bänder sehen die hohen. Ob der Schnitt die Maske verschiebt, entscheidet das Ohr.
Tags:   voice-isolator, performance
"""
import torch
from inworld.voice_isolation import streaming

GRENZEN_KHZ = (0, 12, 8, 6)   # 0 = voll

_orig_encode = streaming._PackedBands.encode


def _encode_geschnitten(self, model, spectrum):
    # Pitfall: _PackedBands.gather füllt Lücken mit dem Index 2 * bins (eine Null-Spalte hinter dem Spektrum).
    # Mit gekürzten Bändern zeigte dieser Index sonst mitten ins volle Spektrum.
    cut = model.edges[-1][1]
    if spectrum.shape[1] != cut:
        spectrum = spectrum[:, :cut]
    return _orig_encode(self, model, spectrum)


streaming._PackedBands.encode = _encode_geschnitten


class Bandgrenze:
    """Hält die vollen Bandlisten eines Netzes und schneidet es auf Wunsch."""

    def __init__(self, model):
        self.m = model
        dec = model.denoise_decoder
        self.voll = dict(edges=list(model.edges), encoders=model.encoders, emb=model.band_embedding,
                         dec_edges=list(dec.edges), heads=dec.heads, correction=model.correction)
        self.hz_bin = model.config.sample_rate / model.config.n_fft
        self.khz = 0

    def baender(self, khz):
        if not khz:
            return len(self.voll["edges"])
        return sum(1 for a, _ in self.voll["edges"] if a * self.hz_bin < khz * 1000)

    def setze(self, khz):
        m, v = self.m, self.voll
        k = self.baender(khz)
        m.edges = v["edges"][:k]
        m.encoders = torch.nn.ModuleList(list(v["encoders"])[:k])
        m.band_embedding = torch.nn.Parameter(v["emb"].data[:k], requires_grad=False)
        m.denoise_decoder.edges = v["dec_edges"][:k]
        m.denoise_decoder.heads = torch.nn.ModuleList(list(v["heads"])[:k])
        if k == len(v["edges"]):
            m.correction = v["correction"]
        else:
            cut, orig = m.edges[-1][1], v["correction"]

            def correction(raw, spectrum):
                # Die Lage der Bin-Achse unterscheidet sich zwischen Offline ([b, bins, frames]) und Stream
                # ([..., bins]); sie ist die einzige Achse, in der raw und Spektrum verschieden lang sind.
                d = next(i for i, (a, b) in enumerate(zip(raw.shape[:-1], spectrum.shape)) if a != b)
                c = orig(raw, spectrum.narrow(d, 0, cut))
                return torch.cat([c, -spectrum.narrow(d, cut, spectrum.shape[d] - cut)], dim=d)
            m.correction = correction
        self.khz = khz
        return k
