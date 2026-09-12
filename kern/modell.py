"""
DAS MODELL -- herausgeloest aus schritte/04_transformer.py.

WARUM HERAUSGELOEST: ab Schritt 5 brauchen Pretraining, Ablation,
Skalierungskurve und Interpretierbarkeit dasselbe Modell. Vier Kopien
davon liefen garantiert auseinander -- und dann misst die Ablation ein
anderes Modell als das, was trainiert wurde.

`schritte/04_transformer.py` bleibt als LEHRDATEI unveraendert stehen:
dort ist jede Zeile erklaert. Hier steht die Fassung, mit der gearbeitet
wird.

ZWEI ATTENTION-PFADE, EIN ERGEBNIS
----------------------------------
    MehrKoepfe         Lehrpfad -- eine Python-Schleife ueber die Koepfe,
                       je Kopf eigene Linear-Schichten. Gut lesbar,
                       langsam: jeder Kopf ist ein eigener Kernel-Aufruf.

    MehrKoepfeSchnell  Ein einziges fusioniertes QKV-Linear plus
                       F.scaled_dot_product_attention. Dasselbe Ergebnis,
                       deutlich weniger Aufrufe.

`test_qkv_gleichheit.py` weist nach, dass beide bei gleichen Gewichten auf
1e-5 dasselbe liefern. OHNE DIESEN NACHWEIS waere der schnelle Pfad
wertlos: eine Beschleunigung, die nebenbei das Ergebnis aendert, ist keine
Beschleunigung, sondern ein Fehler mit Stoppuhr.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ===========================================================================
# ATTENTION -- Lehrpfad
# ===========================================================================

class Kopf(nn.Module):
    """Ein einzelner Attention-Kopf. Siehe schritte/03_attention.py."""

    def __init__(self, dim, kopf_dim, block):
        super().__init__()
        self.key = nn.Linear(dim, kopf_dim, bias=False)
        self.query = nn.Linear(dim, kopf_dim, bias=False)
        self.value = nn.Linear(dim, kopf_dim, bias=False)
        self.register_buffer("maske", torch.tril(torch.ones(block, block)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        punkte = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        punkte = punkte.masked_fill(self.maske[:T, :T] == 0, float("-inf"))
        return F.softmax(punkte, dim=-1) @ v


class MehrKoepfe(nn.Module):
    """Lehrpfad: mehrere Koepfe als Python-Schleife."""

    def __init__(self, anzahl, dim, block):
        super().__init__()
        self.anzahl = anzahl
        self.dim = dim
        self.koepfe = nn.ModuleList([Kopf(dim, dim // anzahl, block) for _ in range(anzahl)])
        self.zusammen = nn.Linear(dim, dim)

    def forward(self, x):
        return self.zusammen(torch.cat([k(x) for k in self.koepfe], dim=-1))


# ===========================================================================
# ATTENTION -- schneller Pfad
# ===========================================================================

class MehrKoepfeSchnell(nn.Module):
    """
    Dasselbe, in einem Rutsch.

    ZWEI UNTERSCHIEDE ZUM LEHRPFAD, beide rein technisch:

    1. EIN Linear(dim, 3*dim) statt 3*anzahl einzelner Linear-Schichten.
       Q, K und V aller Koepfe entstehen in einem Matrixprodukt. Die
       Rechnung ist identisch -- nur werden statt 3*anzahl kleinen
       Aufrufen ein grosser gemacht, und genau das entscheidet auf der GPU.

    2. F.scaled_dot_product_attention statt der Formel von Hand. Die
       Funktion waehlt selbst die beste verfuegbare Umsetzung (auf
       passender Hardware FlashAttention) und baut die Causal Mask selbst,
       ohne die volle TxT-Matrix zu materialisieren.

    Mathematisch ist NICHTS anders. Das ist keine Behauptung, sondern in
    test_qkv_gleichheit.py nachgewiesen.
    """

    def __init__(self, anzahl, dim, block):
        super().__init__()
        assert dim % anzahl == 0, "dim muss durch die Kopfzahl teilbar sein"
        self.anzahl = anzahl
        self.dim = dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.zusammen = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        kopf_dim = C // self.anzahl
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, Koepfe, T, kopf_dim): jeder Kopf bekommt seinen
        # eigenen Abschnitt, alle rechnen gleichzeitig statt nacheinander.
        q = q.view(B, T, self.anzahl, kopf_dim).transpose(1, 2)
        k = k.view(B, T, self.anzahl, kopf_dim).transpose(1, 2)
        v = v.view(B, T, self.anzahl, kopf_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.zusammen(y)


def uebertrage_gewichte(langsam: MehrKoepfe, schnell: MehrKoepfeSchnell) -> None:
    """
    Kopiert die Gewichte des Lehrpfads in den schnellen Pfad.

    Der schnelle Pfad teilt seine Ausgabe in DREI Bloecke (Q, K, V) und
    JEDEN davon nochmal in Koepfe -- in genau dieser Reihenfolge. Der
    Lehrpfad hat pro Kopf drei getrennte Matrizen. Die Zuordnung ist
    deshalb: alle Q-Koepfe untereinander, dann alle K, dann alle V.

    Wer hier die Reihenfolge vertauscht, bekommt ein Modell, das rechnet,
    plausible Zahlen liefert und still falsch ist. Deshalb der Test.
    """
    q = torch.cat([k.query.weight for k in langsam.koepfe], dim=0)
    k_ = torch.cat([k.key.weight for k in langsam.koepfe], dim=0)
    v = torch.cat([k.value.weight for k in langsam.koepfe], dim=0)
    with torch.no_grad():
        schnell.qkv.weight.copy_(torch.cat([q, k_, v], dim=0))
        schnell.zusammen.weight.copy_(langsam.zusammen.weight)
        schnell.zusammen.bias.copy_(langsam.zusammen.bias)


# ===========================================================================
# BLOCK UND MODELL
# ===========================================================================

class KleinesNetz(nn.Module):
    """Nach dem Hinschauen wird nachgedacht. 4x breiter in der Mitte."""

    def __init__(self, dim):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.netz(x)


class Block(nn.Module):
    """
    Hinschauen, dann Nachdenken.

    `x + ...` sind Residual Connections: ohne sie kommt der Gradient bei
    tiefen Modellen nicht mehr bis nach vorn durch. LayerNorm davor haelt
    die Zahlen stabil.

    BEIDES IST HIER BEHAUPTUNG UND WIRD IN STUFE 6 GEMESSEN
    (Ablationsstudie, 6 Konfigurationen x 3 Seeds). Bis dahin steht es als
    Begruendung da, nicht als Befund.
    """

    def __init__(self, dim, koepfe, block, schnell=True, mit_norm=True, mit_residual=True):
        super().__init__()
        Attn = MehrKoepfeSchnell if schnell else MehrKoepfe
        self.aufmerksamkeit = Attn(koepfe, dim, block)
        self.denken = KleinesNetz(dim)
        # nn.Identity statt LayerNorm, wenn abgeschaltet -- so bleibt der
        # Vorwaertspfad struktur-identisch und die Ablation misst genau
        # eine Sache.
        self.norm1 = nn.LayerNorm(dim) if mit_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if mit_norm else nn.Identity()
        self.mit_residual = mit_residual

    def forward(self, x):
        if self.mit_residual:
            x = x + self.aufmerksamkeit(self.norm1(x))
            x = x + self.denken(self.norm2(x))
        else:
            x = self.aufmerksamkeit(self.norm1(x))
            x = self.denken(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """
    Ein decoder-only Sprachmodell.

    Die Schalter `schnell`, `mit_norm`, `mit_residual`, `mit_position`
    existieren NICHT als Konfigurations-Wildwuchs, sondern weil Stufe 6
    genau diese vier Dinge einzeln messen soll. Jeder Schalter ist ein Arm
    der Ablationsstudie.
    """

    def __init__(self, vokabular, dim=384, koepfe=6, schichten=6, block=256,
                 schnell=True, mit_norm=True, mit_residual=True, mit_position=True):
        super().__init__()
        self.block = block
        self.mit_position = mit_position
        self.token_embedding = nn.Embedding(vokabular, dim)
        self.pos_embedding = nn.Embedding(block, dim) if mit_position else None
        self.bloecke = nn.Sequential(*[
            Block(dim, koepfe, block, schnell, mit_norm, mit_residual)
            for _ in range(schichten)
        ])
        self.norm = nn.LayerNorm(dim) if mit_norm else nn.Identity()
        self.ausgabe = nn.Linear(dim, vokabular)

    def anzahl_parameter(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, ziele=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(torch.arange(T, device=idx.device))
        x = self.norm(self.bloecke(x))
        logits = self.ausgabe(x)
        if ziele is None:
            return logits, None
        # ignore_index=-100 ist die Vorgabe von cross_entropy: Ziele mit -100
        # (Stufe 4, Instruktionsmaske) zaehlen nicht -- hier ausgeschrieben,
        # damit es beim Lesen nicht wie Zufall aussieht.
        verlust = F.cross_entropy(logits.view(B * T, -1), ziele.reshape(B * T), ignore_index=-100)
        return logits, verlust

    @torch.no_grad()
    def erzeuge(self, idx, anzahl, temperatur=0.8, top_k=None):
        for _ in range(anzahl):
            logits, _ = self(idx[:, -self.block:])
            logits = logits[:, -1, :] / temperatur
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            naechstes = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, naechstes), dim=1)
        return idx
