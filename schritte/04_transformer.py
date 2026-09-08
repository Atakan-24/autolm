#!/usr/bin/env python3
"""
SCHRITT 4 -- DER TRANSFORMER: ALLES ZUSAMMENGESETZT

Ab hier PyTorch. Nicht, weil du es nicht selbst koenntest -- Schritt 1 und 3
haben genau das gezeigt -- sondern weil reines Python fuer echtes Training
zu langsam ist.

WAS PYTORCH FUER DICH MACHT
---------------------------
    loss.backward()      <- das, was du in Schritt 1 selbst geschrieben hast
    nn.Linear            <- eine Matrix mal Vektor, wie in Schritt 3
    F.softmax            <- deine softmax()-Funktion aus Schritt 3

Kein einziges neues Konzept. Nur schneller und mit Grafikkarten-Unterstuetzung.

WAS HIER PASSIERT
-----------------
Wir bauen ein vollstaendiges GPT -- klein, aber vollstaendig -- und
trainieren es darauf, Automations-Saetze zu erzeugen. Am Ende siehst du,
wie aus zufaelligem Zeichensalat echte Saetze werden.

    python schritte/04_transformer.py
"""

import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
BPETokenizer = import_module("02_tokenizer").BPETokenizer   # dein Tokenizer aus Schritt 2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.manual_seed(1337)
random.seed(1337)


def trenner(titel):
    print(f"\n{'=' * 72}\n{titel}\n{'=' * 72}")


# ===========================================================================
# DER DATENSATZ
# ===========================================================================
# Eine kleine kuenstliche "Automations-Sprache". Bewusst mit fester Grammatik:
# so laesst sich am Ergebnis ABLESEN, ob das Modell die Struktur gelernt hat
# oder nur Woerter aneinanderreiht.
#
# Das ist die Miniaturfassung des Projektziels: n8n-Workflows haben ebenfalls
# eine strenge Struktur. Wenn ein kleines Modell DIESE Grammatik lernt, ist
# das ein Hinweis darauf, dass es auch JSON-Struktur lernen kann.

AUSLOESER = ["ein neuer lead reinkommt", "eine mail ankommt", "ein anruf endet",
             "ein kontakt antwortet", "ein auftrag reinkommt", "eine rechnung offen ist",
             "ein lead nicht antwortet", "ein formular abgeschickt wird"]
WARTEN = ["sofort", "warte 24 stunden dann", "warte 3 tage dann",
          "warte 14 tage dann", "warte 1 stunde dann"]
AKTIONEN = ["schick eine mail", "speichere es in der datenbank",
            "schick eine erinnerung", "lege eine rechnung an",
            "benachrichtige das team", "erstelle eine aufgabe",
            "schick eine mahnung", "aktualisiere den kontakt"]


def baue_korpus(zeilen=4000):
    saetze = []
    for _ in range(zeilen):
        saetze.append(f"wenn {random.choice(AUSLOESER)} {random.choice(WARTEN)} "
                      f"{random.choice(AKTIONEN)}\n")
    return "".join(saetze)


# ===========================================================================
# DAS MODELL
# ===========================================================================

class Kopf(nn.Module):
    """
    Ein Attention-Kopf -- genau das aus Schritt 3, nur in PyTorch.
    Vergleiche die forward()-Methode Zeile fuer Zeile mit deiner
    attention()-Funktion von dort. Es ist dasselbe.
    """

    def __init__(self, dim, kopf_dim, block):
        super().__init__()
        self.key   = nn.Linear(dim, kopf_dim, bias=False)   # W_k aus Schritt 3
        self.query = nn.Linear(dim, kopf_dim, bias=False)   # W_q
        self.value = nn.Linear(dim, kopf_dim, bias=False)   # W_v
        # Die Maske als feste Dreiecksmatrix. "buffer" heisst: gehoert zum
        # Modell, wird aber NICHT trainiert -- es ist eine Regel, keine
        # Stellschraube.
        self.register_buffer("maske", torch.tril(torch.ones(block, block)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)

        # Frage gegen Etikett, geteilt durch Wurzel(dim) -- wie in Schritt 3
        punkte = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # Causal Mask: nicht in die Zukunft schauen
        punkte = punkte.masked_fill(self.maske[:T, :T] == 0, float("-inf"))
        gewichte = F.softmax(punkte, dim=-1)
        return gewichte @ v          # Values einsammeln, gewichtet


class MehrKoepfe(nn.Module):
    """Mehrere Koepfe parallel, Ergebnisse aneinandergehaengt -- Schritt 3, Teil 4."""

    def __init__(self, anzahl, dim, block):
        super().__init__()
        self.koepfe = nn.ModuleList([Kopf(dim, dim // anzahl, block) for _ in range(anzahl)])
        self.zusammen = nn.Linear(dim, dim)

    def forward(self, x):
        return self.zusammen(torch.cat([k(x) for k in self.koepfe], dim=-1))


class KleinesNetz(nn.Module):
    """
    Nach dem Hinschauen muss auch NACHGEDACHT werden.

    Attention sammelt Information ein ("welche Woerter sind wichtig?").
    Dieses kleine Netz verarbeitet sie ("was bedeutet das jetzt?").
    Beides zusammen ist ein Transformer-Block. Ohne diesen Teil koennte
    das Modell nur umsortieren, nicht schlussfolgern.

    Die Verbreiterung auf 4x und zurueck ist Konvention aus dem
    Original-Papier -- mehr Platz zum Rechnen in der Mitte.
    """

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
    Ein Transformer-Block: Hinschauen, dann Nachdenken.

    Die beiden `x + ...` sind RESIDUAL CONNECTIONS und sie sind nicht
    Kosmetik: ohne sie kommt der Gradient bei tiefen Modellen nicht mehr
    bis nach vorn durch, und die unteren Schichten lernen nichts.
    Das `x +` gibt ihm eine Abkuerzung direkt nach unten.

    LayerNorm davor haelt die Zahlen in einem vernuenftigen Bereich.
    Ohne sie laufen die Werte ueber die Schichten hinweg auseinander,
    und das Training wird instabil -- dasselbe Fehlerbild wie eine zu
    grosse Lernrate in Schritt 1.
    """

    def __init__(self, dim, koepfe, block):
        super().__init__()
        self.aufmerksamkeit = MehrKoepfe(koepfe, dim, block)
        self.denken = KleinesNetz(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.aufmerksamkeit(self.norm1(x))
        x = x + self.denken(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """Ein vollstaendiges, decoder-only Sprachmodell."""

    def __init__(self, vokabular, dim=128, koepfe=4, schichten=3, block=96):
        super().__init__()
        self.block = block
        # Jedes Token bekommt einen lernbaren Vektor. DAS ist die Stelle,
        # an der aus einer Zahl "Bedeutung" wird.
        self.token_embedding = nn.Embedding(vokabular, dim)
        # Und jede Position auch -- ohne sie waere der Satz eine Menge
        # ohne Reihenfolge (siehe Schritt 3, Teil 3).
        self.pos_embedding = nn.Embedding(block, dim)
        self.bloecke = nn.Sequential(*[Block(dim, koepfe, block) for _ in range(schichten)])
        self.norm = nn.LayerNorm(dim)
        # Zurueck aufs Vokabular: fuer JEDES moegliche naechste Token eine Punktzahl
        self.ausgabe = nn.Linear(dim, vokabular)

    def forward(self, idx, ziele=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.pos_embedding(torch.arange(T, device=idx.device))
        x = self.norm(self.bloecke(x))
        logits = self.ausgabe(x)

        if ziele is None:
            return logits, None
        # DER LOSS. Cross-Entropy misst: wie ueberrascht war das Modell vom
        # tatsaechlich naechsten Token? Wenig ueberrascht = kleiner Loss.
        # Dieselbe Rolle wie fehler**2 in Schritt 1, nur passend fuer
        # "welches von N" statt "welche Zahl".
        verlust = F.cross_entropy(logits.view(B * T, -1), ziele.reshape(B * T))
        return logits, verlust

    @torch.no_grad()
    def erzeuge(self, idx, anzahl, temperatur=0.8):
        """Token fuer Token weiterschreiben."""
        for _ in range(anzahl):
            # nur die letzten `block` Token ansehen -- weiter reicht das
            # Kontextfenster nicht
            logits, _ = self(idx[:, -self.block:])
            logits = logits[:, -1, :] / temperatur
            wahrscheinlichkeiten = F.softmax(logits, dim=-1)
            # WUERFELN statt immer das Wahrscheinlichste nehmen. Sonst
            # wiederholt sich das Modell endlos -- "greedy decoding" liefert
            # bei jedem Aufruf denselben Satz.
            naechstes = torch.multinomial(wahrscheinlichkeiten, num_samples=1)
            idx = torch.cat((idx, naechstes), dim=1)
        return idx


# ===========================================================================
# TRAINING
# ===========================================================================

def main():
    print(__doc__)

    # GROESSE AN DIE MASCHINE ANPASSEN -- gemessen, nicht geraten.
    # Auf diesem Server (3 Kerne, 2 GHz, geteilt mit dem Scraper):
    #   dim=128, 3 Schichten, block=96, batch=32  ->  686 ms/Schritt = 17 min
    #   dim=64,  2 Schichten, block=64, batch=16  ->  307 ms/Schritt =  5 min
    # Auf einer Colab-GPU ist die grosse Fassung schneller als die kleine
    # hier. Deshalb ein Schalter statt einer festen Zahl.
    gross = "--gross" in sys.argv
    if gross:
        dim, koepfe, schichten, block, batch, schritte, saetze, vok = 128, 4, 3, 96, 32, 3000, 4000, 512
    else:
        dim, koepfe, schichten, block, batch, schritte, saetze, vok = 64, 4, 2, 64, 16, 1000, 1200, 384
    print(f"  Groesse: {'GROSS (fuer GPU)' if gross else 'KLEIN (fuer diese CPU)'}"
          f"   --  mit --gross umschalten")

    # --- Daten ---
    trenner("1 -- Daten und Tokenizer")
    text = baue_korpus(saetze)
    print(f"  Korpus: {len(text):,} Zeichen, {text.count(chr(10)):,} Saetze")

    print("  Tokenizer trainieren (dein BPE aus Schritt 2) ...", end=" ", flush=True)
    t0 = time.time()
    tok = BPETokenizer()
    tok.trainiere(text[:12000], ziel_groesse=vok)     # auf einem Ausschnitt, reicht
    ids = tok.kodiere(text)
    print(f"fertig in {time.time() - t0:.1f}s")
    print(f"  Vokabular: {len(tok.vokabular)} Token")
    print(f"  Korpus:    {len(ids):,} Token  ({len(text) / len(ids):.2f} Zeichen/Token)")

    daten = torch.tensor(ids, dtype=torch.long)
    grenze = int(0.9 * len(daten))
    train, test = daten[:grenze], daten[grenze:]
    print(f"  Aufteilung: {len(train):,} Training / {len(test):,} Test")
    print("""
  WARUM DIE AUFTEILUNG: auf den Trainingsdaten wird das Modell immer
  besser -- notfalls durch Auswendiglernen. Nur der Test-Teil, den es nie
  gesehen hat, verraet, ob es wirklich etwas gelernt hat.""".rstrip())

    # --- Modell ---
    trenner("2 -- Das Modell")
    modell = MiniGPT(vokabular=len(tok.vokabular), dim=dim, koepfe=koepfe,
                     schichten=schichten, block=block)
    anzahl = sum(p.numel() for p in modell.parameters())
    print(f"  Parameter: {anzahl:,}")
    print(f"  Aufbau:    {schichten} Schichten x {koepfe} Koepfe, {dim} Dimensionen, Kontext {block} Token")
    print(f"""
  {anzahl:,} Stellschrauben. In Schritt 1 waren es zwei (gewicht_x, gewicht_y).
  Gefunden werden sie mit genau demselben Verfahren.""".rstrip())

    def stapel(quelle, groesse=batch):
        start = torch.randint(len(quelle) - block - 1, (groesse,))
        x = torch.stack([quelle[i:i + block] for i in start])
        y = torch.stack([quelle[i + 1:i + block + 1] for i in start])
        return x, y

    @torch.no_grad()
    def messe(quelle, runden=8):
        modell.eval()
        werte = [modell(*stapel(quelle))[1].item() for _ in range(runden)]
        modell.train()
        return sum(werte) / len(werte)

    # --- Vorher ---
    trenner("3 -- VOR dem Training")
    start = torch.zeros((1, 1), dtype=torch.long)
    vorher = tok.dekodiere(modell.erzeuge(start, 120)[0].tolist())
    print(f"  Loss: {messe(test):.3f}   (zufaellig waere {math.log(len(tok.vokabular)):.3f})")
    print(f"\n  Erzeugter Text:\n  {vorher[:200]!r}")
    print("\n  Zeichensalat. Das Modell hat noch nie Text gesehen.")

    # --- Training ---
    trenner("4 -- Training")
    optimierer = torch.optim.AdamW(modell.parameters(), lr=3e-3)
    print(f"  {schritte} Schritte, Lernrate 3e-3, AdamW")
    print("""
  AdamW statt "Gewicht = Gewicht - Lernrate * Steigung" aus Schritt 1:
  es merkt sich, wie sich jede Stellschraube zuletzt bewegt hat, und passt
  ihre Schrittgroesse einzeln an. Dieselbe Idee, nur klueger.
""".rstrip())
    print(f"\n  {'Schritt':>8} | {'Train-Loss':>11} | {'Test-Loss':>10} | {'Zeit':>7}")
    print(f"  {'-' * 8}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 7}")

    t0 = time.time()
    for schritt in range(schritte + 1):
        x, y = stapel(train)
        _, verlust = modell(x, y)

        optimierer.zero_grad(set_to_none=True)   # alte Gradienten loeschen
        verlust.backward()                       # <- Schritt 1, in schnell
        optimierer.step()                        # Stellschrauben drehen

        if schritt % (schritte // 6) == 0:
            print(f"  {schritt:>8} | {messe(train):>11.3f} | {messe(test):>10.3f} "
                  f"| {time.time() - t0:>6.1f}s")

    # --- Nachher ---
    trenner("5 -- NACH dem Training")
    for i in range(3):
        erzeugt = tok.dekodiere(modell.erzeuge(start, 150)[0].tolist())
        zeilen = [z for z in erzeugt.split("\n") if z.strip()][:3]
        print(f"\n  Versuch {i + 1}:")
        for z in zeilen:
            print(f"    {z}")

    # --- Der ehrliche Teil ---
    trenner("6 -- Hat es die Grammatik gelernt, oder auswendig?")
    treffer = 0
    versuche = 60
    bekannt = set()
    for a in AUSLOESER:
        for w in WARTEN:
            for k in AKTIONEN:
                bekannt.add(f"wenn {a} {w} {k}")

    erzeugte = []
    for _ in range(versuche // 3):
        roh = tok.dekodiere(modell.erzeuge(start, 200)[0].tolist())
        erzeugte += [z.strip() for z in roh.split("\n")[1:-1] if z.strip()]
    erzeugte = erzeugte[:versuche]

    gueltig = [z for z in erzeugte if z in bekannt]
    print(f"  Erzeugte vollstaendige Saetze:  {len(erzeugte)}")
    print(f"  davon grammatisch korrekt:      {len(gueltig)} "
          f"({len(gueltig) / max(1, len(erzeugte)) * 100:.0f} %)")
    print(f"  moegliche Saetze insgesamt:     {len(bekannt)}")
    if erzeugte:
        falsch = [z for z in erzeugte if z not in bekannt][:3]
        if falsch:
            print(f"\n  Beispiele fuer FALSCHE Saetze:")
            for z in falsch:
                print(f"    {z}")
        else:
            print(f"\n  Kein einziger falscher Satz.")

    trenner("GESCHAFFT")
    print(f"""
  Du hast ein vollstaendiges Sprachmodell gebaut und trainiert.
  {anzahl:,} Parameter, von zufaellig zu funktionierend.

  Alles darin kennst du schon:

    Embedding        Zahl -> Vektor mit Bedeutung
    Attention        Schritt 3, jetzt in PyTorch
    Causal Mask      nicht in die Zukunft schauen
    Residual + Norm  damit der Gradient bis nach unten durchkommt
    Cross-Entropy    "wie ueberrascht war ich?"
    backward()       Schritt 1, nur schneller

  WICHTIG UND EHRLICH: dieser Korpus ist kuenstlich und klein. Das Modell
  lernt eine Grammatik mit ein paar hundert moeglichen Saetzen. Das ist ein
  Funktionstest fuer den CODE -- kein Beweis, dass es echte Sprache kann.

  Genau dafuer kommt Schritt 5: derselbe Code, aber ein echter Korpus.
  Laeuft es dort, ist der Code richtig. Laeuft es nicht, weisst du, dass
  es an den Daten liegt und nicht an dir.
""".rstrip())


if __name__ == "__main__":
    main()
