"""
BPE-KODIERUNG UEBER MEHRERE KERNE PARALLEL.

WARUM DAS HIER KORREKT IST UND NICHT NUR SCHNELL: TinyStories besteht aus
Millionen unabhaengiger kurzer Geschichten, keinem einzigen 2-GB-Fliesstext.
Jede Geschichte wird UNABHAENGIG kodiert -- eine Verschmelzung kann nie ueber
eine Geschichtengrenze hinweg entstehen, weil der langsame Referenz-Encoder
(schritte/02_tokenizer.py) genauso pro Aufruf arbeitet: ein Text rein, eine
Token-Liste raus. Parallelisieren nach Dokumentgrenze aendert das Ergebnis
also NICHT -- anders als ein willkuerlicher Byte-Schnitt mitten in einem
Dokument, der eine Verschmelzung an der Schnittstelle verhindern koennte.

Das ist der Unterschied zwischen "schnell, aber vielleicht falsch" und
"schnell, weil die Aufgabe von Natur aus in unabhaengige Stuecke zerfaellt".

    python kern/bpe_parallel.py --hilfe   (Selbsttest an synthetischen Daten)
"""

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bpe_schnell import SchnellerBPETokenizer

_worker_tok = None


def _init_worker(verschmelzungen, rang, vokabular):
    """Laeuft einmal je Worker-Prozess -- der Tokenizer wird EINMAL pro
    Prozess aufgebaut, nicht bei jedem einzelnen Dokument neu."""
    global _worker_tok
    _worker_tok = SchnellerBPETokenizer()
    _worker_tok.verschmelzungen = verschmelzungen
    _worker_tok.rang = rang
    _worker_tok.vokabular = vokabular


def _kodiere_stapel(texte):
    return [_worker_tok.kodiere(t) for t in texte]


def kodiere_parallel(tokenizer: SchnellerBPETokenizer, dokumente: list[str],
                      prozesse: int | None = None, stapelgroesse: int = 200):
    """
    Kodiert eine Liste von Dokumenten (z.B. einzelne TinyStories-Geschichten)
    parallel ueber mehrere Kerne. Gibt eine flache Liste von Token-Listen
    zurueck, IN DERSELBEN REIHENFOLGE wie `dokumente` -- imap statt
    imap_unordered, sonst waere die Reihenfolge der Geschichten im Memmap
    zufaellig und ein spaeterer "Position im Datenstrom"-Resume waere falsch.
    """
    # Vorgabe 4, nicht cpu_count()-1: gemessen auf diesem Server (3 physische
    # Kerne, 6 von Python gemeldete logische) lieferte mp.cpu_count()-1=5
    # WENIGER Durchsatz als 4 -- zu viele Prozesse konkurrieren um zu wenige
    # physische Kerne. 4 war auf 9 MB Testdaten das gemessene Optimum
    # (Faktor 2.97x). Auf kiserver (4 echte ARM-Kerne) passt derselbe Wert.
    prozesse = prozesse or 4
    if len(dokumente) < stapelgroesse * 2 or prozesse <= 1:
        # Zu wenig Arbeit fuer Parallelisierung -- der Prozess-Start kostet
        # selbst schon etwas.
        return [tokenizer.kodiere(t) for t in dokumente]

    stapel = [dokumente[i:i + stapelgroesse] for i in range(0, len(dokumente), stapelgroesse)]

    ctx = mp.get_context("spawn")  # portabel, auch unter Windows korrekt
    with ctx.Pool(
        prozesse, initializer=_init_worker,
        initargs=(tokenizer.verschmelzungen, tokenizer.rang, tokenizer.vokabular),
    ) as pool:
        ergebnisse = []
        for stapel_ergebnis in pool.imap(_kodiere_stapel, stapel):
            ergebnisse.extend(stapel_ergebnis)
    return ergebnisse


def _selbsttest():
    """Vergleicht parallel gegen sequentiell auf synthetischen 'Geschichten'."""
    import random
    import time

    random.seed(0)
    woerter = ["the", "cat", "sat", "on", "mat", "once", "upon", "a", "time",
               "there", "was", "little", "girl", "who", "loved", "park"]
    dokumente = []
    for _ in range(4000):
        n = random.randint(20, 80)
        dokumente.append(" ".join(random.choice(woerter) for _ in range(n)))

    tok = SchnellerBPETokenizer()
    tok.trainiere(" ".join(dokumente[:200]), ziel_groesse=4096)

    print(f"Selbsttest: {len(dokumente)} Dokumente, {sum(len(d) for d in dokumente):,} Zeichen")

    t0 = time.time()
    sequentiell = [tok.kodiere(d) for d in dokumente]
    dt_seq = time.time() - t0
    print(f"  sequentiell: {dt_seq:.2f}s")

    t0 = time.time()
    parallel = kodiere_parallel(tok, dokumente)
    dt_par = time.time() - t0
    print(f"  parallel ({max(1, mp.cpu_count() - 1)} Prozesse): {dt_par:.2f}s")

    assert sequentiell == parallel, "PARALLEL WEICHT VOM SEQUENTIELLEN ERGEBNIS AB"
    faktor = dt_seq / dt_par if dt_par > 0 else float("inf")
    print(f"  Faktor: {faktor:.1f}x -- Ergebnisse IDENTISCH (geprueft)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selbsttest()
