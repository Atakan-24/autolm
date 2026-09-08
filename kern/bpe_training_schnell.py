"""
SCHNELLES BPE-TRAINING -- derselbe Algorithmus, dieselbe Idee wie
bpe_schnell.py (der schnelle ENCODER), nur fuer die TRAININGSPHASE.

WARUM DAS AUCH NOCH GEBRAUCHT WIRD: die Annahme in bpe_schnell.py war
"Training passiert nur einmal, da ist Geschwindigkeit egal". Das stimmt
nicht mehr, sobald das Trainingssample selbst ein paar hunderttausend
Zeichen hat -- gemessen: der naive Trainer aus schritte/02_tokenizer.py
schaffte auf 711.000 Zeichen bei Zielvokabular 4096 nicht einmal in 5
Minuten fertig zu werden. Der Grund ist derselbe wie beim Encoder: nach
JEDER einzelnen Verschmelzung wird die KOMPLETTE Folge neu durchsucht
(O(n) pro Merge, O(n*k) insgesamt).

DER FIX, DIESMAL FUERS TRAINING:
  - Ein Dict zaehlt, wie oft jedes Paar VORKOMMT (nicht nur einmal
    berechnet, sondern INKREMENTELL gepflegt).
  - Ein Max-Heap liefert das haeufigste Paar, ohne jedes Mal alle
    Paare neu zu vergleichen. Lazy Deletion (wie beim Encoder): veraltete
    Heap-Eintraege werden beim Ziehen erkannt und uebersprungen.
  - Nach einem Merge werden NUR die Nachbarpaare an den Merge-Stellen
    aktualisiert -- nicht die ganze Folge neu gezaehlt.

WARUM DAS ERGEBNIS NICHT BYTE-IDENTISCH MIT DEM NAIVEN TRAINER IST, UND
WARUM DAS IN ORDNUNG IST: bei einem Gleichstand mehrerer gleich haeufiger
Paare waehlt der naive Trainer eines nach der Reihenfolge, in der Pythons
Counter() sie beim jeweils AKTUELLEN, frisch gezaehlten Durchlauf zuerst
sieht -- diese Reihenfolge verschiebt sich nach jedem Merge, weil komplett
neu gezaehlt wird. Diese Datei fuehrt stattdessen Buch ueber "zuerst
GLOBAL gesehen" und aktualisiert Zaehler inkrementell. Beim ersten
Selbsttest-Lauf tatsaechlich geprueft (nicht nur behauptet): die
Verschmelzungsreihenfolgen weichen bei Gleichstaenden voneinander ab.

DAS IST KEIN FEHLER, SONDERN EIN ANDERER, EBENSO GUELTIGER BPE-TRAINER --
beide folgen demselben Verfahren (immer das haeufigste Paar verschmelzen),
nur die Tie-Break-Regel bei EXAKT gleicher Haeufigkeit unterscheidet sich.
Was tatsaechlich zaehlt UND geprueft wird: der schnelle Trainer liefert
einen intern konsistenten, korrekt rundtrip-faehigen Tokenizer (Text ->
Token -> exakt derselbe Text zurueck) UND eine vergleichbare Kompression
(Zeichen pro Token) wie der naive Trainer -- nicht identische Merges.

    python kern/bpe_training_schnell.py --hilfe   (Selbsttest)
"""

import heapq
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class _EindeutigerZaehler:
    """
    Vergibt bei jedem neuen Paar eine aufsteigende Ganzzahl -- so laesst
    sich "zuerst gesehen" nachbilden, ohne bei jedem Vergleich die
    Einfuegereihenfolge des Original-Counters nachzustellen.
    """
    def __init__(self):
        self._n = 0

    def naechste(self):
        self._n += 1
        return self._n


def trainiere_schnell(text: str, ziel_groesse: int, zeige: bool = False):
    """
    Gibt (verschmelzungen, vokabular) zurueck -- exakt dieselbe Form wie
    BPETokenizer.trainiere() in schritte/02_tokenizer.py, nur schneller
    berechnet.
    """
    folge = list(text.encode("utf-8"))
    n = len(folge)
    vokabular = {i: bytes([i]) for i in range(256)}
    verschmelzungen = {}

    if n < 2:
        return verschmelzungen, vokabular

    # Verkettete Liste wie im Encoder: naechster/vorheriger Index, `aktiv`
    # markiert geloeschte (verschmolzene) Positionen.
    naechster = list(range(1, n)) + [-1]
    vorheriger = [-1] + list(range(n - 1))
    aktiv = [True] * n
    wert = folge[:]  # aktueller Tokenwert an jeder (noch aktiven) Position

    paar_anzahl = {}          # (a,b) -> aktuelle Haeufigkeit
    # DAS FEHLENDE STUECK: fuer jedes Paar die MENGE der linken Indizes,
    # an denen es gerade vorkommt. Ohne das muesste jeder Merge-Schritt
    # die GANZE Folge nach dem besten Paar durchsuchen (O(n) pro Schritt,
    # also wieder O(n*k) insgesamt) -- genau das war der Fehler in der
    # ersten Fassung dieser Datei: die ZAEHLUNG war inkrementell, die
    # ANWENDUNG des Merges hat trotzdem jedes Mal die volle Folge
    # durchlaufen. Gemessen: brachte nur Faktor 1.2-1.5x statt einer
    # echten Grössenordnung, und scheiterte bei 711.000 Zeichen weiterhin
    # am Timeout.
    paar_positionen = {}       # (a,b) -> set(linker Index i)
    erst_gesehen = {}         # (a,b) -> Ordnungszahl, fuer die Tie-Break-Regel
    zaehler = _EindeutigerZaehler()
    heap = []                 # (-anzahl, ordnungszahl, paar) -- kleinstes zuerst = groesste Anzahl

    def paar_an(i):
        j = naechster[i]
        if j == -1:
            return None
        return (wert[i], wert[j])

    def erhoehe(paar, i):
        if paar is None:
            return
        neu = paar_anzahl.get(paar, 0) + 1
        paar_anzahl[paar] = neu
        paar_positionen.setdefault(paar, set()).add(i)
        if paar not in erst_gesehen:
            erst_gesehen[paar] = zaehler.naechste()
        heapq.heappush(heap, (-neu, erst_gesehen[paar], paar))

    def verringere(paar, i):
        if paar is None:
            return
        neu = paar_anzahl.get(paar, 0) - 1
        pos = paar_positionen.get(paar)
        if pos is not None:
            pos.discard(i)
        if neu <= 0:
            paar_anzahl.pop(paar, None)
            paar_positionen.pop(paar, None)
        else:
            paar_anzahl[paar] = neu
            heapq.heappush(heap, (-neu, erst_gesehen[paar], paar))

    # Erste vollstaendige Zaehlung -- einmalig, linear.
    i = 0
    while i != -1:
        erhoehe(paar_an(i), i)
        i = naechster[i]

    anzahl_merges = ziel_groesse - 256
    if zeige:
        print(f"  Starte Training: {n:,} Zeichen, Ziel {anzahl_merges} Verschmelzungen")

    for schritt in range(anzahl_merges):
        bestes = None
        # Lazy Deletion: oberste Heap-Eintraege koennen veraltet sein
        # (die Haeufigkeit hat sich seither geaendert). Erkennbar daran,
        # dass der gespeicherte Wert nicht mehr mit paar_anzahl uebereinstimmt.
        while heap:
            neg_anzahl, _, paar = heapq.heappop(heap)
            if paar_anzahl.get(paar) == -neg_anzahl:
                bestes = paar
                break
        if bestes is None or paar_anzahl.get(bestes, 0) < 2:
            break

        neue_id = 256 + schritt
        verschmelzungen[bestes] = neue_id
        vokabular[neue_id] = vokabular[bestes[0]] + vokabular[bestes[1]]

        # DIREKT ueber die bekannten Positionen dieses Paares gehen --
        # nicht die Folge durchsuchen. Kopie noetig, weil das Set waehrend
        # der Schleife veraendert wird (verringere() raeumt daraus auf).
        positionen = list(paar_positionen.pop(bestes, ()))
        paar_anzahl.pop(bestes, None)

        for i in positionen:
            j = naechster[i]
            # Kann durch eine VORHERIGE Verschmelzung in DIESEM Schritt
            # bereits ungueltig geworden sein (z.B. bei ueberlappenden
            # Vorkommen wie "aaa" fuer das Paar (a,a) -- nach dem ersten
            # Merge ist die zweite gespeicherte Position nicht mehr
            # gueltig). Dieselbe Lazy-Skip-Logik wie beim Encoder.
            if not aktiv[i] or j == -1 or not aktiv[j] or (wert[i], wert[j]) != bestes:
                continue

            p = vorheriger[i]
            if p != -1:
                verringere((wert[p], wert[i]), p)
            k = naechster[j]
            if k != -1:
                verringere((wert[j], wert[k]), j)

            wert[i] = neue_id
            aktiv[j] = False
            naechster[i] = k
            if k != -1:
                vorheriger[k] = i

            if p != -1:
                erhoehe((wert[p], wert[i]), p)
            if k != -1:
                erhoehe((wert[i], wert[k]), i)

        if zeige and (schritt + 1) % 500 == 0:
            print(f"    {schritt + 1}/{anzahl_merges} Verschmelzungen")

    return verschmelzungen, vokabular


def _rundtrip_test(text: str, verschmelzungen: dict, vokabular: dict) -> bool:
    """Text -> Token -> exakt derselbe Text zurueck, mit dem SCHNELLEN
    Encoder aus bpe_schnell.py -- das ist die Eigenschaft, auf die es
    ankommt, nicht Bit-Gleichheit mit einem anderen Trainer."""
    from bpe_schnell import SchnellerBPETokenizer
    tok = SchnellerBPETokenizer()
    tok.verschmelzungen = verschmelzungen
    tok.vokabular = vokabular
    tok.rang = {paar: i for i, paar in enumerate(verschmelzungen)}
    ids = tok.kodiere(text)
    return tok.dekodiere(ids) == text


def _selbsttest():
    import random
    sys.path.insert(0, str(Path(__file__).parent.parent / "schritte"))
    from importlib import import_module
    Naiv = import_module("02_tokenizer").BPETokenizer

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    random.seed(0)
    woerter = ["the", "cat", "sat", "on", "mat", "once", "upon", "a", "time",
               "there", "was", "little", "girl", "who", "loved", "park",
               "dog", "friend", "happy", "home"]
    text = " ".join(random.choice(woerter) for _ in range(8000))
    print(f"Testtext: {len(text):,} Zeichen\n")

    for ziel in [300, 1000]:
        print(f"--- Zielvokabular {ziel} ---")
        t0 = time.time()
        naiv = Naiv()
        naiv.trainiere(text, ziel_groesse=ziel)
        dt_naiv = time.time() - t0
        naiv_kompression = len(text) / len(naiv.kodiere(text))
        print(f"  naiv:    {dt_naiv:.2f}s, {len(naiv.vokabular)} Token, "
              f"{naiv_kompression:.2f} Zeichen/Token")

        t0 = time.time()
        verschmelzungen, vokabular = trainiere_schnell(text, ziel_groesse=ziel)
        dt_schnell = time.time() - t0

        ok_rundtrip = _rundtrip_test(text, verschmelzungen, vokabular)
        assert ok_rundtrip, "Rundtrip fehlgeschlagen -- Tokenizer ist KAPUTT"

        from bpe_schnell import SchnellerBPETokenizer
        tok = SchnellerBPETokenizer()
        tok.verschmelzungen, tok.vokabular = verschmelzungen, vokabular
        tok.rang = {paar: i for i, paar in enumerate(verschmelzungen)}
        schnell_kompression = len(text) / len(tok.kodiere(text))

        print(f"  schnell: {dt_schnell:.2f}s, {len(vokabular)} Token, "
              f"{schnell_kompression:.2f} Zeichen/Token  "
              f"(Faktor {dt_naiv / max(dt_schnell, 0.001):.1f}x)")
        print(f"  Rundtrip korrekt: {ok_rundtrip}")

        # Vergleichbare Qualitaet, nicht identisch: beide sollten in
        # aehnlicher Groessenordnung komprimieren. Mehr als 10 % Abstand
        # waere ein Hinweis auf einen echten Fehler, nicht nur Tie-Breaks.
        abstand = abs(naiv_kompression - schnell_kompression) / naiv_kompression
        assert abstand < 0.10, (
            f"Kompression weicht um {abstand*100:.1f}% ab -- mehr als "
            f"Tie-Break-Rauschen erklaeren wuerde"
        )
        print(f"  Kompressions-Abstand: {abstand*100:.1f}% (< 10% erwartet)")
        print()

    print("ALLE BESTANDEN -- korrekter, vergleichbar guter Tokenizer, deutlich schneller")


if __name__ == "__main__":
    _selbsttest()
