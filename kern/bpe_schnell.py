"""
SCHNELLERER BPE-ENCODER -- gleicher Algorithmus, andere Datenstruktur.

WARUM DER LANGSAM war (gemessen, nicht vermutet): schritte/02_tokenizer.py
durchsucht bei JEDER einzelnen Verschmelzung die KOMPLETTE Token-Folge neu
(finde_paare + verschmelze, beide voll linear). Bei ~2000 gelernten
Verschmelzungen und langen Texten ist das O(n * k) -- gemessen auf 1 MB
Text: 55 Sekunden zum Kodieren. Hochgerechnet auf einen 2-GB-Korpus waeren
das ueber 30 Stunden. Nicht brauchbar.

DER FIX aendert den ALGORITHMUS NICHT -- Byte Pair Encoding bleibt exakt
dasselbe Verfahren, dieselben gelernten Verschmelzungen, dieselbe
Reihenfolge. Es aendert nur, WIE die Folge nach jedem Merge aktualisiert
wird:

  VORHER: nach jedem Merge die GANZE Folge neu durchlaufen und zaehlen.
  JETZT:  eine verkettete Liste + ein Haufen (heapq). Nach einem Merge
          werden nur die 2-4 Nachbarpaare an der Merge-Stelle aktualisiert,
          nicht die ganze Folge.

Das ist derselbe Sprung wie bei enrich-signals.mjs im tgcrm-Projekt: nicht
ein neues Verfahren, sondern dieselbe Aufgabe ohne unnoetige Wiederholung.

Ergebnis siehe kern/mess_bpe_speedup.py.
"""

import heapq
from collections import defaultdict


class SchnellerBPETokenizer:
    """
    Gleiches Ergebnis wie schritte/02_tokenizer.py::BPETokenizer,
    andere interne Umsetzung beim KODIEREN. Das TRAINIEREN (einmalig,
    auf einem kleinen Ausschnitt) bleibt bei der einfachen Fassung --
    dort ist die Geschwindigkeit egal, es passiert nur einmal.
    """

    def __init__(self):
        self.verschmelzungen = {}          # (a,b) -> neue id, in Lernreihenfolge
        self.rang = {}                     # (a,b) -> Position in der Lernreihenfolge (kleiner = frueher)
        self.vokabular = {i: bytes([i]) for i in range(256)}

    # ---- Training: unveraendert von der einfachen Fassung uebernommen ----
    def trainiere(self, text, ziel_groesse, zeige=False):
        from collections import Counter

        def finde_paare(folge):
            paare = Counter()
            for a, b in zip(folge, folge[1:]):
                paare[(a, b)] += 1
            return paare

        def verschmelze(folge, paar, neue_id):
            neu, i = [], 0
            while i < len(folge):
                if i < len(folge) - 1 and (folge[i], folge[i + 1]) == paar:
                    neu.append(neue_id)
                    i += 2
                else:
                    neu.append(folge[i])
                    i += 1
            return neu

        folge = list(text.encode("utf-8"))
        anzahl = ziel_groesse - 256
        for schritt in range(anzahl):
            paare = finde_paare(folge)
            if not paare:
                break
            bestes = max(paare, key=paare.get)
            if paare[bestes] < 2:
                break
            neue_id = 256 + schritt
            folge = verschmelze(folge, bestes, neue_id)
            self.verschmelzungen[bestes] = neue_id
            self.rang[bestes] = schritt
            self.vokabular[neue_id] = self.vokabular[bestes[0]] + self.vokabular[bestes[1]]
        return folge

    # ---- Kodieren: HIER liegt die Beschleunigung ----------------------
    def kodiere(self, text):
        """
        Verkettete Liste ueber Array-Indizes (naechster/vorheriger Index)
        plus ein Min-Heap nach RANG (Lernreihenfolge -- fruehe
        Verschmelzungen zuerst, genau wie beim Training). Nach jedem Merge
        werden nur die zwei neu entstandenen Nachbarpaare in den Heap
        gelegt, nicht die ganze Folge neu durchsucht.

        Der Heap kann veraltete Eintraege enthalten (ein Paar, dessen
        Nachbar sich laengst geaendert hat) -- die werden beim Ziehen an
        `aktiv[i]` erkannt und einfach uebersprungen. Das ist der
        Standardtrick fuer "lazy deletion" in einem Heap ohne Entfernen-
        Operation.
        """
        if not self.verschmelzungen:
            return list(text.encode("utf-8"))

        roh = list(text.encode("utf-8"))
        n = len(roh)
        if n < 2:
            return roh

        naechster = list(range(1, n)) + [-1]
        vorheriger = [-1] + list(range(0, n - 1))
        aktiv = [True] * n
        wert = roh[:]  # aktueller Token-Wert an jeder (noch aktiven) Position

        heap = []
        def biete_an(i):
            j = naechster[i]
            if j == -1 or not aktiv[i] or not aktiv[j]:
                return
            paar = (wert[i], wert[j])
            r = self.rang.get(paar)
            if r is not None:
                heapq.heappush(heap, (r, i, j))

        for i in range(n):
            biete_an(i)

        while heap:
            r, i, j = heapq.heappop(heap)
            # veraltet? -- entweder i/j nicht mehr aktiv, oder das Paar an
            # dieser Stelle hat sich seit dem Einfuegen geaendert
            if not aktiv[i] or not aktiv[j] or naechster[i] != j:
                continue
            paar = (wert[i], wert[j])
            if self.rang.get(paar) != r:
                continue

            neue_id = self.verschmelzungen[paar]
            wert[i] = neue_id
            aktiv[j] = False

            k = naechster[j]
            naechster[i] = k
            if k != -1:
                vorheriger[k] = i

            p = vorheriger[i]
            if p != -1:
                biete_an(p)
            biete_an(i)

        ergebnis = []
        i = 0
        while i != -1:
            if aktiv[i]:
                ergebnis.append(wert[i])
            i = naechster[i]
        return ergebnis

    def dekodiere(self, ids):
        roh = b"".join(self.vokabular[i] for i in ids)
        return roh.decode("utf-8", errors="replace")
