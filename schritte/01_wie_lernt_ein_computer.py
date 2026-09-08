#!/usr/bin/env python3
"""
SCHRITT 1 -- WIE LERNT EIN COMPUTER?

Keine Bibliothek. Kein PyTorch, kein numpy. Nur Python.
Alles, was hier passiert, steht in dieser Datei.

Am Ende dieser Datei hast du gesehen, was "ein Modell wird trainiert"
tatsaechlich bedeutet -- an einem Beispiel, das klein genug ist, um es
komplett zu ueberblicken. Genau dasselbe passiert spaeter in deinem
Sprachmodell, nur milliardenfach.

    python schritte/01_wie_lernt_ein_computer.py
"""


# ===========================================================================
# TEIL 1 -- EINE ZAHL, DIE SICH MERKT, WO SIE HERKOMMT
# ===========================================================================
#
# Das ist die eine Idee, auf der die gesamte KI aufbaut.
#
# Eine normale Zahl in Python weiss nichts ueber sich:
#
#     a = 2.0
#     b = 3.0
#     c = a * b        # c ist 6.0 -- und das ist alles, was c weiss
#
# c weiss NICHT, dass es aus a und b entstanden ist. Damit kann man nicht
# lernen, denn Lernen heisst: "das Ergebnis war falsch -- wer war schuld?"
#
# Also bauen wir eine Zahl, die sich ihre Herkunft merkt.

class Wert:
    """Eine Zahl, die weiss, aus welcher Rechnung sie entstanden ist."""

    def __init__(self, zahl, eltern=(), rechenart="", name=""):
        self.zahl = zahl            # der eigentliche Wert, z.B. 6.0
        self.steigung = 0.0         # <- der GRADIENT. Dazu gleich mehr.
        self.eltern = eltern        # aus welchen Werten bin ich entstanden?
        self.rechenart = rechenart  # durch welche Rechnung?
        self.name = name
        self._rueckwaerts = lambda: None   # wie gebe ich Schuld weiter?

    def __repr__(self):
        etikett = f"{self.name}=" if self.name else ""
        return f"{etikett}{self.zahl:.4f} (Steigung {self.steigung:+.4f})"

    # ---- Rechnen -----------------------------------------------------------
    # Jede Rechenart macht ZWEI Dinge:
    #   1. das Ergebnis ausrechnen  (vorwaerts)
    #   2. festlegen, wie Schuld an die Eltern verteilt wird (rueckwaerts)

    def __add__(self, andere):
        andere = andere if isinstance(andere, Wert) else Wert(andere)
        ergebnis = Wert(self.zahl + andere.zahl, (self, andere), "+")

        def rueckwaerts():
            # PLUS: beide Eltern bekommen die Schuld unveraendert weitergereicht.
            # Erhoehe ich einen Summanden um 1, steigt die Summe um 1.
            self.steigung += 1.0 * ergebnis.steigung
            andere.steigung += 1.0 * ergebnis.steigung

        ergebnis._rueckwaerts = rueckwaerts
        return ergebnis

    def __mul__(self, andere):
        andere = andere if isinstance(andere, Wert) else Wert(andere)
        ergebnis = Wert(self.zahl * andere.zahl, (self, andere), "*")

        def rueckwaerts():
            # MAL: jeder Elternteil bekommt die Schuld multipliziert mit dem
            # ANDEREN Elternteil. Erhoehe ich a um 1 bei b=3, steigt a*b um 3.
            self.steigung += andere.zahl * ergebnis.steigung
            andere.steigung += self.zahl * ergebnis.steigung

        ergebnis._rueckwaerts = rueckwaerts
        return ergebnis

    def __pow__(self, exponent):
        ergebnis = Wert(self.zahl ** exponent, (self,), f"**{exponent}")

        def rueckwaerts():
            # Potenzregel aus der Schule: aus x^n wird n * x^(n-1).
            # Die brauchst du nicht auswendig -- nur wissen, dass hier
            # steht, wie die Schuld weitergereicht wird.
            self.steigung += (exponent * self.zahl ** (exponent - 1)) * ergebnis.steigung

        ergebnis._rueckwaerts = rueckwaerts
        return ergebnis

    def __neg__(self):        return self * -1
    def __sub__(self, a):     return self + (-a)
    def __radd__(self, a):    return self + a
    def __rmul__(self, a):    return self * a

    # ---- Der eigentliche Trick ---------------------------------------------

    def rueckwaerts(self):
        """
        BACKPROPAGATION.

        Das ist DAS Verfahren, mit dem jedes neuronale Netz der Welt lernt --
        auch GPT, auch Claude. Es beantwortet eine einzige Frage:

            "Der Fehler ist X. Welche Stellschraube ist wie sehr schuld?"

        Es geht dafuer rueckwaerts durch die Rechnung und verteilt Schuld,
        Schritt fuer Schritt, bis ganz nach vorn zu den Stellschrauben.

        Mehr ist es nicht. Wirklich nicht.
        """
        # Erst die Reihenfolge bestimmen: bevor ein Wert Schuld weitergibt,
        # muss er seine eigene vollstaendig erhalten haben.
        reihenfolge, besucht = [], set()

        def sortiere(v):
            if id(v) in besucht:
                return
            besucht.add(id(v))
            for elternteil in v.eltern:
                sortiere(elternteil)
            reihenfolge.append(v)

        sortiere(self)

        # Ich selbst bin zu 100 % fuer mich verantwortlich.
        self.steigung = 1.0
        # Und jetzt rueckwaerts durch die ganze Rechnung.
        for v in reversed(reihenfolge):
            v._rueckwaerts()


def trenner(titel):
    print(f"\n{'=' * 66}\n{titel}\n{'=' * 66}")


# ===========================================================================
# TEIL 2 -- SCHULD VERTEILEN, EINMAL VON HAND ANGESEHEN
# ===========================================================================

def teil2_schuld_verteilen():
    trenner("TEIL 2 -- Wer ist schuld am Ergebnis?")

    a = Wert(2.0, name="a")
    b = Wert(3.0, name="b")
    c = Wert(10.0, name="c")

    ergebnis = a * b + c          # = 16.0
    ergebnis.name = "ergebnis"

    print(f"  Rechnung:  a * b + c  =  {a.zahl} * {b.zahl} + {c.zahl}  =  {ergebnis.zahl}")
    print("\n  Jetzt rueckwaerts fragen: wenn ich jede Zahl um 1 erhoehe,")
    print("  um wie viel aendert sich das Ergebnis?\n")

    ergebnis.rueckwaerts()

    print(f"    {a}   <- a haengt an b, also zaehlt a mit dem Faktor b={b.zahl}")
    print(f"    {b}   <- b haengt an a, also zaehlt b mit dem Faktor a={a.zahl}")
    print(f"    {c}   <- c wird nur addiert, also zaehlt es 1:1")

    print("\n  Diese drei Steigungen sind die GRADIENTEN.")
    print("  Nachgerechnet ohne Formel: a von 2 auf 3 erhoehen ->")
    print(f"    3 * {b.zahl} + {c.zahl} = {3 * b.zahl + c.zahl}, "
          f"also {3 * b.zahl + c.zahl - ergebnis.zahl:+.0f}. Genau die Steigung von a.")


# ===========================================================================
# TEIL 3 -- LERNEN
# ===========================================================================
#
# Jetzt die eigentliche Sache. Wir geben dem Computer eine Aufgabe, deren
# Loesung er NICHT kennt, und lassen ihn sie selbst finden.
#
# Aufgabe: aus zwei Zahlen die richtige Antwort berechnen.
#
#     rein: 2, 3   ->   raus soll: 13      (denn 2*2 + 3*3 = 13)
#     rein: 1, 4   ->   raus soll: 14
#     rein: 5, 2   ->   raus soll: 16
#
# Die Regel dahinter ist "2*x + 3*y". Der Computer bekommt sie NICHT gesagt.
# Er bekommt nur die Beispiele und muss die beiden Zahlen selbst finden.

def teil3_lernen():
    trenner("TEIL 3 -- Der Computer findet eine Regel, die ihm niemand sagt")

    beispiele = [((2, 3), 13), ((1, 4), 14), ((5, 2), 16),
                 ((3, 3), 15), ((4, 1), 11), ((0, 5), 15)]

    # DIE STELLSCHRAUBEN. Sie starten bewusst falsch.
    # Richtig waeren 2.0 und 3.0 -- wir setzen absichtlich etwas anderes.
    gewicht_x = Wert(0.1, name="gewicht_x")
    gewicht_y = Wert(0.1, name="gewicht_y")

    lernrate = 0.01     # wie grosse Schritte gemacht werden. Dazu unten mehr.
    runden = 200

    print(f"  Wahrheit (kennt der Computer NICHT): 2.0 und 3.0")
    print(f"  Startwerte (geraten):                {gewicht_x.zahl} und {gewicht_y.zahl}")
    print(f"  Lernrate: {lernrate}   Runden: {runden}\n")
    print(f"  {'Runde':>6} | {'Fehler (Loss)':>14} | {'gewicht_x':>10} | {'gewicht_y':>10}")
    print(f"  {'-' * 6}-+-{'-' * 14}-+-{'-' * 10}-+-{'-' * 10}")

    for runde in range(runden + 1):

        # --- SCHRITT 1: VORWAERTS ------------------------------------------
        # Mit den aktuellen Stellschrauben raten und ausrechnen, wie falsch
        # das war. Diese eine Zahl heisst LOSS.
        loss = Wert(0.0)
        for (x, y), soll in beispiele:
            vorhersage = gewicht_x * x + gewicht_y * y
            fehler = vorhersage - soll
            # Quadriert, damit -3 genauso schlimm zaehlt wie +3
            # und grosse Fehler staerker ins Gewicht fallen.
            loss = loss + fehler ** 2

        # --- SCHRITT 2: RUECKWAERTS ----------------------------------------
        # Wer ist wie sehr schuld an diesem Fehler?
        gewicht_x.steigung = 0.0     # alte Schuld loeschen, sonst summiert sie sich
        gewicht_y.steigung = 0.0
        loss.rueckwaerts()

        if runde % 20 == 0:
            print(f"  {runde:>6} | {loss.zahl:>14.6f} | "
                  f"{gewicht_x.zahl:>10.4f} | {gewicht_y.zahl:>10.4f}")

        # --- SCHRITT 3: STELLSCHRAUBEN DREHEN ------------------------------
        # Die Steigung sagt, in welche Richtung der Fehler GROESSER wird.
        # Also gehen wir in die GEGENrichtung -- daher das Minus.
        # Das ist GRADIENT DESCENT: "den Hang hinunter".
        gewicht_x = Wert(gewicht_x.zahl - lernrate * gewicht_x.steigung, name="gewicht_x")
        gewicht_y = Wert(gewicht_y.zahl - lernrate * gewicht_y.steigung, name="gewicht_y")

    print(f"\n  Gefunden:  {gewicht_x.zahl:.4f} und {gewicht_y.zahl:.4f}")
    print(f"  Wahrheit:  2.0000 und 3.0000")
    print("\n  Niemand hat ihm 2 und 3 gesagt. Er hat sie aus den Beispielen gefunden --")
    print("  nur ueber 'wie falsch war ich' und 'in welche Richtung wird es besser'.")
    print("\n  GENAU DAS passiert beim Training eines Sprachmodells.")
    print("  Statt 2 Stellschrauben sind es dort Millionen, und statt Zahlenpaaren")
    print("  ist die Aufgabe 'welches Wort kommt als naechstes'. Das Verfahren")
    print("  ist Zeile fuer Zeile dasselbe wie hier.")


# ===========================================================================
# TEIL 4 -- DIE LERNRATE, UND WARUM SIE DICH IRGENDWANN AERGERT
# ===========================================================================

def teil4_lernrate():
    trenner("TEIL 4 -- Die Lernrate: zu klein, richtig, zu gross")

    print("  Dieselbe Aufgabe, nur die Schrittgroesse aendert sich.\n")
    print(f"  {'Lernrate':>10} | {'Fehler am Ende':>18} | Was passiert ist")
    print(f"  {'-' * 10}-+-{'-' * 18}-+-{'-' * 34}")

    for lernrate in [0.0001, 0.001, 0.01, 0.05, 0.1]:
        beispiele = [((2, 3), 13), ((1, 4), 14), ((5, 2), 16),
                     ((3, 3), 15), ((4, 1), 11), ((0, 5), 15)]
        gx, gy = Wert(0.1), Wert(0.1)

        for _ in range(200):
            loss = Wert(0.0)
            for (x, y), soll in beispiele:
                loss = loss + (gx * x + gy * y - soll) ** 2
            gx.steigung = gy.steigung = 0.0
            loss.rueckwaerts()
            gx = Wert(gx.zahl - lernrate * gx.steigung)
            gy = Wert(gy.zahl - lernrate * gy.steigung)
            if abs(loss.zahl) > 1e12:      # explodiert
                break

        endwert = loss.zahl
        if endwert != endwert or abs(endwert) > 1e12:
            urteil = "EXPLODIERT -- Schritte zu gross"
            anzeige = "unendlich"
        elif endwert > 1.0:
            urteil = "zu langsam -- kommt nicht an"
            anzeige = f"{endwert:.6f}"
        else:
            urteil = "gut"
            anzeige = f"{endwert:.6f}"
        print(f"  {lernrate:>10} | {anzeige:>18} | {urteil}")

    print("\n  Das ist die haeufigste Stellschraube beim Training -- und die haeufigste")
    print("  Fehlerquelle. Bleibt der Fehler stehen: Lernrate zu klein. Wird er")
    print("  ploetzlich riesig oder 'nan': Lernrate zu gross.")


if __name__ == "__main__":
    print(__doc__)
    teil2_schuld_verteilen()
    teil3_lernen()
    teil4_lernrate()

    trenner("GESCHAFFT")
    print("""
  Du hast gerade gesehen:

    LOSS       -- eine Zahl, die sagt "wie falsch war ich"
    GRADIENT   -- welche Stellschraube ist wie sehr schuld
    BACKPROP   -- das Rueckwaerts-Verteilen dieser Schuld
    LERNRATE   -- wie grosse Schritte gemacht werden

  Das sind die vier Begriffe. Ein Sprachmodell hat keine fuenften.

  NAECHSTER SCHRITT: dieselbe Mechanik, aber die Aufgabe wird
  "welches Wort kommt als naechstes" -- und daraus wird ein Transformer.
""".rstrip())
