#!/usr/bin/env python3
"""
SCHRITT 3 -- ATTENTION: WIE WOERTER AUFEINANDER SCHAUEN

Immer noch keine Bibliothek. Nur Python.

Das ist die Idee, die 2017 alles veraendert hat ("Attention Is All You Need").
Ohne sie gaebe es kein GPT und kein Claude.

DAS PROBLEM, DAS SIE LOEST
--------------------------
Nimm den Satz:

    "wenn ein lead reinkommt schick ihm eine mail"

Worauf bezieht sich "ihm"? Auf "lead". Zwischen den beiden stehen aber drei
andere Woerter. Ein Modell, das Woerter nur der Reihe nach abarbeitet, muss
sich "lead" die ganze Zeit merken -- und vergisst es bei langen Saetzen.

Attention macht es anders: JEDES Wort darf DIREKT auf JEDES vorherige Wort
schauen und selbst entscheiden, welches davon wichtig ist. Kein Merken,
kein Weiterreichen -- ein direkter Blick.

    python schritte/03_attention.py
"""

import math
import random

random.seed(7)


def trenner(titel):
    print(f"\n{'=' * 74}\n{titel}\n{'=' * 74}")


# ===========================================================================
# Kleine Rechenhelfer -- was numpy sonst macht, hier von Hand
# ===========================================================================

def skalarprodukt(a, b):
    """Wie aehnlich sind sich zwei Vektoren? Gross = aehnlich."""
    return sum(x * y for x, y in zip(a, b))


def softmax(zahlen):
    """
    Macht aus beliebigen Zahlen Anteile, die zusammen 1 ergeben.

        [2.0, 1.0, 0.1]  ->  [0.66, 0.24, 0.10]

    Wird ueberall gebraucht, wo etwas "verteilt" werden soll. Hier:
    wie viel Aufmerksamkeit geht an welches Wort.

    Das Abziehen des Maximums aendert am Ergebnis NICHTS, verhindert aber
    einen Ueberlauf bei grossen Zahlen -- Standardtrick, steht in jeder
    echten Implementierung genauso drin.
    """
    groesstes = max(zahlen)
    exponenten = [math.exp(z - groesstes) for z in zahlen]
    summe = sum(exponenten)
    return [e / summe for e in exponenten]


def zufallsmatrix(zeilen, spalten, streuung=0.5):
    return [[random.gauss(0, streuung) for _ in range(spalten)] for _ in range(zeilen)]


def mat_mal_vektor(matrix, vektor):
    return [skalarprodukt(zeile, vektor) for zeile in matrix]


def zeige_matrix(matrix, zeilen_namen, spalten_namen, titel):
    """Gibt eine Aufmerksamkeitsmatrix lesbar aus."""
    breite = max(len(n) for n in spalten_namen) + 2
    breite = max(breite, 7)
    print(f"\n  {titel}")
    print(f"  {'':<12}" + "".join(f"{n:>{breite}}" for n in spalten_namen))
    for name, zeile in zip(zeilen_namen, matrix):
        zellen = "".join(
            f"{'  .   ':>{breite}}" if w < 0.005 else f"{w:>{breite}.2f}"
            for w in zeile
        )
        print(f"  {name:<12}" + zellen)


# ===========================================================================
# TEIL 1 -- DIE DREI ROLLEN: FRAGE, ETIKETT, INHALT
# ===========================================================================
#
# Jedes Wort bekommt drei verschiedene Vektoren. Die Namen sind unglaeubig
# oft erklaert worden; hier die Fassung, die wirklich traegt:
#
#   QUERY  (Frage)    "Wonach suche ich?"
#   KEY    (Etikett)  "Wofuer bin ich zustaendig?"
#   VALUE  (Inhalt)   "Was gebe ich weiter, wenn man mich nimmt?"
#
# Bild: eine Bibliothek. Deine FRAGE wird mit den ETIKETTEN aller Buecher
# verglichen. Wo Frage und Etikett zusammenpassen, nimmst du den INHALT.
#
# Alle drei entstehen aus demselben Wort -- durch drei verschiedene
# Matrizen, die das Modell im Training LERNT. Das sind Stellschrauben,
# genau wie gewicht_x und gewicht_y in Schritt 1.

def teil1_rollen():
    trenner("TEIL 1 -- Query, Key, Value: die drei Rollen jedes Wortes")
    print("""
  Jedes Wort wird zu drei Vektoren:

      QUERY   "Wonach suche ich?"                 <- die Frage
      KEY     "Wofuer bin ich zustaendig?"        <- das Etikett
      VALUE   "Was gebe ich weiter?"              <- der Inhalt

  Ablauf fuer EIN Wort:

      1. Nimm meine QUERY
      2. Vergleiche sie mit den KEYS aller Woerter davor
      3. Wo es passt, gibt es eine hohe Punktzahl
      4. Softmax macht daraus Anteile, die 1 ergeben
      5. Hole dir die VALUES, gewichtet nach diesen Anteilen

  Schritt 2 ist der ganze Zauber: der Vergleich ist ein Skalarprodukt.
  Zwei Vektoren, die in dieselbe Richtung zeigen, ergeben eine grosse Zahl.
  "Passt zusammen" ist also nichts anderes als "zeigt in dieselbe Richtung".""".rstrip())


# ===========================================================================
# TEIL 2 -- EIN ATTENTION-KOPF, VON HAND
# ===========================================================================

def attention(woerter, W_q, W_k, W_v, maskieren=True):
    """
    Ein einzelner Attention-Kopf. Das ist der ganze Mechanismus.

    woerter  -- Liste von Vektoren, einer je Token
    W_q/k/v  -- die gelernten Matrizen (hier: zufaellig)
    maskieren-- darf ein Wort in die Zukunft schauen?
    """
    dim = len(W_q)

    # 1. Jedes Wort bekommt seine drei Rollen
    queries = [mat_mal_vektor(W_q, w) for w in woerter]
    keys    = [mat_mal_vektor(W_k, w) for w in woerter]
    values  = [mat_mal_vektor(W_v, w) for w in woerter]

    aufmerksamkeit, ausgaben = [], []
    for i, q in enumerate(queries):
        # 2. Meine Frage gegen alle Etiketten
        punkte = [skalarprodukt(q, k) for k in keys]

        # 3. Durch Wurzel(dim) teilen.
        #    OHNE DAS BRICHT ES: bei grossen Dimensionen werden die
        #    Skalarprodukte gross, softmax wird dadurch fast zu einem
        #    "alles auf das groesste" -- und das Modell lernt nichts mehr,
        #    weil die Gradienten verschwinden. Steht so im Original-Papier.
        punkte = [p / math.sqrt(dim) for p in punkte]

        # 4. CAUSAL MASK -- der Punkt, an dem ein GPT entsteht.
        #    Wort 3 darf Wort 1 und 2 sehen, aber nicht 4 und 5.
        #    Ohne diese Zeile koennte das Modell beim Training in die
        #    Antwort spicken: "sage das naechste Wort voraus" waere
        #    trivial, wenn es das naechste Wort schon sehen darf.
        if maskieren:
            punkte = [p if j <= i else -float("inf") for j, p in enumerate(punkte)]

        # 5. Anteile bilden
        gewichte = softmax(punkte)
        aufmerksamkeit.append(gewichte)

        # 6. Values einsammeln, gewichtet
        ausgabe = [sum(g * v[d] for g, v in zip(gewichte, values)) for d in range(dim)]
        ausgaben.append(ausgabe)

    return aufmerksamkeit, ausgaben


def teil2_ungelernt():
    trenner("TEIL 2 -- Ein Attention-Kopf mit ZUFAELLIGEN Gewichten")

    satz = ["wenn", "ein", "lead", "reinkommt", "schick", "ihm", "mail"]
    dim = 8
    woerter = zufallsmatrix(len(satz), dim)      # Embeddings, hier zufaellig
    W_q, W_k, W_v = (zufallsmatrix(dim, dim) for _ in range(3))

    gewichte, _ = attention(woerter, W_q, W_k, W_v, maskieren=True)
    zeige_matrix(gewichte, satz, satz,
                 "Wer schaut wie stark auf wen?  (Zeile = schaut, Spalte = wird angeschaut)")

    print("""
  ZWEI DINGE SIND HIER WICHTIG:

  1. Die obere rechte Haelfte ist LEER. Das ist die Causal Mask.
     "wenn" sieht nur sich selbst. "mail" sieht alles davor.
     Genau deshalb kann so ein Modell Text FORTSETZEN.

  2. Der Inhalt ist Unsinn -- die Gewichte sind zufaellig, das Modell
     wurde nie trainiert. Es schaut irgendwohin. Was FEHLT, ist nicht
     der Mechanismus, sondern das Training.""".rstrip())
    return satz, dim


# ===========================================================================
# TEIL 3 -- EIN KOPF, DER ETWAS KANN
# ===========================================================================
#
# Um zu zeigen, dass der Mechanismus wirklich etwas leisten KANN, bauen wir
# die Gewichte einmal von Hand so, dass ein sinnvolles Verhalten entsteht.
#
# Aufgabe: "schau immer auf das Wort direkt vor dir."
# So ein Kopf existiert in echten Modellen wirklich -- er ist ein Baustein
# der sogenannten Induction Heads, mit denen Modelle Muster fortsetzen.

def teil3_gebauter_kopf(satz):
    trenner("TEIL 3 -- Ein Kopf, dem wir von Hand beibringen: 'schau nach links'")

    n = len(satz)
    # Wir geben jedem Wort seine POSITION als Vektor mit. Genau dafuer gibt
    # es in echten Modellen die Positional Embeddings: ohne sie waere ein
    # Satz fuer Attention nur eine ungeordnete Menge -- "hund beisst mann"
    # und "mann beisst hund" waeren identisch.
    dim = 2
    woerter = [[math.cos(i * 0.9), math.sin(i * 0.9)] for i in range(n)]

    # Query: "ich suche die Position, die eins vor meiner liegt"
    # (Drehung um 0.9 zurueck)
    w = 0.9
    W_q = [[math.cos(w), math.sin(w)], [-math.sin(w), math.cos(w)]]
    W_k = [[1.0, 0.0], [0.0, 1.0]]        # Key: einfach die eigene Position
    W_v = [[1.0, 0.0], [0.0, 1.0]]

    # Schaerfer machen, damit die Verteilung deutlich wird statt verwaschen
    skaliert = [[x * 12 for x in v] for v in woerter]
    gewichte, _ = attention(skaliert, W_q, W_k, W_v, maskieren=True)

    zeige_matrix(gewichte, satz, satz, "Derselbe Mechanismus, andere Gewichte:")
    print("""
  Die Diagonale ist um eins nach links verrutscht. Jedes Wort schaut
  jetzt gezielt auf seinen Vorgaenger -- "ihm" schaut auf "schick",
  "mail" schaut auf "ihm".

  NICHTS am Code hat sich geaendert. Nur die Zahlen in W_q.

  DAS ist der Punkt: der Mechanismus ist fest, das VERHALTEN steckt in
  den Gewichten -- und die findet das Training selbst, so wie in
  Schritt 1 die 2 und die 3 gefunden wurden.""".rstrip())


# ===========================================================================
# TEIL 4 -- WARUM MEHRERE KOEPFE
# ===========================================================================

def teil4_mehrere_koepfe(satz, dim=8):
    trenner("TEIL 4 -- Multi-Head: mehrere Blicke gleichzeitig")

    woerter = zufallsmatrix(len(satz), dim)
    print("\n  Drei Koepfe, dieselben Woerter, verschiedene zufaellige Gewichte.")
    print("  Jeder achtet auf etwas anderes:\n")

    for kopf in range(3):
        W_q, W_k, W_v = (zufallsmatrix(dim, dim) for _ in range(3))
        gewichte, _ = attention(woerter, W_q, W_k, W_v, maskieren=True)
        # Wohin schaut das letzte Wort bei diesem Kopf am staerksten?
        letzte = gewichte[-1]
        ziel = max(range(len(letzte)), key=lambda i: letzte[i])
        verteilung = " ".join(f"{satz[i][:4]}:{letzte[i]:.2f}" for i in range(len(satz)))
        print(f"    Kopf {kopf + 1}: {satz[-1]!r} schaut am staerksten auf {satz[ziel]!r}")
        print(f"            {verteilung}")

    print("""
  Ein einzelner Kopf kann nur EINE Art von Beziehung erfassen. Echte
  Sprache hat viele gleichzeitig: wer gehoert zu wem, was ist das Verb,
  worauf bezieht sich das Fuerwort, welches Thema laeuft gerade.

  Deshalb laufen mehrere Koepfe parallel und ihre Ergebnisse werden
  zusammengefuegt. GPT-2 klein hat 12 Koepfe pro Schicht, 12 Schichten.""".rstrip())


if __name__ == "__main__":
    print(__doc__)
    teil1_rollen()
    satz, dim = teil2_ungelernt()
    teil3_gebauter_kopf(satz)
    teil4_mehrere_koepfe(satz, dim)

    trenner("GESCHAFFT")
    print("""
  Du hast jetzt Attention selbst gebaut. Die vier Teile:

    QUERY/KEY/VALUE   drei Rollen je Wort, aus gelernten Matrizen
    SKALARPRODUKT     "passt zusammen" = "zeigt in dieselbe Richtung"
    CAUSAL MASK       nicht in die Zukunft schauen -- macht daraus ein GPT
    SOFTMAX           Punktzahlen werden zu Anteilen, die 1 ergeben

  Und die Erkenntnis aus Teil 3: der Mechanismus ist immer derselbe.
  Was ein Kopf TUT, steht allein in seinen Zahlen -- und die findet
  Backpropagation aus Schritt 1.

  Ein Transformer ist jetzt fast nur noch Zusammenbauen:

      Token -> Embedding -> [ Attention + kleines Netz ] x N -> naechstes Token

  NAECHSTER SCHRITT (04): genau dieser Zusammenbau -- in PyTorch, weil
  reines Python fuer echtes Training zu langsam ist. PyTorch macht dabei
  exakt das, was du in Schritt 1 selbst geschrieben hast: Gradienten
  ausrechnen. Nur schneller und auf der Grafikkarte.
""".rstrip())
