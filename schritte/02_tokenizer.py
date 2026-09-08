#!/usr/bin/env python3
"""
SCHRITT 2 -- WIE AUS TEXT ZAHLEN WERDEN (TOKENIZER)

Wieder keine Bibliothek. Nur Python.

Ein Modell kann nicht mit Buchstaben rechnen, nur mit Zahlen. Der Tokenizer
ist das Stueck, das dazwischen sitzt:

    "Wenn ein neuer Lead reinkommt"
              |
              v   Tokenizer
        [ 847, 12, 2301, 559, 88 ]
              |
              v   Modell
        [ 559, 88, 1204 ]
              |
              v   Tokenizer rueckwaerts
    ", warte 24 Stunden"

Das klingt nach Nebensache. Ist es nicht: der Tokenizer entscheidet, WIE VIEL
Text ueberhaupt in dein Modell passt und wie schwer es zu lernen hat.

    python schritte/02_tokenizer.py
"""

import sys
from collections import Counter

# Die Windows-Konsole laeuft standardmaessig nicht auf UTF-8. Ohne diese
# Zeile stuerzt das Skript ab, sobald ein Emoji oder ein chinesisches
# Zeichen ausgegeben wird -- und zwar mit einer Fehlermeldung, die nach
# einem Fehler im Tokenizer aussieht. Ist aber nur die Ausgabe.
# Genau der Grund, warum BPE auf BYTES arbeitet und nicht auf Zeichen.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def trenner(titel):
    print(f"\n{'=' * 70}\n{titel}\n{'=' * 70}")


# Beispieltext. Bewusst aus deiner Welt -- Automations-Beschreibungen.
# Ein echter Tokenizer wird spaeter auf Millionen solcher Zeilen trainiert.
TEXT = """
wenn ein neuer lead reinkommt warte 24 stunden dann schick eine mail
wenn eine mail ankommt speichere sie in der datenbank
wenn ein neuer kontakt angelegt wird schick eine mail an das team
wenn ein anruf endet speichere die notiz in der datenbank
wenn ein lead nicht antwortet warte 3 tage dann schick eine erinnerung
wenn eine mail nicht ankommt schick eine warnung an das team
wenn ein kontakt antwortet stoppe die erinnerung
wenn ein neuer auftrag reinkommt lege eine rechnung an
wenn eine rechnung offen ist warte 14 tage dann schick eine mahnung
wenn ein anruf reinkommt speichere die nummer in der datenbank
""".strip()


# ===========================================================================
# TEIL 1 -- DIE NAIVE LOESUNG: JEDER BUCHSTABE EINE ZAHL
# ===========================================================================

def teil1_zeichen():
    trenner("TEIL 1 -- Naiv: jedes Zeichen bekommt eine Nummer")

    zeichen = sorted(set(TEXT))
    zu_zahl = {z: i for i, z in enumerate(zeichen)}
    zu_zeichen = {i: z for z, i in zu_zahl.items()}

    probe = "wenn ein lead reinkommt"
    zahlen = [zu_zahl[z] for z in probe]

    print(f"  Vokabular: {len(zeichen)} verschiedene Zeichen")
    print(f"  {''.join(zeichen)!r}\n")
    print(f"  Text:    {probe!r}")
    print(f"  Zahlen:  {zahlen[:14]} ...")
    print(f"  Zurueck: {''.join(zu_zeichen[i] for i in zahlen)!r}")

    print(f"\n  Laenge: {len(probe)} Zeichen -> {len(zahlen)} Zahlen (1:1)")
    print("""
  DAS FUNKTIONIERT -- und ist trotzdem schlecht.

  Ein Modell kann nur eine begrenzte Anzahl Zahlen auf einmal ansehen
  (das "Kontextfenster"). Bei einem Zeichen pro Zahl passt kaum Text hinein.
  Und das Modell muss muehsam selbst lernen, dass 'l','e','a','d'
  zusammengehoeren -- Arbeit, die man ihm abnehmen kann.""".rstrip())
    return len(probe), len(zahlen)


# ===========================================================================
# TEIL 2 -- DIE ANDERE NAIVE LOESUNG: JEDES WORT EINE ZAHL
# ===========================================================================

def teil2_woerter():
    trenner("TEIL 2 -- Auch naiv: jedes ganze Wort bekommt eine Nummer")

    woerter = sorted(set(TEXT.split()))
    zu_zahl = {w: i for i, w in enumerate(woerter)}

    probe = "wenn ein lead reinkommt"
    zahlen = [zu_zahl[w] for w in probe.split()]

    print(f"  Vokabular: {len(woerter)} verschiedene Woerter")
    print(f"  Text:    {probe!r}")
    print(f"  Zahlen:  {zahlen}")
    print(f"\n  Laenge: {len(probe)} Zeichen -> {len(zahlen)} Zahlen. Viel kompakter!")

    print("""
  UND TROTZDEM SCHLECHTER. Zwei Gruende:

  1. UNBEKANNTE WOERTER. Kommt "reinkam" statt "reinkommt", steht es nicht
     im Vokabular -- das Modell ist blind. Bei echtem Text passiert das
     staendig (Namen, Tippfehler, Fachbegriffe, andere Sprachen).

  2. DAS VOKABULAR EXPLODIERT. Deutsch hat Millionen Wortformen
     ("Datenbank", "Datenbanken", "Datenbankeintrag", ...). Jede braeuchte
     eine eigene Zahl -- und das Modell braucht fuer JEDE Zahl im Vokabular
     eigene Parameter.""".rstrip())

    unbekannt = "wenn ein lead reinkam"
    fehlend = [w for w in unbekannt.split() if w not in zu_zahl]
    print(f"\n  Probe: {unbekannt!r}")
    print(f"  -> unbekannt: {fehlend}   (Modell kann damit gar nichts anfangen)")


# ===========================================================================
# TEIL 3 -- BPE: DER WEG DAZWISCHEN
# ===========================================================================
#
# Byte Pair Encoding. Die Idee, die GPT, Claude und Llama alle benutzen.
#
# Start:  jedes Zeichen ist ein Token (wie Teil 1)
# Dann:   suche das haeufigste NEBENEINANDERSTEHENDE Paar und
#         verschmelze es zu EINEM neuen Token. Wiederhole.
#
# Ergebnis: haeufige Woerter werden zu einem Token ("wenn", "eine").
#           Seltene zerfallen in Teile ("Datenbankeintrag" -> "daten|bank|eintrag").
#           Nichts ist je unbekannt, weil zur Not die Einzelzeichen bleiben.

def finde_paare(folge):
    """Zaehlt, wie oft je zwei benachbarte Token nebeneinander stehen."""
    paare = Counter()
    for a, b in zip(folge, folge[1:]):
        paare[(a, b)] += 1
    return paare


def verschmelze(folge, paar, neue_id):
    """Ersetzt jedes Vorkommen von `paar` durch `neue_id`."""
    neu, i = [], 0
    while i < len(folge):
        if i < len(folge) - 1 and (folge[i], folge[i + 1]) == paar:
            neu.append(neue_id)
            i += 2
        else:
            neu.append(folge[i])
            i += 1
    return neu


class BPETokenizer:
    """Ein Byte-Pair-Encoding-Tokenizer. Selbst gebaut, ~60 Zeilen."""

    def __init__(self):
        # Start: die 256 moeglichen Bytes. Damit ist JEDER denkbare Text
        # darstellbar -- auch Emojis, chinesische Zeichen, kaputte Daten.
        # Deshalb kann es bei BPE kein "unbekanntes Wort" geben.
        self.verschmelzungen = {}                       # (a,b) -> neue id
        self.vokabular = {i: bytes([i]) for i in range(256)}

    def trainiere(self, text, ziel_groesse, zeige=False):
        folge = list(text.encode("utf-8"))
        anzahl = ziel_groesse - 256

        for schritt in range(anzahl):
            paare = finde_paare(folge)
            if not paare:
                break
            bestes = max(paare, key=paare.get)
            haeufigkeit = paare[bestes]
            if haeufigkeit < 2:          # lohnt nicht mehr
                break

            neue_id = 256 + schritt
            folge = verschmelze(folge, bestes, neue_id)
            self.verschmelzungen[bestes] = neue_id
            self.vokabular[neue_id] = self.vokabular[bestes[0]] + self.vokabular[bestes[1]]

            if zeige and schritt < 15:
                stueck = self.vokabular[neue_id].decode("utf-8", errors="replace")
                print(f"    {schritt + 1:>3}. {haeufigkeit:>3}x gesehen  ->  neues Token {neue_id}: {stueck!r}")
        return folge

    def kodiere(self, text):
        folge = list(text.encode("utf-8"))
        # Verschmelzungen in derselben Reihenfolge anwenden wie beim Training.
        # Andere Reihenfolge = anderes Ergebnis, deshalb sortiert nach id.
        for paar, neue_id in sorted(self.verschmelzungen.items(), key=lambda p: p[1]):
            folge = verschmelze(folge, paar, neue_id)
        return folge

    def dekodiere(self, ids):
        roh = b"".join(self.vokabular[i] for i in ids)
        return roh.decode("utf-8", errors="replace")


def teil3_bpe():
    trenner("TEIL 3 -- BPE: haeufiges verschmelzen, seltenes zerlegen")

    print("  Training laeuft. Die ersten Verschmelzungen:\n")
    tok = BPETokenizer()
    tok.trainiere(TEXT, ziel_groesse=350, zeige=True)

    print(f"\n  Fertig. Vokabular: {len(tok.vokabular)} Token "
          f"({len(tok.verschmelzungen)} davon selbst gelernt)")

    probe = "wenn ein lead reinkommt"
    ids = tok.kodiere(probe)
    stuecke = [tok.vokabular[i].decode("utf-8", errors="replace") for i in ids]

    print(f"\n  Text:    {probe!r}")
    print(f"  Zerlegt: {stuecke}")
    print(f"  Zahlen:  {ids}")
    print(f"  Zurueck: {tok.dekodiere(ids)!r}")

    # Der entscheidende Test: ein Wort, das im Training NIE vorkam.
    trenner("TEIL 4 -- Der Test: ein Wort, das es nie gesehen hat")
    for fremd in ["reinkam", "Datenbankeintrag", "n8n-workflow", "Grüße 🎉"]:
        ids = tok.kodiere(fremd)
        stuecke = [tok.vokabular[i].decode("utf-8", errors="replace") for i in ids]
        zurueck = tok.dekodiere(ids)
        ok = "OK" if zurueck == fremd else "KAPUTT"
        print(f"  {fremd!r:<22} -> {stuecke}")
        print(f"  {'':<22}    {len(ids)} Token, zurueck: {zurueck!r}  [{ok}]")

    print("""
  KEIN EINZIGES UNBEKANNT. Das ist der ganze Trick: was BPE nicht als
  ganzes Stueck kennt, zerfaellt in kleinere -- notfalls bis auf einzelne
  Bytes. Es gibt keinen Text, den dieser Tokenizer nicht darstellen kann.""".rstrip())
    return tok


# ===========================================================================
# TEIL 5 -- WARUM DAS WICHTIG IST: KOMPRESSION
# ===========================================================================

def teil5_vergleich(tok):
    trenner("TEIL 5 -- Wie viel Text passt ins Modell?")

    zeichen_anzahl = len(TEXT)
    bpe_anzahl = len(tok.kodiere(TEXT))
    wort_anzahl = len(TEXT.split())

    print(f"  Derselbe Text, drei Tokenizer:\n")
    print(f"  {'Verfahren':<22} | {'Token':>7} | {'Zeichen/Token':>14} | Problem")
    print(f"  {'-' * 22}-+-{'-' * 7}-+-{'-' * 14}-+-{'-' * 24}")
    print(f"  {'jedes Zeichen':<22} | {zeichen_anzahl:>7} | "
          f"{1.0:>14.2f} | wenig Text passt rein")
    print(f"  {'jedes Wort':<22} | {wort_anzahl:>7} | "
          f"{zeichen_anzahl / wort_anzahl:>14.2f} | unbekannte Woerter")
    print(f"  {'BPE (selbst gebaut)':<22} | {bpe_anzahl:>7} | "
          f"{zeichen_anzahl / bpe_anzahl:>14.2f} | keines")

    faktor = zeichen_anzahl / bpe_anzahl
    print(f"""
  BPE packt {faktor:.1f}x mehr Text in dieselbe Anzahl Token wie die
  Zeichen-Variante -- ohne je an einem unbekannten Wort zu scheitern.

  WARUM DAS GELD IST: ein Modell mit Kontextfenster 1024 sieht bei
  Zeichen-Tokens ~1024 Zeichen, bei diesem BPE ~{int(1024 * faktor)} Zeichen.
  Dieselbe Rechenleistung, {faktor:.1f}x mehr Inhalt. Und beim Training
  bedeutet es: {faktor:.1f}x weniger Rechenschritte fuer denselben Text.

  Deshalb ist der Tokenizer keine Nebensache. Er entscheidet mit,
  was dein Modell auf kostenloser Hardware ueberhaupt schaffen kann.""".rstrip())


if __name__ == "__main__":
    print(__doc__)
    teil1_zeichen()
    teil2_woerter()
    tok = teil3_bpe()
    teil5_vergleich(tok)

    trenner("GESCHAFFT")
    print("""
  Du hast jetzt einen eigenen Tokenizer -- dasselbe Verfahren, das GPT,
  Claude und Llama benutzen, nur kleiner.

    ZEICHEN   sicher, aber verschwenderisch
    WORT      kompakt, aber scheitert an unbekannten Woertern
    BPE       beides: kompakt UND nie unbekannt

  Wichtig fuer spaeter: der Tokenizer wird EINMAL trainiert und dann
  eingefroren. Aendert man ihn nachtraeglich, bedeuten alle Zahlen im
  Modell etwas anderes -- das Modell ist dann Schrott.

  NAECHSTER SCHRITT (03): der Transformer. Das Stueck, das aus
  [847, 12, 2301] das naechste Token vorhersagt.
""".rstrip())
