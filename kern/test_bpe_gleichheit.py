"""
BEWEIST, DASS DER SCHNELLE ENCODER DASSELBE LIEFERT WIE DER LANGSAME.

Dieselbe Regel wie bei test_qkv_gleichheit.py: eine Beschleunigung, die
nebenbei andere Token erzeugt, ist keine Beschleunigung. Sie waere sogar
gefaehrlicher als beim Attention-Pfad -- ein falsch tokenisierter Korpus
faellt erst nach Stunden Training auf, wenn die Loss-Kurve nicht passt.

    python kern/test_bpe_gleichheit.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "schritte"))
from importlib import import_module

LangsamerTok = import_module("02_tokenizer").BPETokenizer
from bpe_schnell import SchnellerBPETokenizer

random.seed(42)


def _trainiertes_paar(text, vok=1024):
    langsam = LangsamerTok()
    langsam.trainiere(text, ziel_groesse=vok)

    schnell = SchnellerBPETokenizer()
    schnell.verschmelzungen = dict(langsam.verschmelzungen)
    schnell.vokabular = dict(langsam.vokabular)
    schnell.rang = {paar: i for i, paar in enumerate(langsam.verschmelzungen)}
    return langsam, schnell


TRAININGSTEXT = """
wenn ein neuer lead reinkommt warte 24 stunden dann schick eine mail
wenn eine mail ankommt speichere sie in der datenbank
wenn ein neuer kontakt angelegt wird schick eine mail an das team
wenn ein anruf endet speichere die notiz in der datenbank
wenn ein lead nicht antwortet warte 3 tage dann schick eine erinnerung
once upon a time there was a little girl who loved to play in the park
she saw a dog and said hello they became friends and ran together
""".strip() * 20


def test_gleiche_ausgabe_auf_trainingstext():
    langsam, schnell = _trainiertes_paar(TRAININGSTEXT)
    proben = ["wenn ein lead reinkommt", "once upon a time",
              TRAININGSTEXT[:500], "hello hello hello"]
    for probe in proben:
        a = langsam.kodiere(probe)
        b = schnell.kodiere(probe)
        assert a == b, f"Abweichung bei {probe!r}:\n  langsam={a}\n  schnell={b}"


def test_gleiche_ausgabe_auf_unbekanntem_text():
    """Der Test, der beim Attention-Pfad am meisten wert war: Randfaelle."""
    langsam, schnell = _trainiertes_paar(TRAININGSTEXT)
    fremd = ["Datenbankeintrag", "n8n-workflow", "xyzxyzxyz",
             "Grüße 🎉", "", "a", "aa", "aaaa aaaa aaaa",
             "wenn wenn wenn ein ein ein lead lead lead"]
    for probe in fremd:
        a = langsam.kodiere(probe)
        b = schnell.kodiere(probe)
        assert a == b, f"Abweichung bei {probe!r}:\n  langsam={a}\n  schnell={b}"
        # Und beide muessen sich verlustfrei zurueckwandeln lassen
        assert langsam.dekodiere(a) == probe


def test_gleich_auf_zufaelligem_text():
    """
    Zufaellige Zeichenfolgen -- der haerteste Test. Zufaelliger Text hat
    keine der Regelmaessigkeiten des Trainingskorpus, deshalb entstehen
    hier die meisten Merges pro Zeichen und damit die meisten Gelegenheiten
    fuer die verkettete Liste, sich zu verheddern.
    """
    langsam, schnell = _trainiertes_paar(TRAININGSTEXT, vok=512)
    alphabet = "abcdefghijklmnopqrstuvwxyz .,!?"
    for laenge in [1, 5, 20, 100, 500]:
        for _ in range(5):
            probe = "".join(random.choice(alphabet) for _ in range(laenge))
            a = langsam.kodiere(probe)
            b = schnell.kodiere(probe)
            assert a == b, f"Abweichung (Laenge {laenge}) bei {probe!r}"


def test_grossen_text_stueckweise_gegen_ganzen():
    """
    Randfall der verketteten Liste: ein Merge, der genau am Rand eines
    zuvor entstandenen Tokens ansetzt, darf keine falschen Nachbarn
    entstehen lassen. Dafuer denselben langen Text einmal am Stueck und
    einmal versetzt geschnitten kodieren und mit dem langsamen Referenz-
    Encoder vergleichen -- beide muessen fuer denselben Teiltext identisch
    sein.
    """
    langsam, schnell = _trainiertes_paar(TRAININGSTEXT)
    lang = TRAININGSTEXT * 3
    a = langsam.kodiere(lang)
    b = schnell.kodiere(lang)
    assert a == b, "Abweichung auf langem, wiederholtem Text"


if __name__ == "__main__":
    fehler = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK    {name}")
            except AssertionError as e:
                print(f"  FEHLT {name}: {e}")
                fehler += 1
    print(f"\n{'ALLE BESTANDEN' if not fehler else f'{fehler} FEHLGESCHLAGEN'}")
    raise SystemExit(1 if fehler else 0)
