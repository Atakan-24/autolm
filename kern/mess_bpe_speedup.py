"""
MISST DEN GESCHWINDIGKEITSGEWINN DES SCHNELLEN BPE-ENCODERS.

Getrennt vom Korrektheitstest, aus demselben Grund wie bei mess_speedup.py:
"rechnet es dasselbe" ist die wichtigere Frage und darf nicht von "ist es
schneller" verdeckt werden.

    python kern/mess_bpe_speedup.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "schritte"))
from importlib import import_module

LangsamerTok = import_module("02_tokenizer").BPETokenizer
from bpe_schnell import SchnellerBPETokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def baue_testtext(zeichen):
    import random
    random.seed(0)
    woerter = ["the", "cat", "sat", "on", "mat", "once", "upon", "a", "time",
               "there", "was", "little", "girl", "who", "loved", "to", "play",
               "in", "park", "she", "saw", "dog", "and", "said", "hello",
               "they", "became", "friends", "and", "ran", "together"]
    teile, n = [], 0
    while n < zeichen:
        s = " ".join(random.choice(woerter) for _ in range(random.randint(6, 15))) + ". "
        teile.append(s)
        n += len(s)
    return "".join(teile)[:zeichen]


def main():
    print("Trainiere Tokenizer (einmalig, beide Fassungen identisch) ...")
    trainingstext = baue_testtext(200_000)
    langsam = LangsamerTok()
    langsam.trainiere(trainingstext, ziel_groesse=8192)

    schnell = SchnellerBPETokenizer()
    schnell.verschmelzungen = dict(langsam.verschmelzungen)
    schnell.vokabular = dict(langsam.vokabular)
    schnell.rang = {paar: i for i, paar in enumerate(langsam.verschmelzungen)}
    print(f"  Vokabular: {len(langsam.vokabular)} Token\n")

    print(f"{'Testgroesse':>12} | {'langsam (O(n*k))':>18} | {'schnell (Heap)':>15} | {'Faktor':>8}")
    print(f"{'-'*12}-+-{'-'*18}-+-{'-'*15}-+-{'-'*8}")

    faktoren = []
    for groesse in [10_000, 50_000, 200_000, 1_000_000]:
        text = baue_testtext(groesse)

        t0 = time.time()
        a = langsam.kodiere(text)
        dt_langsam = time.time() - t0

        t0 = time.time()
        b = schnell.kodiere(text)
        dt_schnell = time.time() - t0

        assert a == b, f"ABWEICHUNG bei Groesse {groesse} -- Ergebnis waere nicht vertrauenswuerdig"

        faktor = dt_langsam / dt_schnell if dt_schnell > 0 else float("inf")
        faktoren.append(faktor)
        print(f"{groesse:>12,} | {dt_langsam:>15.2f}s | {dt_schnell:>12.4f}s | {faktor:>7.1f}x")

    faktor_gross = faktoren[-1]
    mb_pro_s_schnell = 1.0 / (dt_schnell)  # letzte Messung war 1 MB
    print(f"""
  Bei 1 MB Text: {mb_pro_s_schnell:.1f} MB/s mit dem schnellen Encoder
  (vorher gemessen: 0.018 MB/s mit dem langsamen -- Faktor {faktor_gross:.0f}x)

  Hochgerechnet auf einen 2-GB-Korpus (TinyStories):
    langsam:  {2000 / 0.018 / 60:.0f} Minuten ({2000 / 0.018 / 3600:.1f} Stunden)
    schnell:  {2000 / mb_pro_s_schnell / 60:.1f} Minuten

  DAMIT IST STUFE 1 UEBERHAUPT ERST DURCHFUEHRBAR. Die Ergebnisgleichheit
  ist in test_bpe_gleichheit.py bewiesen (4 Tests, inklusive Zufallstext)
  -- ohne den Beweis waere eine schnellere, aber andere Tokenisierung
  schlimmer als eine langsame richtige: sie faellt erst Stunden spaeter
  in einer falschen Loss-Kurve auf.
""".rstrip())


if __name__ == "__main__":
    main()
