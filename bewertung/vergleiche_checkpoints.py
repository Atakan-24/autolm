"""VERGLEICH ZWEIER CHECKPOINTS AUF DEMSELBEN TEST-SPLIT -- gepaart, mit Trennschaerfe.

Ergaenzt `workflow/vergleiche_lauefe.py`: das vergleicht TRAININGSKURVEN
(maskierter Validierungsverlust ueber die Schritte). Dieses Skript vergleicht
die BEWERTUNGSERGEBNISSE zweier Staende -- also die Ausgaben von
`workflow/erzeuge.py` -- und beantwortet die Frage, die der Verlustvergleich
offen laesst: traegt ein niedrigerer Validierungsverlust bis in die
Aufgaben-Metriken?

    python bewertung/vergleiche_checkpoints.py \
      --a bewertung/ergebnisse/stufe4-bestrun-best1850.json \
      --b bewertung/ergebnisse/stufe4-bestrun-end2400.json \
      --out bewertung/ergebnisse/stufe4-best-gegen-ende.json

FUENF ENTSCHEIDUNGEN, JEDE GEGEN EINEN KONKRETEN FEHLSCHLUSS:

1. GEPAART, NICHT UNABHAENGIG. Beide Staende werden auf DENSELBEN Testfaellen
   bewertet. Ein Zwei-Stichproben-Test behandelt sie als unabhaengig und
   verschenkt damit Trennschaerfe: die Schwankung zwischen den Faellen
   (manche Vorlagen sind fuer jedes Modell schwer) faellt bei der Paarung
   heraus. Gepaart wird ueber die Fall-ID, nicht ueber die Reihenfolge.

2. McNEMAR EXAKT (Binomial), NICHT Chi-Quadrat. Die Chi-Quadrat-Naeherung ist
   bei wenigen diskordanten Paaren unzuverlaessig; hier liegen sie im Bereich
   von 20 bis 40. Der exakte Test braucht keine Naeherung und keine
   Stetigkeitskorrektur, ueber die man sich streiten muesste.

3. JACCARD ZWEIMAL, UND DIE BEDINGTE FASSUNG IST NICHT DAS HAUPTMASS.
   "Nur Faelle, in denen BEIDE Staende etwas Gueltiges erzeugt haben" klingt
   fairer, konditioniert aber auf ein Ergebnis, das NACH der Behandlung
   entsteht -- wer mehr gueltige Ausgaben liefert, aendert damit die
   Teilmenge, auf der er bewertet wird (Collider). Hauptmass ist deshalb die
   unbedingte Fassung, in der eine ungueltige Ausgabe als 0 zaehlt.

4. FUER DIE JACCARD-WERTE DER PERMUTATIONSTEST, NICHT DER VORZEICHENTEST.
   Berichtet wird ein MITTELWERT, also muss auch der Mittelwert getestet
   werden. Der Vorzeichentest wirft die Betraege weg und testet den Median --
   bei asymmetrisch verteilten Differenzen kann er ein deutliches Ergebnis
   verfehlen. Alle drei Werte werden ausgegeben (Permutation, Wilcoxon,
   Vorzeichen), damit die Testwahl nachpruefbar ist und nicht nachtraeglich
   nach dem guenstigsten p gegriffen werden kann.

5. NICHT-SIGNIFIKANZ WIRD NICHT ALS GLEICHHEIT AUSGEGEBEN. Zu jedem
   Unterschied kommt ein Bootstrap-Konfidenzintervall. Ein Intervall wie
   [-21, +7] Prozentpunkte sagt "die Messung entscheidet es nicht" -- ein
   blosses p groesser 0,05 wuerde als "kein Unterschied" gelesen.
"""

import argparse
import json
import random
from math import comb, erfc, sqrt
from pathlib import Path

TORE = ["tor0_kurzschrift", "tor1_json", "tor2_struktur", "tor3_verbindungen",
        "gueltig_at_1", "gueltig_at_k", "abgebrochen"]
JACCARD = ["typen_jaccard", "kanten_jaccard"]
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 12345


def lade(pfad: Path):
    roh = json.loads(pfad.read_text(encoding="utf8"))
    return roh["zusammenfassung"], {f["id"]: f for f in roh["einzel"]}


def mcnemar_exakt(nur_a: int, nur_b: int) -> float:
    """Zweiseitiger exakter Binomialtest auf den diskordanten Paaren."""
    n = nur_a + nur_b
    if n == 0:
        return 1.0
    k = min(nur_a, nur_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(treffer: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = treffer / n
    nenner = 1 + z * z / n
    mitte = (p + z * z / (2 * n)) / nenner
    halb = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / nenner
    return (max(0.0, mitte - halb), min(1.0, mitte + halb))


def wilcoxon_signed_rank(diffs):
    """Wilcoxon-Vorzeichen-Rang, zweiseitig, Normalnaeherung mit Bindungskorrektur.

    Der Vorzeichentest wirft die BETRAEGE weg und testet damit den Median,
    waehrend berichtet wird der MITTELWERT -- Test und Kennzahl messen dann
    verschiedene Dinge. Wilcoxon nutzt die Rangfolge der Betraege und ist
    deutlich trennschaerfer, wenn die Betraege asymmetrisch verteilt sind.
    """
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n < 6:
        return None
    paare = sorted(((abs(x), 1 if x > 0 else -1) for x in d), key=lambda t: t[0])
    raenge, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and paare[j + 1][0] == paare[i][0]:
            j += 1
        mittel = (i + j) / 2 + 1
        for k in range(i, j + 1):
            raenge[k] = mittel
        i = j + 1
    wplus = sum(r for r, (_, vz) in zip(raenge, paare) if vz > 0)
    mu = n * (n + 1) / 4
    bindungen = {}
    for betrag, _ in paare:
        bindungen[betrag] = bindungen.get(betrag, 0) + 1
    korr = sum(t ** 3 - t for t in bindungen.values()) / 48
    sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24 - korr)
    if sigma == 0:
        return None
    z = (wplus - mu) / sigma
    return min(1.0, erfc(abs(z) / sqrt(2)))


def permutationstest(diffs, rng, runden=20_000):
    """Gepaarter Permutationstest auf die MITTLERE Differenz (Vorzeichen-Flip).

    Testet genau die Groesse, die berichtet wird. Braucht keine Annahme ueber
    die Verteilung der Differenzen.
    """
    n = len(diffs)
    if n == 0:
        return 1.0
    beobachtet = abs(sum(diffs) / n)
    treffer = 0
    for _ in range(runden):
        s = 0.0
        for x in diffs:
            s += x if rng.random() < 0.5 else -x
        if abs(s / n) >= beobachtet - 1e-15:
            treffer += 1
    return (treffer + 1) / (runden + 1)


def bootstrap_ki(paare, rng):
    """KI fuer die gepaarte Differenz -- es werden FAELLE gezogen, nicht Werte,
    damit die Paarung erhalten bleibt."""
    n = len(paare)
    if n == 0:
        return (0.0, 0.0)
    diffs = []
    for _ in range(BOOTSTRAP):
        s = 0.0
        for _ in range(n):
            a, b = paare[rng.randrange(n)]
            s += a - b
        diffs.append(s / n)
    diffs.sort()
    return (diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, help="Ergebnis-JSON Stand A")
    p.add_argument("--b", type=Path, required=True, help="Ergebnis-JSON Stand B")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    za, fa = lade(args.a)
    zb, fb = lade(args.b)
    ids = sorted(set(fa) & set(fb))
    if not ids:
        raise SystemExit("keine gemeinsamen Fall-IDs -- verschiedene Test-Splits bewertet?")

    rng = random.Random(BOOTSTRAP_SEED)
    bericht = {
        "a": {"datei": str(args.a), "schritt": za.get("checkpoint_schritt")},
        "b": {"datei": str(args.b), "schritt": zb.get("checkpoint_schritt")},
        "gepaarte_faelle": len(ids),
        "nur_in_a": len(set(fa) - set(fb)),
        "nur_in_b": len(set(fb) - set(fa)),
        "tore": {},
        "jaccard": {},
    }

    print("Gepaart: {} Faelle  (A Schritt {}, B Schritt {})\n".format(
        len(ids), za.get("checkpoint_schritt"), zb.get("checkpoint_schritt")))
    print("{:<20}{:>15}{:>15}{:>10}{:>20}{:>11}".format(
        "Metrik", "A", "B", "Diff", "95%-KI der Diff", "McNemar p"))

    for feld in TORE:
        ka = sum(1 for i in ids if fa[i].get(feld))
        kb = sum(1 for i in ids if fb[i].get(feld))
        nur_a = sum(1 for i in ids if fa[i].get(feld) and not fb[i].get(feld))
        nur_b = sum(1 for i in ids if fb[i].get(feld) and not fa[i].get(feld))
        pw = mcnemar_exakt(nur_a, nur_b)
        ki = bootstrap_ki([(1.0 if fa[i].get(feld) else 0.0,
                            1.0 if fb[i].get(feld) else 0.0) for i in ids], rng)
        bericht["tore"][feld] = {
            "a_anteil": round(ka / len(ids), 4), "b_anteil": round(kb / len(ids), 4),
            "a_wilson": [round(x, 4) for x in wilson(ka, len(ids))],
            "b_wilson": [round(x, 4) for x in wilson(kb, len(ids))],
            "diskordant_nur_a": nur_a, "diskordant_nur_b": nur_b,
            "differenz": round((ka - kb) / len(ids), 4),
            "differenz_ki95": [round(x, 4) for x in ki],
            "mcnemar_p": round(pw, 4),
            "signifikant_005": pw < 0.05,
        }
        print("{:<20}{:>7}/{} {:5.1f}%{:>6}/{} {:5.1f}%{:+9.1f}pp  [{:+5.1f},{:+5.1f}]pp{:>10.4f}".format(
            feld, ka, len(ids), ka / len(ids) * 100, kb, len(ids), kb / len(ids) * 100,
            (ka - kb) / len(ids) * 100, ki[0] * 100, ki[1] * 100, pw))

    print()
    for feld in JACCARD:
        for modus in ("unbedingt", "beidseitig_gueltig"):
            paare = []
            for i in ids:
                va, vb = fa[i].get(feld), fb[i].get(feld)
                if modus == "beidseitig_gueltig":
                    if va is None or vb is None:
                        continue
                    paare.append((va, vb))
                else:
                    paare.append((va or 0.0, vb or 0.0))
            d = [x - y for x, y in paare]
            pos = sum(1 for x in d if x > 1e-9)
            neg = sum(1 for x in d if x < -1e-9)
            p_vz = mcnemar_exakt(pos, neg)        # Vorzeichentest (nur Richtung)
            p_wx = wilcoxon_signed_rank(d)        # Raenge der Betraege
            p_pm = permutationstest(d, rng)       # HAUPTMASS: mittlere Differenz
            ki = bootstrap_ki(paare, rng)
            ma = sum(x for x, _ in paare) / len(paare)
            mb = sum(y for _, y in paare) / len(paare)
            bericht["jaccard"]["{}__{}".format(feld, modus)] = {
                "n": len(paare), "a_mittel": round(ma, 4), "b_mittel": round(mb, 4),
                "differenz": round(ma - mb, 4),
                "differenz_ki95": [round(x, 4) for x in ki],
                "a_besser": pos, "b_besser": neg,
                "permutationstest_p": round(p_pm, 4),
                "wilcoxon_p": None if p_wx is None else round(p_wx, 4),
                "vorzeichentest_p": round(p_vz, 4),
                "signifikant_005": p_pm < 0.05,
                "hauptmass": modus == "unbedingt",
                "warnung": (None if modus == "unbedingt" else
                            "Konditioniert auf ein Ergebnis NACH der Behandlung "
                            "(nur Faelle, die BEIDE Staende loesen). Ein Unterschied "
                            "hier kann allein durch die Auswahl entstehen und ist "
                            "keine Aussage ueber die Staende."),
            }
            print("{:<40} n={:>3}  A {:.3f}  B {:.3f}  Diff {:+.3f}  [{:+.3f},{:+.3f}]  "
                  "perm={:.4f} wilc={} vz={:.4f}{}".format(
                      feld + " [" + modus + "]", len(paare), ma, mb, ma - mb, ki[0], ki[1],
                      p_pm, "n/a" if p_wx is None else "{:.4f}".format(p_wx), p_vz,
                      "  <- konditioniert, siehe Warnung" if modus != "unbedingt" else ""))

    alle_p = ([v["mcnemar_p"] for v in bericht["tore"].values()]
              + [v["permutationstest_p"] for v in bericht["jaccard"].values()
                 if v["hauptmass"]])
    keiner = all(x >= 0.05 for x in alle_p)
    bericht["urteil"] = {
        "tests_gesamt": len(alle_p),
        "davon_signifikant_005": sum(1 for x in alle_p if x < 0.05),
        "kleinstes_p": min(alle_p),
        "lesart": ("Kein Test trennt die beiden Staende. Das heisst NICHT, dass sie gleich "
                   "gut sind -- die Konfidenzintervalle zeigen, welche Unterschiede diese "
                   "Stichprobe schlicht nicht entscheiden kann.") if keiner else
                  ("Mindestens ein Test trennt die Staende; bei mehreren Tests gleichzeitig "
                   "ist die Mehrfachtestung zu beachten."),
    }
    print("\n{} von {} Tests signifikant (kleinstes p = {:.4f}).".format(
        bericht["urteil"]["davon_signifikant_005"], len(alle_p), min(alle_p)))
    print(bericht["urteil"]["lesart"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bericht, indent=2, ensure_ascii=False), encoding="utf8")
        print("\n-> {}".format(args.out))


if __name__ == "__main__":
    main()
