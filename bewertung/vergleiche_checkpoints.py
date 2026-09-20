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


def gleiche_spalten(dateien, ids):
    """Findet Tor-Spalten, die ueber ALLE Faelle und Seeds dasselbe Ereignis
    sind, und fasst sie zu einer Messgroesse zusammen.

    Gemessen am 17.09.2026: tor0/tor1/tor3 sind in 98 von 98 Faellen identisch,
    ebenso tor2 und gueltig@k. Sieben Tabellenzeilen sind also vier
    Messgroessen. Wer die Mehrfachtest-Korrektur ueber sieben statt vier
    rechnet, bestraft sich fuer Tests, die er gar nicht gemacht hat.
    """
    werte = {}
    for feld in TORE:
        werte[feld] = tuple(bool(fall[i].get(feld))
                            for _, fall in dateien for i in ids)
    gruppen = {}
    for feld, muster in werte.items():
        gruppen.setdefault(muster, []).append(feld)
    return [sorted(g) for g in gruppen.values()]


def holm(p_werte):
    """Holm-Bonferroni: haelt die Familienfehlerrate ueber ALLE Tests eines
    Laufs. Ohne diese Korrektur wird bei neun gleichzeitigen Tests irgendwann
    einer zufaellig signifikant, und genau der wuerde dann berichtet.
    Gibt ein Dict Name -> angepasster p-Wert zurueck (monoton gemacht)."""
    sortiert = sorted(p_werte.items(), key=lambda kv: kv[1])
    m = len(sortiert)
    angepasst, bisher = {}, 0.0
    for rang, (name, p) in enumerate(sortiert):
        wert = min(1.0, (m - rang) * p)
        bisher = max(bisher, wert)
        angepasst[name] = round(bisher, 4)
    return angepasst


def seed_mittel(dateien, ids, feld, binaer):
    """Wert je Fall, gemittelt ueber die Seeds.

    Binaere Tore werden dabei zur Rate in [0,1], Jaccard unbedingt gerechnet
    (eine ungueltige Ausgabe zaehlt als 0). Zuerst ueber die Seeds mitteln und
    DANN vergleichen senkt den Sampling-Rauschanteil mit der Wurzel aus der
    Zahl der Seeds -- das ist der ganze Zweck des Sweeps.
    """
    werte = {}
    for i in ids:
        xs = []
        for _, fall in dateien:
            v = fall[i].get(feld)
            xs.append((1.0 if v else 0.0) if binaer else (v or 0.0))
        werte[i] = sum(xs) / len(xs)
    return werte


def _seed_sd(dateien, ids, feld, binaer):
    """Streuung derselben Metrik ueber die Seeds -- gleicher Stand, anderer Seed."""
    je_seed = []
    for _, fall in dateien:
        xs = [(1.0 if fall[i].get(feld) else 0.0) if binaer else (fall[i].get(feld) or 0.0)
              for i in ids]
        je_seed.append(sum(xs) / len(xs))
    if len(je_seed) < 2:
        return None
    m = sum(je_seed) / len(je_seed)
    return sqrt(sum((x - m) ** 2 for x in je_seed) / (len(je_seed) - 1))


def mehrseed_vergleich(a_dateien, b_dateien, rng, bericht):
    """Vergleich zweier Staende ueber MEHRERE Seeds je Stand.

    McNemar entfaellt hier bewusst: nach dem Mitteln ueber Seeds sind die Werte
    keine 0/1-Ereignisse mehr, sondern Raten. Getestet wird durchgehend die
    mittlere Differenz (Permutationstest), begleitet vom Bootstrap-Intervall.

    Entscheidend ist die letzte Spalte: die SEED-STREUUNG desselben Standes.
    Ohne sie laesst sich nicht beurteilen, ob ein Checkpoint-Unterschied
    ueberhaupt groesser ist als das Rauschen des Pruefstands. Genau daran ist
    am 17.09.2026 eine erste, voreilige Auswertung gescheitert.
    """
    ids = sorted(set.intersection(*[set(f) for _, f in a_dateien + b_dateien]))
    bericht["gepaarte_faelle"] = len(ids)
    bericht["seeds_a"] = [z.get("seed") for z, _ in a_dateien]
    bericht["seeds_b"] = [z.get("seed") for z, _ in b_dateien]
    print("Gepaart: {} Faelle | Stand A Seeds {} | Stand B Seeds {}".format(
        len(ids), bericht["seeds_a"], bericht["seeds_b"]))
    print()
    print("{:<22}{:>9}{:>9}{:>10}{:>19}{:>9}{:>10}{:>9}".format(
        "Metrik", "A", "B", "Diff", "95%-KI der Diff", "perm p", "Seed-SD", "in SD"))

    for feld in TORE + JACCARD:
        binaer = feld in TORE
        wa = seed_mittel(a_dateien, ids, feld, binaer)
        wb = seed_mittel(b_dateien, ids, feld, binaer)
        paare = [(wa[i], wb[i]) for i in ids]
        d = [x - y for x, y in paare]
        ma = sum(wa[i] for i in ids) / len(ids)
        mb = sum(wb[i] for i in ids) / len(ids)
        ki = bootstrap_ki(paare, rng)
        p_pm = permutationstest(d, rng)
        sd_a = _seed_sd(a_dateien, ids, feld, binaer)
        sd_b = _seed_sd(b_dateien, ids, feld, binaer)
        vorhandene = [x for x in (sd_a, sd_b) if x is not None]
        sd_max = max(vorhandene) if vorhandene else None
        verhaeltnis = (abs(ma - mb) / sd_max) if sd_max else None
        ziel = bericht["tore"] if binaer else bericht["jaccard"]
        ziel[feld] = {
            "a_mittel": round(ma, 4), "b_mittel": round(mb, 4),
            "differenz": round(ma - mb, 4),
            "differenz_ki95": [round(x, 4) for x in ki],
            "permutationstest_p": round(p_pm, 4),
            "signifikant_005": p_pm < 0.05,
            "seed_sd_a": None if sd_a is None else round(sd_a, 4),
            "seed_sd_b": None if sd_b is None else round(sd_b, 4),
            "effekt_in_seed_sd": None if verhaeltnis is None else round(verhaeltnis, 2),
            "hauptmass": True,
        }
        print("{:<22}{:>9.3f}{:>9.3f}{:>+10.3f}  [{:+6.3f},{:+6.3f}]{:>9.4f}{:>10}{:>9}".format(
            feld, ma, mb, ma - mb, ki[0], ki[1], p_pm,
            "n/a" if sd_max is None else "{:.4f}".format(sd_max),
            "n/a" if verhaeltnis is None else "{:.1f}x".format(verhaeltnis)))
    return bericht


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, nargs="+",
                   help="Ergebnis-JSON(s) Stand A -- mehrere = verschiedene Seeds")
    p.add_argument("--b", type=Path, required=True, nargs="+",
                   help="Ergebnis-JSON(s) Stand B -- mehrere = verschiedene Seeds")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    a_dateien = [lade(x) for x in args.a]
    b_dateien = [lade(x) for x in args.b]
    za, fa = a_dateien[0]
    zb, fb = b_dateien[0]
    mehrfach = len(a_dateien) > 1 or len(b_dateien) > 1
    ids = sorted(set.intersection(*[set(f) for _, f in a_dateien + b_dateien]))
    if not ids:
        raise SystemExit("keine gemeinsamen Fall-IDs -- verschiedene Test-Splits bewertet?")

    rng = random.Random(BOOTSTRAP_SEED)
    bericht = {
        "a": {"dateien": [str(x) for x in args.a], "schritt": za.get("checkpoint_schritt")},
        "b": {"dateien": [str(x) for x in args.b], "schritt": zb.get("checkpoint_schritt")},
        "gepaarte_faelle": len(ids),
        "nur_in_a": len(set(fa) - set(fb)),
        "nur_in_b": len(set(fb) - set(fa)),
        "tore": {},
        "jaccard": {},
    }

    if mehrfach:
        mehrseed_vergleich(a_dateien, b_dateien, rng, bericht)
        alle_p = [v["permutationstest_p"] for v in
                  list(bericht["tore"].values()) + list(bericht["jaccard"].values())]
        # Mehrfachtest-Korrektur nur ueber die UNTERSCHEIDBAREN Messgroessen
        gruppen = gleiche_spalten(a_dateien + b_dateien, ids)
        bericht["messgroessen"] = {
            "tor_gruppen": gruppen,
            "tor_spalten": len(TORE),
            "unterscheidbar": len(gruppen) + len(JACCARD),
        }
        roh = {}
        for g in gruppen:
            roh[" = ".join(g)] = bericht["tore"][g[0]]["permutationstest_p"]
        for name, v in bericht["jaccard"].items():
            roh[name] = v["permutationstest_p"]
        ang = holm(roh)
        for g in gruppen:
            for feld in g:
                bericht["tore"][feld]["holm_p"] = ang[" = ".join(g)]
                bericht["tore"][feld]["signifikant_005_holm"] = ang[" = ".join(g)] < 0.05
        for name, v in bericht["jaccard"].items():
            v["holm_p"] = ang[name]
            v["signifikant_005_holm"] = ang[name] < 0.05
        print()
        print("{} Tor-Spalten sind {} unterscheidbare Messgroessen: {}".format(
            len(TORE), len(gruppen),
            " | ".join(" = ".join(g) for g in gruppen)))
        bericht["urteil"] = {
            "tests_gesamt": len(alle_p),
            "davon_signifikant_005_roh": sum(1 for x in alle_p if x < 0.05),
            "davon_signifikant_005_holm": sum(1 for x in ang.values() if x < 0.05),
            "kleinstes_p_roh": min(alle_p),
            "kleinstes_p_holm": min(ang.values()),
            "lesart": ("Ueber mehrere Seeds gemittelt. Massgeblich ist die "
                       "Holm-korrigierte Spalte -- neun gleichzeitige Tests "
                       "liefern sonst zufaellige Treffer. Die Seed-SD je Metrik "
                       "sagt, ob ein Unterschied ueberhaupt groesser ist als das "
                       "Rauschen des Pruefstands."),
        }
        print()
        print("Holm-korrigiert: " + "  ".join(
            "{} {:.4f}{}".format(k, v, "*" if v < 0.05 else "")
            for k, v in sorted(ang.items(), key=lambda kv: kv[1])))
        print()
        print("{} von {} Tests signifikant (kleinstes p = {:.4f}).".format(
            bericht["urteil"]["davon_signifikant_005_roh"], len(alle_p), min(alle_p)))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(bericht, indent=2, ensure_ascii=False), encoding="utf8")
            print()
            print("-> {}".format(args.out))
        return

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
