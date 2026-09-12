"""
KONTROLLE: NAECHSTER NACHBAR -- was "abschreiben" auf dem Test-Split erreicht.

Bevor eine Zahl des eigenen Modells etwas bedeutet, braucht sie einen
Vergleich, der KEIN Modell ist: fuer jede Test-Instruktion wird das
Trainingsbeispiel mit der groessten Wort-Ueberlappung (Jaccard ueber
Kleinbuchstaben-Woerter) gesucht und dessen Kurzschrift unveraendert als
Antwort abgegeben.

Diese Antwort ist per Konstruktion gueltig (sie stammt aus dem Validator-
gesicherten Trainingssatz) -- die Gueltigkeitsquote ist also 100 % und
sagt nichts. Was sie sagt: wie viel Typen-Ueberdeckung (Jaccard gegen die
Referenz) allein durch Abschreiben erreichbar ist. Ein Modell, das darunter
liegt, hat nichts gelernt, was Nachschlagen nicht koennte.

    python workflow/baseline_naechster_nachbar.py --daten daten/workflow \
        --out bewertung/ergebnisse/stufe4-baseline-nn.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

WURZEL = Path(__file__).parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "bewertung"))
from workflow import kurzschrift as ks  # noqa: E402
from tore import pruefe_alle_tore  # noqa: E402

_WORT = re.compile(r"[a-z0-9]+")


def woerter(s: str) -> set[str]:
    return set(_WORT.findall(s.lower()))


def lade_jsonl(p: Path) -> list[dict]:
    return [json.loads(z) for z in p.read_text(encoding="utf8").splitlines() if z.strip()]


def typen(kurz: str) -> set[str]:
    return {n["type"] for n in ks.rendere(kurz)["nodes"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daten", type=Path, default=WURZEL / "daten" / "workflow")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    train = lade_jsonl(args.daten / "train.jsonl")
    test = lade_jsonl(args.daten / "test.jsonl")
    index = [(woerter(b["instruktion"]), b) for b in train]

    einzel, jacc, gueltig = [], [], 0
    for t in test:
        w = woerter(t["instruktion"])
        bester, score = None, -1.0
        for tw, b in index:
            s = len(w & tw) / max(1, len(w | tw))
            if s > score:
                bester, score = b, s
        ref, m = typen(t["kurzschrift"]), typen(bester["kurzschrift"])
        j = len(ref & m) / max(1, len(ref | m))
        jacc.append(j)
        ok = pruefe_alle_tore(json.dumps(ks.rendere(bester["kurzschrift"])))["alle_bestanden_ohne_import"]
        gueltig += ok
        einzel.append({"id": f"t{t['quelle_id']}", "nachbar_quelle_id": bester["quelle_id"],
                       "instruktions_aehnlichkeit": round(score, 3), "typen_jaccard": round(j, 3),
                       "gueltig": ok})
    z = {
        "modell": "baseline: naechster Nachbar (Wort-Jaccard der Instruktion)",
        "n": len(test),
        "gueltig": round(gueltig / max(1, len(test)), 3),
        "typen_jaccard_mittel": round(sum(jacc) / max(1, len(jacc)), 3),
        "typen_jaccard_median": round(sorted(jacc)[len(jacc) // 2], 3) if jacc else None,
        "hinweis": "Gueltigkeit ist per Konstruktion 100 % -- die aussagekraeftige Zahl ist die Typen-Ueberdeckung.",
    }
    print(json.dumps(z, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({"zusammenfassung": z, "einzel": einzel}, indent=1,
                                       ensure_ascii=False), encoding="utf8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
