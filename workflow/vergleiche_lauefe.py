"""VERGLEICH ZWEIER TRAININGSLAEUFE -- gleiche Konfiguration, andere Daten.

Der B8-Ablationslauf aendert nur die Trainingsdaten gegen V3. Dieses Skript
liest beide `verlauf.jsonl`-Dateien und schreibt die Vergleichszahlen, statt
eine einzelne guenstige Logzeile herauszugreifen:

    python workflow/vergleiche_lauefe.py \
      --basis bewertung/ergebnisse/stufe4-v3-verlauf-7M.jsonl \
      --kandidat daten/workflow_b8/ckpt/verlauf.jsonl \
      --out bewertung/ergebnisse/stufe4-b8-gegen-v3.json

Die Einordnung bleibt bewusst eng: gleicher Seed, Modell, Block und
Schrittbudget sind Voraussetzung. Das Skript sagt nur, ob sich der
maskierte Validierungsverlust verschiebt; Gültigkeit/semantische Qualität
werden danach separat mit workflow/erzeuge.py gemessen.
"""

import argparse
import json
from pathlib import Path


def lade(pfad: Path) -> dict[int, dict]:
    zeilen = {}
    for zeile in pfad.read_text(encoding="utf8").splitlines():
        if not zeile.strip():
            continue
        e = json.loads(zeile)
        schritt = int(e["schritt"])
        if schritt in zeilen:
            raise SystemExit(f"{pfad}: Schritt {schritt} doppelt")
        zeilen[schritt] = e
    if not zeilen:
        raise SystemExit(f"{pfad}: leer")
    return zeilen


def zusammenfassung(basis: dict[int, dict], kandidat: dict[int, dict]) -> dict:
    gemeinsam = sorted(set(basis) & set(kandidat))
    if not gemeinsam:
        raise SystemExit("keine gemeinsamen Log-Schritte")
    stichproben = []
    for s in gemeinsam:
        b, k = basis[s], kandidat[s]
        stichproben.append({
            "schritt": s,
            "basis_val": b["val"],
            "kandidat_val": k["val"],
            "delta_val": round(k["val"] - b["val"], 4),
            "basis_luecke": round(b["val"] - b["train"], 4),
            "kandidat_luecke": round(k["val"] - k["train"], 4),
        })
    best_b = min(basis.values(), key=lambda e: e["val"])
    best_k = min(kandidat.values(), key=lambda e: e["val"])
    letztes = stichproben[-1]
    return {
        "gemeinsame_schritte": [x["schritt"] for x in stichproben],
        "letzter_gemeinsamer_schritt": letztes,
        "minimum_basis": {"schritt": best_b["schritt"], "val": best_b["val"]},
        "minimum_kandidat": {"schritt": best_k["schritt"], "val": best_k["val"]},
        "stichproben": stichproben,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", type=Path, required=True)
    p.add_argument("--kandidat", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    bericht = {
        "erzeugt_von": "workflow/vergleiche_lauefe.py",
        "basis": str(args.basis),
        "kandidat": str(args.kandidat),
        **zusammenfassung(lade(args.basis), lade(args.kandidat)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bericht, indent=2, ensure_ascii=False), encoding="utf8")
    letzter = bericht["letzter_gemeinsamer_schritt"]
    print(f"Schritt {letzter['schritt']}: Delta Val {letzter['delta_val']:+.4f}; "
          f"Luecke Basis {letzter['basis_luecke']:.4f} / Kandidat {letzter['kandidat_luecke']:.4f}")


if __name__ == "__main__":
    main()
