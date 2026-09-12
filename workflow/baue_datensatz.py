"""
DATENSATZ-BAU -- von 1.978 Vorlagen zum Trainings-Memmap, in einem Lauf.

    python workflow/baue_datensatz.py                      # voller Bau
    python workflow/baue_datensatz.py --hoechstens 200     # Rauchtest

Reihenfolge, und warum sie so ist:

  1. Vorlagen laden (workflow/vorlagen.py), nach Seed mischen.
  2. SPLIT NACH VORLAGEN-ID: 90 % Training, 5 % Validierung, 5 % Test.
     Nicht nach Zeile. Sonst stuende eine Mutante von Vorlage 3830 im
     Training und das Original im Test -- die Antwort waere geleckt, die
     Testzahl wertlos. Exakt dieselbe Fehlerklasse wie das `ANSAGE:`-Leck
     im tgcrm-Anruf-Klassifikator, und deshalb hier ein HARTER ABBRUCH,
     kein Warnhinweis: pruefe_split() beendet den Lauf, wenn eine ID in
     zwei Splits liegt oder eine Test-Kurzschrift wortgleich im Training
     vorkommt.
  3. Mutanten NUR fuer den Trainings-Split (workflow/mutationen.py), jede
     durch den Validator. Validierung und Test bleiben unveraendert echte
     Vorlagen -- gemessen wird gegen die Wirklichkeit, nicht gegen
     Erzeugnisse des eigenen Generators.
  4. Instruktionen deterministisch (workflow/instruktionen.py).
  5. Tokenizer auf dem Trainings-Text trainieren, alle Splits kodieren,
     als uint16-Strom schreiben -- dasselbe Format wie
     daten/tinystories_train.bin, damit kern/trainiere.py unveraendert
     laeuft.

Alles Erzeugte landet unter daten/workflow/ (gitignored, in Minuten neu
erzeugbar). Der Bericht (bericht.json) ist klein und enthaelt jede Zahl,
die das README nennt.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from workflow import instruktionen, katalog, kurzschrift as ks, mutationen as mu, vorlagen  # noqa: E402
from workflow.tokenizer_workflow import WorkflowTokenizer  # noqa: E402

WURZEL = Path(__file__).parent.parent
AUS_STANDARD = WURZEL / "daten" / "workflow"


def pruefe_split(train: list[dict], val: list[dict], test: list[dict]) -> None:
    """Harter Abbruch bei jedem Leck -- eine Warnung wuerde ueberlesen."""
    ids = [{b["quelle_id"] for b in s} for s in (train, val, test)]
    for i in range(3):
        for j in range(i + 1, 3):
            gemeinsam = ids[i] & ids[j]
            if gemeinsam:
                raise SystemExit(f"SPLIT-LECK: Vorlagen-IDs in zwei Splits: {sorted(gemeinsam)[:10]}")
    train_texte = {b["kurzschrift"] for b in train}
    for name, s in (("val", val), ("test", test)):
        doppelt = [b["quelle_id"] for b in s if b["kurzschrift"] in train_texte]
        if doppelt:
            raise SystemExit(f"SPLIT-LECK: {len(doppelt)} {name}-Kurzschriften wortgleich im Training: {doppelt[:10]}")


def baue_beispiele(vs: list[dict], kat: dict, varianten: int, seed: int) -> list[dict]:
    aus = []
    for v in vs:
        text = ks.serialisiere(v["wf"])
        for k in range(varianten):
            aus.append({
                "quelle_id": v["id"],
                "ops": [],
                "instruktion": instruktionen.erzeuge(v["wf"], v["metadata"], kat,
                                                     f"{seed}:orig:{v['id']}:{k}", True),
                "kurzschrift": text,
            })
    return aus


def schreibe_jsonl(pfad: Path, zeilen: list[dict]) -> None:
    with open(pfad, "w", encoding="utf8") as f:
        for z in zeilen:
            f.write(json.dumps(z, ensure_ascii=False) + "\n")


def kodiere_split(tok: WorkflowTokenizer, beispiele: list[dict], pfad: Path) -> dict:
    ids = []
    laengen = []
    for b in beispiele:
        t = tok.kodiere_beispiel(b["instruktion"], b["kurzschrift"])
        laengen.append(len(t))
        ids.extend(t)
    arr = np.array(ids, dtype=np.uint16)
    if arr.max(initial=0) >= 65536:
        raise SystemExit("Token-ID passt nicht in uint16")
    arr.tofile(pfad)
    laengen.sort()
    return {
        "beispiele": len(beispiele),
        "token": int(len(arr)),
        "token_je_beispiel_median": laengen[len(laengen) // 2] if laengen else 0,
        "token_je_beispiel_p90": laengen[int(len(laengen) * 0.9)] if laengen else 0,
        "token_je_beispiel_max": laengen[-1] if laengen else 0,
        "datei": pfad.name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=katalog.DB_STANDARD)
    p.add_argument("--out", type=Path, default=AUS_STANDARD)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mutanten-je-vorlage", type=int, default=12)
    p.add_argument("--varianten-je-original", type=int, default=3,
                   help="verschiedene Instruktions-Formulierungen je Original im Training")
    p.add_argument("--vokabular", type=int, default=4096)
    p.add_argument("--tokenizer-textmenge", type=int, default=1_500_000,
                   help="Zeichen, auf denen der BPE trainiert (Zeit ~ linear)")
    p.add_argument("--hoechstens", type=int, default=None, help="nur die ersten N Vorlagen (Rauchtest)")
    args = p.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    kat = katalog.lade()
    vs = vorlagen.lade(args.db, args.hoechstens)
    rng = random.Random(args.seed)
    rng.shuffle(vs)
    n = len(vs)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    test_v, val_v, train_v = vs[:n_test], vs[n_test:n_test + n_val], vs[n_test + n_val:]
    print(f"Vorlagen: {n}  ->  train {len(train_v)} / val {len(val_v)} / test {len(test_v)}")

    # Mutanten: nur Training. Versionen-Tabelle ebenfalls nur aus dem Training.
    gruppen = mu.tauschgruppen(kat)
    versionen = mu.versionen_je_typ(train_v)
    train = baue_beispiele(train_v, kat, args.varianten_je_original, args.seed)
    ops_zaehler, mutanten = {}, 0
    t1 = time.time()
    for v in train_v:
        for m in mu.erzeuge_mutanten(v, args.mutanten_je_vorlage, args.seed, kat, gruppen, versionen):
            mutanten += 1
            for op in m["ops"]:
                ops_zaehler[op] = ops_zaehler.get(op, 0) + 1
            train.append({
                "quelle_id": v["id"],
                "ops": m["ops"],
                "instruktion": instruktionen.erzeuge(m["wf"], None, kat,
                                                     f"{args.seed}:mut:{v['id']}:{mutanten}", False),
                "kurzschrift": ks.serialisiere(m["wf"]),
            })
    print(f"Mutanten: {mutanten} ({time.time() - t1:.0f}s), Operationen: {ops_zaehler}")
    val = baue_beispiele(val_v, kat, 1, args.seed)
    test = baue_beispiele(test_v, kat, 1, args.seed)
    pruefe_split(train, val, test)
    rng.shuffle(train)

    # Tokenizer auf Trainings-Text (Instruktion + Kurzschrift)
    korpus = "\n".join(b["instruktion"] + "\n" + b["kurzschrift"] for b in train)
    probe = korpus[:args.tokenizer_textmenge]
    t2 = time.time()
    tok = WorkflowTokenizer.trainiere(probe, args.vokabular)
    print(f"Tokenizer: {tok.groesse} Token, trainiert auf {len(probe):,} Zeichen in {time.time() - t2:.0f}s")
    tok.speichere(args.out / "tokenizer.pkl")

    schreibe_jsonl(args.out / "train.jsonl", train)
    schreibe_jsonl(args.out / "val.jsonl", val)
    schreibe_jsonl(args.out / "test.jsonl", test)
    (args.out / "split.json").write_text(json.dumps({
        "train": sorted(v["id"] for v in train_v),
        "val": sorted(v["id"] for v in val_v),
        "test": sorted(v["id"] for v in test_v),
    }), encoding="utf8")

    t3 = time.time()
    stat = {name: kodiere_split(tok, s, args.out / f"{name}.bin")
            for name, s in (("train", train), ("val", val), ("test", test))}
    print(f"Kodiert in {time.time() - t3:.0f}s: " +
          ", ".join(f"{k} {v['token']:,} Token" for k, v in stat.items()))

    # Kompression als Kennzahl: Zeichen je Token auf dem Trainings-Korpus
    zeichen_je_token = len(korpus) / max(1, stat["train"]["token"])

    meta = {
        "vokabular_groesse": tok.groesse,
        "eos_id": tok.eos_id,
        "spezial": tok.spezial,
        "seed": args.seed,
        "tokenizer_sha256": hashlib.sha256((args.out / "tokenizer.pkl").read_bytes()).hexdigest()[:16],
        "gesamt_token": stat["train"]["token"],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    bericht = {
        "erzeugt_von": "workflow/baue_datensatz.py",
        "seed": args.seed,
        "vorlagen": {"gesamt": n, "train": len(train_v), "val": len(val_v), "test": len(test_v)},
        "mutanten": mutanten,
        "mutanten_je_vorlage_ziel": args.mutanten_je_vorlage,
        "operationen": ops_zaehler,
        "varianten_je_original": args.varianten_je_original,
        "vokabular": tok.groesse,
        "zeichen_je_token": round(zeichen_je_token, 2),
        "splits": stat,
        "dauer_s": round(time.time() - t0),
    }
    (args.out / "bericht.json").write_text(json.dumps(bericht, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(bericht, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
