"""Kleine lokale Demo fuer das trainierte Workflow-Modell.

Beispiel (nach einem Trainingslauf):

    python workflow/demo.py --checkpoints daten/workflow_b8/ckpt \
        --tokenizer daten/workflow_b8/tokenizer.pkl \
        --instruktion "Wenn ein Formular eingeht, warte einen Tag und sende eine Mail" \
        --k 4 --out workflow.json

Die Ausgabe wird nicht repariert: ``--k`` bedeutet ausschliesslich mehrere
unabhaengige Modellstichproben und die erste, die die lokalen Tore 1--3
besteht. Damit ist das Verhalten identisch zu ``workflow/erzeuge.py``.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

WURZEL = Path(__file__).parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "bewertung"))

from workflow import kurzschrift as ks  # noqa: E402
from workflow.erzeuge import bewerte_text, erzeuge_ids, lade_modell  # noqa: E402
from workflow.tokenizer_workflow import WorkflowTokenizer  # noqa: E402
from tore import lade_echte_node_typen  # noqa: E402


def waehle_beste(versuche: list[dict]) -> dict:
    """Nimm die erste gueltige Probe, sonst ehrlich die erste erzeugte."""
    if not versuche:
        raise ValueError("mindestens ein Versuch ist erforderlich")
    return next((v for v in versuche if v["gueltig"]), versuche[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Aus Text einen n8n-Workflow mit AutoLM erzeugen")
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--instruktion", required=True)
    p.add_argument("--k", type=int, default=1, help="Stichproben, keine Ausgabe-Reparatur")
    p.add_argument("--temperatur", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--hoechstens-token", type=int, default=768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="optional: nur den gerenderten Workflow speichern")
    args = p.parse_args()
    if args.k < 1:
        p.error("--k muss mindestens 1 sein")

    torch.manual_seed(args.seed)
    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    tok = WorkflowTokenizer.lade(args.tokenizer)
    modell, konfig = lade_modell(args.checkpoints, geraet)
    echte = lade_echte_node_typen()
    prompt = tok.kodiere_prompt(args.instruktion)

    versuche = []
    for _ in range(args.k):
        ids = erzeuge_ids(modell, prompt, tok.eos_id, args.hoechstens_token,
                           args.temperatur, args.top_k, geraet)
        kurz = tok.dekodiere_antwort(ids)
        bewertung = bewerte_text(kurz, None, echte)
        bewertung["kurzschrift"] = kurz
        bewertung["abgebrochen"] = tok.eos_id not in ids
        versuche.append(bewertung)
        if bewertung["gueltig"]:
            break

    beste = waehle_beste(versuche)
    antwort = {
        "modell_schritt": konfig["schritt"],
        "parameter": modell.anzahl_parameter(),
        "geraet": geraet,
        "instruktion": args.instruktion,
        "versuche": len(versuche),
        "gueltig": beste["gueltig"],
        "probleme": beste["probleme"],
        "kurzschrift": beste["kurzschrift"],
    }
    if beste["tor0_kurzschrift"]:
        workflow = ks.rendere(beste["kurzschrift"])
        antwort["workflow"] = workflow
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(antwort, indent=2, ensure_ascii=False))
    if not beste["gueltig"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
