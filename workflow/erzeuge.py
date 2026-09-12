"""
ERZEUGEN UND BEWERTEN -- das eigene Modell durch dieselben Tore wie die Grossen.

Zwei Betriebsarten:

  1. Test-Split (Vorlagen, die das Modell nie gesehen hat):
       python workflow/erzeuge.py --checkpoints daten/workflow/ckpt \
           --test daten/workflow/test.jsonl --out bewertung/ergebnisse/eigenes-test.json
     Misst je Beispiel: Kurzschrift lesbar? (Tor 0) · Tor 1-3 (tore.py)
     · Typen-Ueberdeckung gegen die Referenz (Jaccard) · erfundene Typen.

  2. Die 15 Instruktionen aus Stufe 3 -- Ausgabe im Kandidaten-Format von
     bewertung/vergleich.py, damit danach exakt derselbe Vergleich inklusive
     Tor 4 (echter n8n-Import) laeuft wie bei den neun Frontier-Modellen:
       python workflow/erzeuge.py --checkpoints daten/workflow/ckpt \
           --instruktionen bewertung/instruktionen.jsonl \
           --kandidaten-out bewertung/eigenes_modell_antworten.jsonl
       python bewertung/vergleich.py --kandidaten bewertung/eigenes_modell_antworten.jsonl ...

Best-of-k gegen den Validator (`--k 8`): k Stichproben, die erste, die
Tor 1-3 besteht, zaehlt. Der Plan nennt es Gueltigkeit@1 gegen
Gueltigkeit@8 -- beides wird berichtet, nie nur die bessere Zahl.

Was hier bewusst NICHT passiert: keine Nachbesserung der Ausgabe. Eine
unlesbare Zeile wird nicht weggelassen, ein erfundener Typ nicht auf den
naechstaehnlichen echten korrigiert. Das waere Stufe 5 (eingeschraenkte
Dekodierung) und ist eine eigene Messung.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

WURZEL = Path(__file__).parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "kern"))
sys.path.insert(0, str(WURZEL / "bewertung"))
from modell import MiniGPT  # noqa: E402
from checkpoint import Checkpointer  # noqa: E402
from tore import pruefe_alle_tore, lade_echte_node_typen  # noqa: E402
from workflow import kurzschrift as ks  # noqa: E402
from workflow.tokenizer_workflow import WorkflowTokenizer  # noqa: E402


def lade_modell(ordner: Path, geraet: str) -> tuple[MiniGPT, dict]:
    konfig = json.loads((ordner / "modell_konfig.json").read_text(encoding="utf8"))
    modell = MiniGPT(konfig["vokabular"], dim=konfig["dim"], koepfe=konfig["koepfe"],
                     schichten=konfig["schichten"], block=konfig["block"]).to(geraet)
    schritt, _ = Checkpointer(ordner).lade_neuesten(modell, torch.optim.AdamW(modell.parameters()))
    if schritt == 0:
        raise SystemExit(f"kein Checkpoint in {ordner}")
    modell.eval()
    konfig["schritt"] = schritt
    return modell, konfig


@torch.no_grad()
def erzeuge_ids(modell: MiniGPT, prompt: list[int], eos_id: int, hoechstens: int,
                temperatur: float, top_k: int | None, geraet: str) -> list[int]:
    idx = torch.tensor([prompt], dtype=torch.long, device=geraet)
    for _ in range(hoechstens):
        logits, _ = modell(idx[:, -modell.block:])
        logits = logits[:, -1, :]
        if temperatur <= 0:
            naechstes = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperatur
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            naechstes = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, naechstes), dim=1)
        if naechstes.item() == eos_id:
            break
    return idx[0].tolist()


def bewerte_text(kurz: str, referenz: str | None, echte: set[str]) -> dict:
    e = {"tor0_kurzschrift": False, "tor1_json": False, "tor2_struktur": False,
         "tor3_verbindungen": False, "gueltig": False, "erfundene_typen": [],
         "typen_jaccard": None, "knoten": 0, "probleme": []}
    try:
        gerendert = ks.rendere(kurz)
    except ks.KurzschriftFehler as x:
        e["probleme"].append(f"Tor 0: {x}")
        return e
    e["tor0_kurzschrift"] = True
    e["knoten"] = len(gerendert["nodes"])
    typen = [n["type"] for n in gerendert["nodes"]]
    e["erfundene_typen"] = sorted({t for t in typen if t not in echte})
    r = pruefe_alle_tore(json.dumps(gerendert))
    for k in ("tor1_json", "tor2_struktur", "tor3_verbindungen"):
        e[k] = r[k]
    e["gueltig"] = r["alle_bestanden_ohne_import"]
    e["probleme"].extend(r["probleme"][:5])
    if referenz:
        ref = {n["type"] for n in ks.rendere(referenz)["nodes"]}
        m = set(typen)
        e["typen_jaccard"] = round(len(ref & m) / max(1, len(ref | m)), 3)
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="Vorgabe: tokenizer.pkl neben dem Datensatz (--test) bzw. daten/workflow/")
    p.add_argument("--test", type=Path, default=None, help="test.jsonl aus baue_datensatz.py")
    p.add_argument("--instruktionen", type=Path, default=None, help="bewertung/instruktionen.jsonl")
    p.add_argument("--kandidaten-out", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="Ergebnis-JSON")
    p.add_argument("--n", type=int, default=None, help="nur die ersten N Beispiele")
    p.add_argument("--k", type=int, default=1, help="Stichproben je Instruktion (Best-of-k)")
    p.add_argument("--temperatur", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--hoechstens-token", type=int, default=768)
    p.add_argument("--modell-name", default="autolm-workflow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zeige", type=int, default=3, help="so viele Beispiele ausdrucken")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    tok_pfad = args.tokenizer or ((args.test.parent if args.test else WURZEL / "daten" / "workflow") / "tokenizer.pkl")
    tok = WorkflowTokenizer.lade(tok_pfad)
    modell, konfig = lade_modell(args.checkpoints, geraet)
    echte = lade_echte_node_typen()
    print(f"Modell: {modell.anzahl_parameter():,} Parameter, Schritt {konfig['schritt']}, {geraet}")

    if args.test:
        faelle = [json.loads(z) for z in args.test.read_text(encoding="utf8").splitlines() if z.strip()]
        faelle = [{"id": f"t{f['quelle_id']}", "instruktion": f["instruktion"],
                   "referenz": f["kurzschrift"]} for f in faelle]
    elif args.instruktionen:
        faelle = [json.loads(z) for z in args.instruktionen.read_text(encoding="utf8").splitlines() if z.strip()]
        faelle = [{"id": f["id"], "instruktion": f["text"], "referenz": None} for f in faelle]
    else:
        raise SystemExit("--test oder --instruktionen angeben")
    if args.n:
        faelle = faelle[:args.n]

    einzel, kandidaten = [], []
    t0 = time.time()
    for i, fall in enumerate(faelle):
        prompt = tok.kodiere_prompt(fall["instruktion"])
        versuche = []
        for _ in range(args.k):
            ids = erzeuge_ids(modell, prompt, tok.eos_id, args.hoechstens_token,
                              args.temperatur, args.top_k, geraet)
            kurz = tok.dekodiere_antwort(ids)
            b = bewerte_text(kurz, fall["referenz"], echte)
            b["kurzschrift"] = kurz
            b["abgebrochen"] = tok.eos_id not in ids
            versuche.append(b)
            if b["gueltig"]:
                break
        bester = next((v for v in versuche if v["gueltig"]), versuche[0])
        einzel.append({"id": fall["id"], "instruktion": fall["instruktion"],
                       "versuche": len(versuche), "gueltig_at_1": versuche[0]["gueltig"],
                       "gueltig_at_k": bester["gueltig"], **bester})
        if args.kandidaten_out:
            try:
                antwort = json.dumps(ks.rendere(bester["kurzschrift"]), ensure_ascii=False)
            except ks.KurzschriftFehler:
                antwort = bester["kurzschrift"]  # unlesbar -> faellt bei Tor 1 durch, wie es soll
            kandidaten.append({"modell": args.modell_name, "id": fall["id"],
                               "instruktion": fall["instruktion"], "antwort_roh": antwort,
                               "kurzschrift": bester["kurzschrift"]})
        if i < args.zeige:
            print(f"\n--- {fall['id']}: {fall['instruktion'][:120]}")
            print(bester["kurzschrift"][:600])
            print("=>", "GUELTIG" if bester["gueltig"] else bester["probleme"][:2])

    n = len(einzel)
    def quote(schl): return round(sum(1 for e in einzel if e[schl]) / max(1, n), 3)
    jacc = [e["typen_jaccard"] for e in einzel if e["typen_jaccard"] is not None]
    erfunden = {}
    for e in einzel:
        for t in e["erfundene_typen"]:
            erfunden[t] = erfunden.get(t, 0) + 1
    zusammenfassung = {
        "modell": args.modell_name, "checkpoint_schritt": konfig["schritt"],
        "parameter": modell.anzahl_parameter(), "n": n, "k": args.k,
        "temperatur": args.temperatur, "top_k": args.top_k, "seed": args.seed,
        "tor0_kurzschrift": quote("tor0_kurzschrift"), "tor1_json": quote("tor1_json"),
        "tor2_struktur": quote("tor2_struktur"), "tor3_verbindungen": quote("tor3_verbindungen"),
        "gueltig_at_1": quote("gueltig_at_1"), "gueltig_at_k": quote("gueltig_at_k"),
        "abgebrochen": quote("abgebrochen"),
        "typen_jaccard_mittel": round(sum(jacc) / len(jacc), 3) if jacc else None,
        "erfundene_typen": dict(sorted(erfunden.items(), key=lambda x: -x[1])),
        "dauer_s": round(time.time() - t0),
    }
    print("\n" + json.dumps(zusammenfassung, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"zusammenfassung": zusammenfassung, "einzel": einzel},
                                       indent=1, ensure_ascii=False), encoding="utf8")
        print(f"-> {args.out}")
    if args.kandidaten_out:
        with open(args.kandidaten_out, "w", encoding="utf8") as f:
            for k in kandidaten:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        print(f"-> {args.kandidaten_out} ({len(kandidaten)} Kandidaten fuer bewertung/vergleich.py)")


if __name__ == "__main__":
    main()
