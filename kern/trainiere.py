"""
STUFE 2b -- DER ECHTE PRETRAINING-LOOP.

Bindet zusammen, was in den vorigen Stufen einzeln gebaut und geprueft
wurde:
  modell.py       das MiniGPT (Stufe 0)
  checkpoint.py   absturzsicheres Speichern/Laden (Stufe 2a, mit echtem
                  Kill-und-Resume-Beweis in test_checkpoint.py)
  die Memmap      aus daten/vortokenisiere.py (Stufe 1c)

Laeuft unveraendert auf CPU (langsam, zum lokalen Testen) und GPU (Colab).
Kein Code-Unterschied zwischen beiden -- `torch.device` waehlt automatisch.

    python kern/trainiere.py --daten daten/tinystories_train.bin \
        --meta daten/tinystories_meta.json --schritte 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from modell import MiniGPT
from checkpoint import Checkpointer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lade_daten(pfad: Path) -> np.memmap:
    """
    memmap statt np.fromfile: die Datei wird NICHT komplett in den
    Arbeitsspeicher geladen. Bei 340M Token (uint16) waeren das 680 MB --
    machbar, aber bei der vollen 470M-Token-TinyStories-Menge oder
    groesseren Korpora waere ein vollstaendiges Laden unnoetig und auf
    manchen Colab-Instanzen (12 GB RAM) riskant.
    """
    return np.memmap(pfad, dtype=np.uint16, mode="r")


def hole_stapel(daten: np.memmap, position: int, block: int, batch: int,
                 geraet: str, maske: np.memmap | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Liest `batch` aufeinanderfolgende Sequenzen ab `position`, rollt am
    Ende der Datei um. Gibt die NEUE Position zurueck -- das ist exakt
    der Wert, der in den Checkpoint wandert (siehe checkpoint.py-
    Docstring: ohne Datenposition liefe ein Resume im Kreis).
    """
    n = len(daten)
    xs, ys = [], []
    for _ in range(batch):
        if position + block + 1 > n:
            position = 0
        stueck = daten[position:position + block + 1].astype(np.int64)
        xs.append(stueck[:-1])
        ziel = stueck[1:]
        if maske is not None:
            # -100 = ignore_index von cross_entropy: Instruktionstoken tragen
            # nichts zum Verlust bei, das Modell lernt nur die Kurzschrift.
            m = maske[position + 1:position + block + 1]
            ziel = np.where(m == 1, ziel, -100)
        ys.append(ziel)
        position += block
    x = torch.from_numpy(np.stack(xs)).to(geraet)
    y = torch.from_numpy(np.stack(ys)).to(geraet)
    return x, y, position


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daten", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--checkpoint-ordner", type=Path, default=Path("checkpoints"))
    p.add_argument("--gross", action="store_true",
                    help="17M-Konfiguration aus dem Plan (dim384/6 Schichten). "
                         "Ohne diesen Schalter eine kleine Testkonfiguration.")
    p.add_argument("--schritte", type=int, default=None,
                    help="Vorgabe: aus --token-budget errechnet")
    p.add_argument("--token-budget", type=int, default=340_000_000,
                    help="Chinchilla-optimal fuer die 17M-Konfiguration (siehe Plan)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-alle", type=int, default=500)
    p.add_argument("--log-alle", type=int, default=50)
    # Stufe 4: freie Modellgroesse und ein Validierungs-Split. Ohne diese
    # Schalter verhaelt sich das Skript exakt wie vorher (Stufe 2).
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--schichten", type=int, default=None)
    p.add_argument("--koepfe", type=int, default=None)
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--val-daten", type=Path, default=None,
                    help="zweiter uint16-Strom; Verlust darauf bei jedem Log-Schritt")
    p.add_argument("--val-stapel", type=int, default=8)
    p.add_argument("--maske", action="store_true",
                    help="Stufe 4: <daten>.maske.bin (uint8) lesen; nur Positionen mit 1 "
                         "zaehlen zum Verlust (Kurzschrift, nicht Instruktion)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    meta = json.loads(args.meta.read_text(encoding="utf8"))
    vokabular_groesse = meta.get("vokabular_groesse")
    if vokabular_groesse is None:
        raise SystemExit(f"{args.meta} hat kein 'vokabular_groesse' -- veraltete Metadatei?")

    if args.gross:
        dim, koepfe, schichten, block, batch = 384, 6, 6, 256, 32
    else:
        dim, koepfe, schichten, block, batch = 128, 4, 3, 96, 16
    dim = args.dim or dim
    koepfe = args.koepfe or koepfe
    schichten = args.schichten or schichten
    block = args.block or block
    batch = args.batch or batch

    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Geraet: {geraet}")
    print(f"Aufbau: dim={dim}, {schichten} Schichten, {koepfe} Koepfe, "
          f"block={block}, batch={batch}, Vokabular={vokabular_groesse}")

    daten = lade_daten(args.daten)
    print(f"Daten: {len(daten):,} Token aus {args.daten}")
    maske = None
    if args.maske:
        maske = np.memmap(args.daten.with_suffix(".maske.bin"), dtype=np.uint8, mode="r")
        if len(maske) != len(daten):
            raise SystemExit(f"Maske ({len(maske):,}) passt nicht zu den Daten ({len(daten):,})")
        print(f"Maske: {int(maske.sum()):,} von {len(maske):,} Token zaehlen zum Verlust")

    schritte = args.schritte or max(1, args.token_budget // (batch * block))
    print(f"Ziel: {schritte:,} Schritte (~{schritte * batch * block:,} Token)")

    modell = MiniGPT(vokabular_groesse, dim=dim, koepfe=koepfe,
                     schichten=schichten, block=block).to(geraet)
    print(f"Parameter: {modell.anzahl_parameter():,}")

    optimierer = torch.optim.AdamW(modell.parameters(), lr=args.lr)
    ckpt = Checkpointer(args.checkpoint_ordner)
    # Die Konfiguration neben die Checkpoints -- ein Checkpoint ohne sie ist
    # nicht ladbar (dieselbe Fehlerklasse wie der fehlende tokenizer.pkl).
    (args.checkpoint_ordner / "modell_konfig.json").write_text(json.dumps({
        "vokabular": vokabular_groesse, "dim": dim, "koepfe": koepfe,
        "schichten": schichten, "block": block, "batch": batch, "lr": args.lr,
        "daten": str(args.daten), "seed": args.seed}, indent=2), encoding="utf8")
    val_daten = lade_daten(args.val_daten) if args.val_daten else None
    val_maske = (np.memmap(args.val_daten.with_suffix(".maske.bin"), dtype=np.uint8, mode="r")
                 if (args.val_daten and args.maske) else None)
    verlauf = open(args.checkpoint_ordner / "verlauf.jsonl", "a", encoding="utf8")

    @torch.no_grad()
    def val_verlust() -> float | None:
        if val_daten is None:
            return None
        modell.eval()
        pos, summe = 0, 0.0
        for _ in range(args.val_stapel):
            vx, vy, pos = hole_stapel(val_daten, pos, block, batch, geraet, val_maske)
            summe += modell(vx, vy)[1].item()
        modell.train()
        return summe / args.val_stapel

    start_schritt, position = ckpt.lade_neuesten(modell, optimierer)
    if start_schritt > 0:
        print(f"Resume bei Schritt {start_schritt}, Datenposition {position:,}")
    else:
        print("Kein Checkpoint gefunden -- Start bei Schritt 0")

    modell.train()
    t0 = time.time()
    verlust_log = []
    for schritt in range(start_schritt, schritte):
        x, y, position = hole_stapel(daten, position, block, batch, geraet, maske)
        _, verlust = modell(x, y)
        optimierer.zero_grad(set_to_none=True)
        verlust.backward()
        optimierer.step()
        verlust_log.append(verlust.item())

        if schritt % args.log_alle == 0:
            dt = time.time() - t0
            vv = val_verlust()
            print(f"  Schritt {schritt:>7,} | Verlust {verlust.item():.4f} | "
                  + (f"Val {vv:.4f} | " if vv is not None else "")
                  + f"{dt / max(1, schritt - start_schritt + 1):.3f}s/Schritt", flush=True)
            verlauf.write(json.dumps({"schritt": schritt, "train": round(verlust.item(), 4),
                                      "val": None if vv is None else round(vv, 4),
                                      "zeit_s": round(dt)}) + "\n")
            verlauf.flush()

        if schritt % args.checkpoint_alle == 0 and schritt > start_schritt:
            ckpt.speichere(modell, optimierer, schritt, position, verlust_log)
            verlust_log = []

    ckpt.speichere(modell, optimierer, schritte, position, verlust_log or [0.0])
    print(f"\nFertig. {schritte:,} Schritte in {time.time() - t0:.0f}s.")

    # Probe-Text, damit man auch ohne separates Skript sofort sieht, ob
    # etwas Sinnvolles gelernt wurde.
    modell.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=geraet)
    erzeugt = modell.erzeuge(start, 100, temperatur=0.8)
    print(f"\nToken-IDs der ersten erzeugten Sequenz (Dekodieren braucht den Tokenizer):")
    print(erzeugt[0].tolist())


if __name__ == "__main__":
    main()
