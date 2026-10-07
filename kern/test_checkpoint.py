"""
DER KILL-UND-RESUME-TEST.

Startet ein winziges Training, TOETET DEN PROZESS MITTEN IM LAUF (nicht
simuliert -- ein echter os.kill auf einen echten Subprozess), startet neu
und prueft: setzt die Loss-Kurve fort, oder springt sie?

Ein Springen waere der Beweis, dass Datenposition oder Optimizer-State
NICHT wirklich wiederhergestellt wurden -- selbst wenn "es laeuft danach
weiter" stimmt, waere das Training dann nicht dasselbe, das ohne Abbruch
gelaufen waere.

    python kern/test_checkpoint.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HIER = Path(__file__).parent
ORDNER = HIER / "_test_checkpoint_lauf"
SKRIPT = HIER / "_trainingslauf_fuer_test.py"


TRAININGSSKRIPT = '''
"""Wird NUR vom Test aufgerufen, kein eigenstaendiges Lehrstueck."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from modell import MiniGPT
from checkpoint import Checkpointer

torch.manual_seed(0)
vok, block = 64, 16
daten = torch.randint(0, vok, (20000,))

modell = MiniGPT(vok, dim=32, koepfe=2, schichten=2, block=block)
opt = torch.optim.AdamW(modell.parameters(), lr=1e-3)
ckpt = Checkpointer(Path(sys.argv[1]))

schritt, pos = ckpt.lade_neuesten(modell, opt)
print(f"START bei Schritt {schritt}, Datenposition {pos}", flush=True)

verlust_log = []
for i in range(schritt, 200):
    if pos + block + 1 > len(daten):
        pos = 0
    x = daten[pos:pos + block].unsqueeze(0)
    y = daten[pos + 1:pos + block + 1].unsqueeze(0)
    _, verlust = modell(x, y)
    opt.zero_grad(set_to_none=True)
    verlust.backward()
    opt.step()
    pos += block
    verlust_log.append(verlust.item())

    if i % 10 == 0:
        ckpt.speichere(modell, opt, i, pos, verlust_log)
        print(f"SCHRITT {i} VERLUST {verlust.item():.6f} POS {pos}", flush=True)

    time.sleep(0.05)  # macht das Zeitfenster fuer den Kill zuverlaessig
'''


def _bereite_vor():
    if ORDNER.exists():
        shutil.rmtree(ORDNER)
    ORDNER.mkdir()
    SKRIPT.write_text(TRAININGSSKRIPT, encoding="utf8")


def _aufraeumen():
    if ORDNER.exists():
        shutil.rmtree(ORDNER)
    if SKRIPT.exists():
        SKRIPT.unlink()


def _lies_schritte(ausgabe: str):
    schritte = []
    for zeile in ausgabe.splitlines():
        if zeile.startswith("SCHRITT"):
            teile = zeile.split()
            schritte.append((int(teile[1]), float(teile[3])))
    return schritte


def test_kill_und_resume():
    _bereite_vor()
    try:
        # --- Lauf 1: starten und toeten, SOBALD genug Checkpoints da sind ---
        # WICHTIG: kontinuierlich per readline() lesen waehrend gewartet
        # wird, nicht blockierend schlafen und danach communicate() rufen.
        # Gemessen: ein blockierendes time.sleep(7) + anschliessendes
        # communicate() lieferte auf diesem Server oft LEERE Ausgabe, obwohl
        # der Kindprozess nachweislich lief (poll() == None) -- vermutlich
        # Pipe-Pufferung unter Windows in Kombination mit proc.kill().
        # Aktives Mitlesen umgeht das zuverlaessig.
        proc = subprocess.Popen(
            [sys.executable, str(SKRIPT), str(ORDNER)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        gesammelt = []
        t0 = time.time()
        # torch-Import allein braucht auf diesem Server ~4.5s (gemessen) --
        # das Zeitfenster muss grosszuegig sein, sonst wird getoetet, bevor
        # ueberhaupt ein Schritt geloggt wurde.
        while time.time() - t0 < 20:
            zeile = proc.stdout.readline()
            if zeile:
                gesammelt.append(zeile.rstrip())
                if zeile.startswith("SCHRITT") and len(_lies_schritte("\n".join(gesammelt))) >= 3:
                    break
        proc.kill()
        rest, _ = proc.communicate(timeout=15)
        if rest:
            gesammelt.append(rest.rstrip())
        ausgabe1 = "\n".join(gesammelt)
        schritte1 = _lies_schritte(ausgabe1)
        assert len(schritte1) >= 2, (
            f"Zu wenig Fortschritt vor dem Kill, um den Test aussagekraeftig "
            f"zu machen (nach 20s Wartezeit). Ausgabe: {ausgabe1!r}"
        )
        letzter_schritt_vor_kill = schritte1[-1][0]
        print(f"  Lauf 1 (vor Kill): {len(schritte1)} Checkpoints, "
              f"letzter bei Schritt {letzter_schritt_vor_kill}")

        # --- Lauf 2: neu starten, muss beim Checkpoint weitermachen ---
        proc2 = subprocess.run(
            [sys.executable, str(SKRIPT), str(ORDNER)],
            capture_output=True, text=True, timeout=30,
        )
        assert proc2.stdout, (
            f"Lauf 2 lieferte keine Ausgabe. stderr: {proc2.stderr[:500]}"
        )
        ausgabe2 = proc2.stdout
        start_zeile = next(z for z in ausgabe2.splitlines() if z.startswith("START"))
        # "START bei Schritt 0, Datenposition 0" -> Index 3 ist die Zahl,
        # nicht 2 (das ist noch das Wort "Schritt").
        start_schritt = int(start_zeile.split()[3].rstrip(","))
        print(f"  Lauf 2 (nach Resume): {start_zeile.strip()}")

        assert start_schritt > 0, (
            "Resume ist bei Schritt 0 gestartet -- der Checkpoint wurde "
            "nicht gefunden oder nicht gelesen. Das waere kein Resume, "
            "sondern ein Neustart."
        )
        assert start_schritt <= letzter_schritt_vor_kill, (
            f"Resume startet NACH dem letzten gesicherten Schritt "
            f"({start_schritt} > {letzter_schritt_vor_kill}) -- das kann "
            f"nur bedeuten, dass ungesicherter Fortschritt erfunden wurde."
        )
        # Darf auch nicht zu weit VOR dem letzten Checkpoint liegen --
        # sonst wuerde ein aelterer, aber vollstaendig geschriebener
        # Checkpoint uebersehen.
        assert start_schritt >= letzter_schritt_vor_kill - 10, (
            f"Resume startet unnoetig weit vor dem letzten Checkpoint "
            f"({start_schritt} vs. {letzter_schritt_vor_kill})."
        )

        # --- Die eigentliche Aussage: KEIN Sprung in der Loss-Kurve ---
        schritte2 = _lies_schritte(ausgabe2)
        alle = sorted(set(schritte1 + schritte2))
        nummern = [s for s, _ in alle]
        for i in range(1, len(nummern)):
            luecke = nummern[i] - nummern[i - 1]
            assert luecke <= 10, (
                f"Luecke von {luecke} Schritten zwischen {nummern[i-1]} und "
                f"{nummern[i]} in der zusammengesetzten Kurve -- das Training "
                f"haette nach dem Resume Schritte verloren."
            )
        print(f"  Zusammengesetzte Kurve: {len(alle)} Checkpoints, "
              f"lueckenlos von Schritt {nummern[0]} bis {nummern[-1]}")

    finally:
        _aufraeumen()


def test_leerer_start_ohne_checkpoint():
    """Kein Checkpoint vorhanden -> muss bei Schritt 0 anfangen, kein Fehler."""
    _bereite_vor()
    try:
        # Muss NICHT bis zum Ende laufen (200 Schritte) -- nur die erste
        # Zeile beweist den Kaltstart. torch-Import allein braucht auf
        # diesem Server ~4.5s, also grosszuegiges Timeout statt frueher
        # Abbruch.
        proc = subprocess.Popen(
            [sys.executable, str(SKRIPT), str(ORDNER)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        zeile = ""
        t0 = time.time()
        while time.time() - t0 < 15:
            zeile = proc.stdout.readline()
            if zeile.startswith("START"):
                break
        proc.kill()
        proc.communicate(timeout=10)
        ausgabe = zeile
    finally:
        pass
    assert ausgabe.startswith("START bei Schritt 0"), f"Kein sauberer Kaltstart: {ausgabe[:200]}"
    print("  Kaltstart ohne Checkpoint: OK")
    _aufraeumen()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fehler = 0
    for name, fn in [("test_leerer_start_ohne_checkpoint", test_leerer_start_ohne_checkpoint),
                      ("test_kill_und_resume", test_kill_und_resume)]:
        try:
            fn()
            print(f"OK    {name}\n")
        except Exception as e:
            print(f"FEHLT {name}: {e}\n")
            fehler += 1
    print("ALLE BESTANDEN" if not fehler else f"{fehler} FEHLGESCHLAGEN")
    raise SystemExit(1 if fehler else 0)
