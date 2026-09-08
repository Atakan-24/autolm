"""
TOR 4 -- LAESST SICH DER WORKFLOW WIRKLICH IN N8N IMPORTIEREN?

Die einzige Pruefung in diesem Projekt, die ECHTES n8n aufruft statt einer
eigenen Nachbildung -- ueber die installierte n8n-CLI (Version 2.25.6,
lokal vorhanden), gegen eine WEGWERF-SQLITE-Datenbank.

ZWEI FUNDE BEIM BAUEN, BEIDE WICHTIG FUER DIE EINORDNUNG DES ERGEBNISSES:

1. DIE ERSTE MIGRATION IST TEUER, JEDE WEITERE NICHT. Eine frische
   N8N_USER_FOLDER-Datenbank durchlaeuft beim ersten Aufruf ueber 150
   Migrationen (mehrere Minuten). Gegen eine BEREITS migrierte Datenbank
   dauert derselbe Import nur noch Sekunden. Deshalb wird die Wegwerf-DB
   EINMAL angelegt und danach fuer alle Kandidaten wiederverwendet -- eine
   frische DB pro Workflow waere sonst bei 200 Kandidaten unbezahlbar.

2. `n8n import:workflow` PRUEFT NICHT, OB DIE VERWENDETEN NODE-TYPEN
   WIRKLICH EXISTIEREN. Ein Workflow mit dem erfundenen Typ
   "n8n-nodes-base.gibtsnicht" wurde beim Test anstandslos importiert
   ("Successfully imported"). Das heisst: TOR 4 ALLEIN WUERDE HALLUZINIERTE
   NODE-TYPEN NICHT ERKENNEN -- genau die Faelle, in denen grosse Modelle
   laut der Projekthypothese am haeufigsten scheitern. Tor 2
   (bewertung/tore.py, Pruefung gegen die echte, extrahierte Liste von 439
   Node-Typen) ist deshalb NICHT redundant zu Tor 4, sondern faengt genau
   das, was Tor 4 durchlaesst. Beide Tore zusammen ergeben erst ein
   verlaessliches Bild.

BEWUSST NICHT GEBAUT: Tor 5 (`n8n execute`, echte Ausfuehrung). Zwei
Gruende, beide echte Umgebungsgrenzen und keine Bequemlichkeit:
  - `n8n execute` bindet zwingend den Task-Broker-Port 5679, und auf diesem
    Server laeuft dort bereits die PRODUKTIVE n8n-Instanz (siehe CLAUDE.md:
    "Laufende Produktivprozesse nicht anfassen"). Ein anderer Port laesst
    sich per N8N_RUNNERS_BROKER_PORT setzen, ohne die Produktivinstanz zu
    beruehren -- das allein loest das Problem aber nicht.
  - `n8n execute --id` verlangt einen "Execute Workflow Trigger"-Node als
    Startpunkt, den generierte Workflows in aller Regel nicht haben
    (sie haben einen Webhook-/Manual-/Schedule-Trigger, wie ein echter
    Automations-Workflow). Tor 5 bliebe damit auf eine kuenstliche
    Teilmenge beschraenkt -- im Plan als "optional, nur die Teilmenge
    nebenwirkungsfreier Workflows" ohnehin schon so vorgesehen.

    python bewertung/importtest.py --init                    -- einmalig: DB migrieren
    python bewertung/importtest.py --ordner <mit .json-Dateien>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

HIER = Path(__file__).parent
DB_ORDNER = HIER / "_wegwerf_n8n_db"
# n8n legt seine Datenbank NICHT direkt unter N8N_USER_FOLDER an, sondern
# in einem verschachtelten ".n8n"-Unterordner -- durch Ausprobieren
# gefunden, nicht dokumentiert vorausgesetzt.
DB_DATEI = DB_ORDNER / ".n8n" / "database.sqlite"

# Unter Windows loest Python subprocess.run(["n8n", ...]) NICHT auf --
# "n8n" ist dort ein .cmd-Shim (npm-Wrapper), keine direkt ausfuehrbare
# Datei, und CreateProcess (das Windows-API darunter) findet sie ohne
# Shell-Interpretation nicht. In einer bash-Shell funktioniert "n8n"
# anstandslos, weil bash selbst die PATH-Aufloesung uebernimmt. Deshalb
# hier der volle Pfad zur .cmd-Datei statt des blossen Befehlsnamens.
N8N_BEFEHL = shutil.which("n8n.cmd") or shutil.which("n8n") or "n8n"


def _umgebung():
    env = dict(os.environ)
    env["N8N_USER_FOLDER"] = str(DB_ORDNER)
    # Verhindert den Port-Konflikt mit der PRODUKTIVEN n8n-Instanz auf
    # diesem Server (Port 5679 ist dort belegt) -- ohne diese Zeile
    # scheitert jeder Aufruf mit "port already in use", auch wenn die
    # Wegwerf-DB voellig getrennt ist.
    env["N8N_RUNNERS_BROKER_PORT"] = "15679"
    return env


def init_db(zeige_ausgabe: bool = True):
    """
    Legt die Wegwerf-Datenbank an und laesst alle Migrationen EINMAL
    durchlaufen -- indem ein einzelner Test-Workflow importiert wird.
    n8n hat keinen expliziten "nur migrieren"-Befehl; ein echter Import
    ist der zuverlaessigste Weg, dieselbe Codepfad-Initialisierung
    auszuloesen, die auch fuer echte Kandidaten laeuft.
    """
    if DB_ORDNER.exists():
        shutil.rmtree(DB_ORDNER)
    DB_ORDNER.mkdir(parents=True)

    test_datei = DB_ORDNER / "_init_test.json"
    test_datei.write_text(json.dumps({
        "id": "init-test",
        "name": "Init",
        "nodes": [{
            "id": str(uuid.uuid4()), "name": "Start",
            "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1,
            "position": [0, 0], "parameters": {},
        }],
        "connections": {},
    }), encoding="utf8")

    ergebnis = subprocess.run(
        [N8N_BEFEHL, "import:workflow", f"--input={test_datei}"],
        capture_output=True, text=True, env=_umgebung(), timeout=600,
    )
    test_datei.unlink()
    if zeige_ausgabe:
        letzte_zeilen = "\n".join(ergebnis.stdout.splitlines()[-5:])
        print(f"DB initialisiert. Letzte Ausgabe:\n{letzte_zeilen}")
    if ergebnis.returncode != 0:
        raise RuntimeError(f"Init fehlgeschlagen:\n{ergebnis.stdout}\n{ergebnis.stderr}")


def _leere_workflow_tabelle():
    """
    Setzt nur die Workflow-Tabelle zurueck, NICHT das ganze Schema --
    fuer eine saubere Zaehlung "wie viele von DIESEM Stapel wurden
    importiert" bei mehreren Laeufen hintereinander, ohne die teure
    Migration erneut durchlaufen zu muessen.
    """
    import sqlite3
    db_datei = DB_DATEI
    if not db_datei.exists():
        return
    con = sqlite3.connect(db_datei)
    con.execute("DELETE FROM workflow_entity")
    con.commit()
    con.close()


def teste_ordner(ordner: Path, zuruecksetzen: bool = True) -> dict:
    """
    Importiert ALLE *.json-Dateien aus `ordner` in einem einzigen
    n8n-Prozessaufruf (--separate) -- das amortisiert den Prozessstart
    ueber den ganzen Stapel, statt ihn pro Datei zu bezahlen.

    Gibt zurueck, wie viele Dateien tatsaechlich importiert wurden
    (`importiert_anzahl` aus n8n's eigener Meldung) UND -- weil n8n keine
    Datei-fuer-Datei-Erfolgsmeldung ausgibt -- welche IDs danach in der DB
    stehen.

    VORAUSSETZUNG, GEMESSEN: `ordner` darf NUR Dateien enthalten, die
    schon Tor 1 (gueltiges JSON) bestanden haben. EIN EINZIGES kaputtes
    JSON in einem --separate-Stapel bricht den KOMPLETTEN Import ab --
    keiner der anderen Kandidaten wird importiert, auch die gueltigen
    nicht ("An error occurred while importing workflows", 0 importiert).
    Das ist kein Bug hier, sondern der Grund, warum die Tore in Reihenfolge
    laufen: vergleich.py schickt nur Kandidaten zu Tor 4, die Tor 1
    bereits bestanden haben.
    """
    if not DB_ORDNER.exists():
        raise RuntimeError("Wegwerf-DB fehlt. Erst init_db() aufrufen.")
    if zuruecksetzen:
        _leere_workflow_tabelle()

    dateien = sorted(ordner.glob("*.json"))
    if not dateien:
        return {"dateien": 0, "importiert_laut_n8n": 0, "gefundene_ids": [], "fehlerausgabe": ""}

    ergebnis = subprocess.run(
        [N8N_BEFEHL, "import:workflow", "--separate", f"--input={ordner}"],
        capture_output=True, text=True, env=_umgebung(), timeout=600,
    )

    importiert = 0
    for zeile in ergebnis.stdout.splitlines():
        if "Successfully imported" in zeile:
            teile = zeile.split()
            for t in teile:
                if t.isdigit():
                    importiert = int(t)
                    break

    import sqlite3
    con = sqlite3.connect(DB_DATEI)
    gefunden = {r[0] for r in con.execute("SELECT id FROM workflow_entity")}
    con.close()

    return {
        "dateien": len(dateien),
        "importiert_laut_n8n": importiert,
        "gefundene_ids": sorted(gefunden),
        "fehlerausgabe": ergebnis.stderr[-500:] if ergebnis.returncode != 0 else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true")
    p.add_argument("--ordner", type=Path)
    args = p.parse_args()

    if args.init:
        init_db()
        return

    if args.ordner:
        if not DB_ORDNER.exists():
            print("Wegwerf-DB fehlt -- initialisiere zuerst (einmalig, dauert laenger) ...")
            init_db(zeige_ausgabe=False)
        ergebnis = teste_ordner(args.ordner)
        print(json.dumps(ergebnis, indent=2, ensure_ascii=False))
        return

    p.print_help()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
