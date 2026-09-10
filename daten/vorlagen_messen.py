"""
STUFE 4, SCHRITT 0 -- MESSEN, BEVOR GEBAUT WIRD.

Die offene Designfrage aus dem Plan lautete: "viele echte n8n-Vorlagen
benutzen Community-/Langchain-Node-Typen, die NICHT in unserer Liste der
439 n8n-nodes-base-Typen stehen (bewertung/echte_node_typen.json) -- wie
gross ist das Problem wirklich?" Bis hierhin war das eine Vermutung aus
einer Stichprobe von Metadaten. Dieses Skript beantwortet es gegen den
VOLLEN Bestand: die Template-Datenbank des n8n-mcp-Pakets (2.352 Vorlagen
mit vollstaendigem workflow_json, gzip+base64 gespeichert).

Warum das VOR der Mutations-Pipeline steht: der Validator (bewertung/tore.py,
Tor 2) lehnt jeden Node-Typ ab, der nicht in der 439er-Liste steht. Wenn
die Haelfte der Vorlagen Langchain-Nodes traegt, verwirft das Orakel die
Haelfte des Trainingsmaterials -- oder, schlimmer, wir trainieren ein
Modell auf Typen, die der Validator hinterher fuer "halluziniert" haelt.
Das waere dieselbe Fehlerklasse wie ein Split, der die Antwort ins Testset
leckt: der Massstab und die Daten passen nicht zueinander.

Nur lesend. Schreibt einen Bericht nach bewertung/ergebnisse/.

    python daten/vorlagen_messen.py
    python daten/vorlagen_messen.py --db <pfad-zu-nodes.db>
"""

import argparse
import base64
import collections
import gzip
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

WURZEL = Path(__file__).parent.parent
TYPEN_DATEI = WURZEL / "bewertung" / "echte_node_typen.json"
ERGEBNIS_ORDNER = WURZEL / "bewertung" / "ergebnisse"

# Standardpfad: der npx-Cache, in dem `npx -y n8n-mcp` (der MCP-Server aus
# .mcp.json) sein Paket ablegt. Maschinenspezifisch -- deshalb ueberschreibbar.
DB_STANDARD = Path.home() / "AppData/Local/npm-cache/_npx/b6a381d62ce0fe56/node_modules/n8n-mcp/data/nodes.db"


def lade_bekannte_typen() -> set[str]:
    d = json.loads(TYPEN_DATEI.read_text(encoding="utf8"))
    if isinstance(d, dict):
        for k in ("typen", "types", "node_typen", "node_types"):
            if k in d:
                return set(d[k])
        # sonst: erster Listenwert
        for v in d.values():
            if isinstance(v, list):
                return set(v)
        raise SystemExit(f"Unbekanntes Format in {TYPEN_DATEI}")
    return set(d)


def entpacke(workflow_json_compressed: str) -> dict | None:
    try:
        roh = gzip.decompress(base64.b64decode(workflow_json_compressed))
        return json.loads(roh)
    except Exception:
        return None


def paket_von(typ: str) -> str:
    # "@n8n/n8n-nodes-langchain.agent" -> "@n8n/n8n-nodes-langchain"
    # "n8n-nodes-base.set"             -> "n8n-nodes-base"
    # "n8n-nodes-firecrawl.firecrawl"  -> "n8n-nodes-firecrawl"
    return typ.rsplit(".", 1)[0] if "." in typ else typ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DB_STANDARD)
    args = p.parse_args()
    if not args.db.exists():
        sys.exit(f"Template-DB nicht gefunden: {args.db}")

    bekannt = lade_bekannte_typen()
    print(f"Bekannte n8n-nodes-base-Typen: {len(bekannt)}")

    con = sqlite3.connect(args.db)
    zeilen = con.execute(
        "select id, nodes_used, workflow_json_compressed from templates"
    ).fetchall()
    print(f"Vorlagen in der DB: {len(zeilen)}\n")

    nur_base = 0             # alle Typen in der 439er-Liste
    base_plus_langchain = 0  # nur base + @n8n/n8n-nodes-langchain
    mit_community = 0        # irgendein anderes Paket
    ohne_typen = 0
    json_kaputt = 0
    fremde_pakete = collections.Counter()
    fremde_typen = collections.Counter()
    unbekannte_base = collections.Counter()  # n8n-nodes-base.X, aber X nicht in Liste
    zeichen_gesamt = 0
    zeichen_nur_base = 0
    knoten_gesamt = 0
    knoten_nur_base = []

    for tid, nodes_used, komp in zeilen:
        try:
            typen = json.loads(nodes_used) if nodes_used else []
        except Exception:
            typen = []
        if not typen:
            ohne_typen += 1
            continue

        wf = entpacke(komp) if komp else None
        if wf is None:
            json_kaputt += 1
        else:
            n_zeichen = len(json.dumps(wf, ensure_ascii=False))
            zeichen_gesamt += n_zeichen
            knoten_gesamt += len(wf.get("nodes", []))

        eindeutig = set(typen)
        fremd = {t for t in eindeutig if t not in bekannt}
        pakete = {paket_von(t) for t in fremd}

        if not fremd:
            nur_base += 1
            if wf is not None:
                zeichen_nur_base += n_zeichen
                knoten_nur_base.append(len(wf.get("nodes", [])))
        elif pakete <= {"@n8n/n8n-nodes-langchain"}:
            base_plus_langchain += 1
        else:
            mit_community += 1

        for t in fremd:
            fremde_typen[t] += 1
            pk = paket_von(t)
            fremde_pakete[pk] += 1
            if pk == "n8n-nodes-base":
                unbekannte_base[t] += 1

    n = len(zeilen) - ohne_typen
    def pct(x): return f"{100 * x / n:.1f} %" if n else "-"

    print("=== Verteilung (je Vorlage, eindeutige Typen) ===")
    print(f"  alle Typen in unserer Liste (Tor 2 wuerde sie annehmen): {nur_base:5d}  {pct(nur_base)}")
    print(f"  nur Langchain fehlt (@n8n/n8n-nodes-langchain):          {base_plus_langchain:5d}  {pct(base_plus_langchain)}")
    print(f"  mit Typen aus sonstigen (Community-)Paketen:              {mit_community:5d}  {pct(mit_community)}")
    # Hinweis: seit die Liste (extrahiere_node_typen.py) Langchain und die
    # *Tool-Varianten enthaelt, ist die mittlere Zeile per Konstruktion 0 --
    # sie bleibt drin, damit ein Rueckfall auf eine alte Liste sofort auffaellt.
    print(f"  ohne Typen-Liste (uebersprungen):           {ohne_typen:5d}")
    print(f"  workflow_json nicht entpackbar:             {json_kaputt:5d}")

    print("\n=== Fremde Pakete (in wie vielen Vorlagen) ===")
    for pk, c in fremde_pakete.most_common(15):
        print(f"  {c:5d}  {pk}")

    print("\n=== Haeufigste fremde Typen ===")
    for t, c in fremde_typen.most_common(20):
        print(f"  {c:5d}  {t}")

    if unbekannte_base:
        print("\n!!! n8n-nodes-base-Typen, die NICHT in unserer 439er-Liste stehen "
              "(Liste veraltet oder Vorlage nutzt neuere/aeltere Version):")
        for t, c in unbekannte_base.most_common(20):
            print(f"  {c:5d}  {t}")

    print("\n=== Groesse (fuer die Token-Schaetzung aus dem Plan) ===")
    print(f"  Zeichen gesamt (alle Vorlagen):    {zeichen_gesamt:,}  "
          f"(~{zeichen_gesamt // 4:,} Token bei ~4 Zeichen/Token, GROB)")
    print(f"  Zeichen nur-base-Vorlagen:         {zeichen_nur_base:,}  "
          f"(~{zeichen_nur_base // 4:,} Token)")
    if knoten_nur_base:
        s = sorted(knoten_nur_base)
        print(f"  Knoten je nur-base-Vorlage: Median {s[len(s)//2]}, "
              f"p90 {s[int(len(s)*0.9)]}, max {s[-1]}")

    ERGEBNIS_ORDNER.mkdir(parents=True, exist_ok=True)
    aus = ERGEBNIS_ORDNER / f"vorlagen-messung-{date.today().isoformat()}.json"
    aus.write_text(json.dumps({
        "datum": date.today().isoformat(),
        "db": str(args.db),
        "bekannte_typen": len(bekannt),
        "vorlagen_gesamt": len(zeilen),
        "ohne_typen": ohne_typen,
        "json_kaputt": json_kaputt,
        "nur_base": nur_base,
        "base_plus_langchain": base_plus_langchain,
        "mit_community": mit_community,
        "fremde_pakete": fremde_pakete.most_common(),
        "fremde_typen_top50": fremde_typen.most_common(50),
        "unbekannte_base_typen": unbekannte_base.most_common(),
        "zeichen_gesamt": zeichen_gesamt,
        "zeichen_nur_base": zeichen_nur_base,
        "knoten_gesamt": knoten_gesamt,
    }, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nBericht: {aus}")


if __name__ == "__main__":
    main()
