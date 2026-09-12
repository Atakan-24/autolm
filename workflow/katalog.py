"""
NODE-KATALOG -- Anzeigename, Kategorie und Ausloeser-Flag je echtem Node-Typ.

Quelle ist die `nodes`-Tabelle der n8n-mcp-Datenbank (dieselbe, aus der
daten/vorlagen_messen.py die Vorlagen liest). Die Tabelle schreibt Typen
in Kurzform (`nodes-base.webhook`), unsere Grundwahrheit
`bewertung/echte_node_typen.json` in Langform (`n8n-nodes-base.webhook`) --
`normalisiere()` uebersetzt. Nur Typen aus der 825er-Liste kommen in den
Katalog: was die lokale n8n-Installation nicht kennt, darf hier nicht als
"echt" auftauchen.

Wozu der Katalog gebraucht wird:
  - instruktionen.py: der Anzeigename ("Google Sheets Trigger") ist der
    Wortschatz, ueber den eine Instruktion in normaler Sprache zum Typ
    findet -- ohne LLM, deterministisch.
  - mutationen.py: Typentausch nur INNERHALB einer Kategorie (trigger,
    input, output, transform) und nur gleichartig (Ausloeser gegen
    Ausloeser, Tool-Variante gegen Tool-Variante), sonst entsteht ein
    Workflow, der zwar die Tore passiert, aber semantisch Unsinn ist.

    python workflow/katalog.py            # schreibt workflow/node_katalog.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HIER = Path(__file__).parent
KATALOG_DATEI = HIER / "node_katalog.json"
TYPEN_DATEI = HIER.parent / "bewertung" / "echte_node_typen.json"
DB_STANDARD = Path.home() / "AppData/Local/npm-cache/_npx/b6a381d62ce0fe56/node_modules/n8n-mcp/data/nodes.db"

PRAEFIXE = {
    "nodes-base.": "n8n-nodes-base.",
    "nodes-langchain.": "@n8n/n8n-nodes-langchain.",
}


def normalisiere(typ: str) -> str:
    for kurz, lang in PRAEFIXE.items():
        if typ.startswith(kurz):
            return lang + typ[len(kurz):]
    return typ


def lade_echte_typen() -> set[str]:
    return set(json.loads(TYPEN_DATEI.read_text(encoding="utf8"))["node_typen"])


def baue(db: Path) -> dict:
    echte = lade_echte_typen()
    con = sqlite3.connect(db)
    rows = con.execute(
        "select node_type, display_name, category, is_trigger, is_tool_variant, "
        "package_name from nodes where package_name in "
        "('n8n-nodes-base', '@n8n/n8n-nodes-langchain')"
    ).fetchall()
    katalog = {}
    for typ, anzeige, kategorie, ausloeser, tool, paket in rows:
        voll = normalisiere(typ)
        if voll not in echte:
            continue
        katalog[voll] = {
            "anzeige": anzeige or voll.rsplit(".", 1)[-1],
            "kategorie": kategorie or "unbekannt",
            "ausloeser": bool(ausloeser),
            "tool": bool(tool) or voll.endswith("Tool"),
            "paket": paket,
        }
    # Typen aus der 825er-Liste, die die Tabelle nicht kennt (z. B. zur
    # Laufzeit erzeugte *Tool-Varianten): mit Minimalangaben aufnehmen,
    # damit jeder echte Typ EINEN Eintrag hat -- aber ohne Kategorie, also
    # vom Typentausch ausgeschlossen (mutationen.py tauscht nur mit Kategorie).
    for voll in echte - set(katalog):
        kurz = voll.rsplit(".", 1)[-1]
        basis = kurz[:-4] if kurz.endswith("Tool") else kurz
        vorlage = katalog.get(voll.rsplit(".", 1)[0] + "." + basis)
        katalog[voll] = {
            "anzeige": (vorlage["anzeige"] + " Tool") if vorlage else kurz,
            "kategorie": "unbekannt",
            "ausloeser": kurz.endswith("Trigger"),
            "tool": kurz.endswith("Tool"),
            "paket": voll.rsplit(".", 1)[0],
        }
    return dict(sorted(katalog.items()))


def lade() -> dict:
    if not KATALOG_DATEI.exists():
        raise SystemExit(f"{KATALOG_DATEI} fehlt -- erst `python workflow/katalog.py`")
    return json.loads(KATALOG_DATEI.read_text(encoding="utf8"))["typen"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DB_STANDARD)
    args = p.parse_args()
    if not args.db.exists():
        raise SystemExit(f"Datenbank fehlt: {args.db}")
    katalog = baue(args.db)
    mit_kat = sum(1 for v in katalog.values() if v["kategorie"] != "unbekannt")
    aus = {
        "quelle": "nodes-Tabelle der n8n-mcp-Datenbank, gefiltert auf bewertung/echte_node_typen.json",
        "erzeugt_von": "workflow/katalog.py",
        "anzahl": len(katalog),
        "mit_kategorie": mit_kat,
        "typen": katalog,
    }
    KATALOG_DATEI.write_text(json.dumps(aus, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"{len(katalog)} Typen im Katalog, {mit_kat} mit Kategorie -> {KATALOG_DATEI}")


if __name__ == "__main__":
    main()
