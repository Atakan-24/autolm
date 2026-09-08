"""
DIE VIER TORE -- der objektive Bewertungsmassstab dieses Projekts.

Kein "klingt gut", keine Punktzahl aus einem zweiten LLM. Vier Ja/Nein-
Fragen, jede maschinell und ohne API-Kosten pruefbar:

  Tor 1  Ist die Ausgabe gueltiges JSON?
  Tor 2  Hat sie die richtige Struktur? (nodes/connections, Pflichtfelder,
         und -- das Herzstueck -- existiert JEDER verwendete Node-Typ
         wirklich in n8n?)
  Tor 3  Sind die Verbindungen in sich konsistent? (jedes Verbindungsziel
         zeigt auf einen existierenden Node)
  Tor 4  Laesst er sich wirklich importieren? -- siehe importtest.py,
         das ist die einzige Pruefung, die ECHTES n8n aufruft statt einer
         eigenen Nachbildung.

WARUM TOR 2 GEGEN EINE ECHTE, EXTRAHIERTE NODE-LISTE PRUEFT UND NICHT GEGEN
DIE STDIO-MCP-VERBINDUNG (n8n-mcp): n8n-mcp ist ein reiner MCP-Server, der
ueber das JSON-RPC-Stdio-Protokoll spricht -- kein Kommandozeilenwerkzeug,
das sich aus einem Python-Skript heraus einfach aufrufen liesse. Statt das
Protokoll nachzubauen, wird die AUTORITATIVE QUELLE direkt benutzt: die
lokal installierte n8n-Version selbst (dist/known/nodes.json aus
n8n-nodes-base, 439 Typen, extrahiert 08.09.2026 -- Datei
bewertung/echte_node_typen.json). Das ist DIESELBE Grundwahrheit, gegen die
auch n8n-mcp letztlich prueft. Und Tor 4 ruft ohnehin das ECHTE n8n auf --
die staerkste, massgebliche Pruefung haengt an keiner Nachbildung.

    python bewertung/tore.py --datei <workflow.json>
    python bewertung/tore.py --text '{"nodes": [...], ...}'
"""

import argparse
import json
import sys
from pathlib import Path

HIER = Path(__file__).parent
NODE_TYPEN_DATEI = HIER / "echte_node_typen.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lade_echte_node_typen() -> set[str]:
    if not NODE_TYPEN_DATEI.exists():
        raise FileNotFoundError(
            f"{NODE_TYPEN_DATEI} fehlt. Ohne die Liste ist Tor 2 nicht pruefbar."
        )
    d = json.loads(NODE_TYPEN_DATEI.read_text(encoding="utf8"))
    return set(d["node_typen"])


ECHTE_NODE_TYPEN = None  # lazy geladen, damit ein Import ohne Datei nicht sofort scheitert


def _typen():
    global ECHTE_NODE_TYPEN
    if ECHTE_NODE_TYPEN is None:
        ECHTE_NODE_TYPEN = lade_echte_node_typen()
    return ECHTE_NODE_TYPEN


# ===========================================================================
# TOR 1 -- gueltiges JSON
# ===========================================================================

def tor1_json(text: str) -> tuple[bool, dict | None, str | None]:
    """Gibt (bestanden, geparstes_objekt_oder_None, fehlermeldung_oder_None) zurueck."""
    try:
        obj = json.loads(text)
        return True, obj, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


# ===========================================================================
# TOR 2 -- Struktur + echte Node-Typen
# ===========================================================================

def tor2_struktur(obj: dict) -> tuple[bool, list[str]]:
    """
    Prueft die minimale n8n-Workflow-Struktur:
      - "nodes" ist eine Liste
      - "connections" ist ein Objekt
      - jeder Node hat id/name/type/typeVersion/position/parameters
      - jeder Node-TYP existiert wirklich in n8n (gegen die echte Liste)

    Gibt (bestanden, liste_der_probleme) zurueck -- eine leere Liste heisst
    bestanden. Sammelt ALLE Probleme statt beim ersten abzubrechen, damit
    ein Bericht auch bei mehreren Fehlern vollstaendig ist.
    """
    probleme = []
    if not isinstance(obj, dict):
        return False, ["Wurzel ist kein Objekt"]

    nodes = obj.get("nodes")
    if not isinstance(nodes, list):
        probleme.append("'nodes' fehlt oder ist keine Liste")
        nodes = []
    if not isinstance(obj.get("connections"), dict):
        probleme.append("'connections' fehlt oder ist kein Objekt")

    pflichtfelder = ["id", "name", "type", "typeVersion", "position", "parameters"]
    typen = _typen()
    node_namen = set()

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            probleme.append(f"Node {i}: kein Objekt")
            continue
        for feld in pflichtfelder:
            if feld not in node:
                probleme.append(f"Node {i} ({node.get('name', '?')}): Feld '{feld}' fehlt")

        typ = node.get("type")
        if typ is not None and typ not in typen:
            probleme.append(f"Node {i} ({node.get('name', '?')}): "
                            f"unbekannter Node-Typ {typ!r} -- existiert nicht in n8n {NODE_TYPEN_DATEI.exists() and json.loads(NODE_TYPEN_DATEI.read_text())['n8n_version']}")

        name = node.get("name")
        if name is not None:
            if name in node_namen:
                probleme.append(f"Doppelter Node-Name: {name!r}")
            node_namen.add(name)

    return len(probleme) == 0, probleme


# ===========================================================================
# TOR 3 -- Verbindungen konsistent
# ===========================================================================

def tor3_verbindungen(obj: dict) -> tuple[bool, list[str]]:
    """
    n8n-Verbindungen haben die Form:
        connections[<Quellname>][<Typ, meist 'main'>][<Ausgang-Index>]
            -> Liste von {node: <Zielname>, type, index}

    Prueft: jeder Quell- UND Zielname zeigt auf einen existierenden Node.
    """
    probleme = []
    nodes = obj.get("nodes", [])
    if not isinstance(nodes, list):
        return False, ["Keine Node-Liste vorhanden -- Tor 2 haette das schon melden muessen"]
    namen = {n.get("name") for n in nodes if isinstance(n, dict)}

    verbindungen = obj.get("connections", {})
    if not isinstance(verbindungen, dict):
        return False, ["'connections' ist kein Objekt"]

    for quelle, typen_dict in verbindungen.items():
        if quelle not in namen:
            probleme.append(f"Verbindung startet bei unbekanntem Node {quelle!r}")
        if not isinstance(typen_dict, dict):
            probleme.append(f"connections[{quelle!r}] ist kein Objekt")
            continue
        for verbindungstyp, ausgaenge in typen_dict.items():
            if not isinstance(ausgaenge, list):
                continue
            for ausgang in ausgaenge:
                if not isinstance(ausgang, list):
                    continue
                for ziel in ausgang:
                    if not isinstance(ziel, dict):
                        probleme.append(f"Verbindungsziel in {quelle!r} ist kein Objekt")
                        continue
                    zielname = ziel.get("node")
                    if zielname not in namen:
                        probleme.append(
                            f"Verbindung {quelle!r} -> unbekannter Zielnode {zielname!r}"
                        )

    return len(probleme) == 0, probleme


# ===========================================================================
# Alle Tore zusammen
# ===========================================================================

def pruefe_alle_tore(text: str) -> dict:
    """
    Gibt ein Ergebnis-Dict zurueck, das direkt als JSON geloggt werden kann
    -- dieselbe Bauart wie bewertung/ergebnisse/*.json im Plan: jede
    README-Zahl muss auf einen echten Lauf zeigen.
    """
    ergebnis = {
        "tor1_json": False,
        "tor2_struktur": False,
        "tor3_verbindungen": False,
        "probleme": [],
    }

    ok1, obj, fehler1 = tor1_json(text)
    ergebnis["tor1_json"] = ok1
    if not ok1:
        ergebnis["probleme"].append(f"Tor 1: {fehler1}")
        return ergebnis  # ohne gueltiges JSON koennen die anderen Tore nicht pruefen

    ok2, probleme2 = tor2_struktur(obj)
    ergebnis["tor2_struktur"] = ok2
    ergebnis["probleme"].extend(f"Tor 2: {p}" for p in probleme2)

    ok3, probleme3 = tor3_verbindungen(obj)
    ergebnis["tor3_verbindungen"] = ok3
    ergebnis["probleme"].extend(f"Tor 3: {p}" for p in probleme3)

    ergebnis["alle_bestanden_ohne_import"] = ok1 and ok2 and ok3
    return ergebnis


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--datei", type=Path)
    g.add_argument("--text")
    args = p.parse_args()

    text = args.datei.read_text(encoding="utf8") if args.datei else args.text
    ergebnis = pruefe_alle_tore(text)
    print(json.dumps(ergebnis, indent=2, ensure_ascii=False))
    sys.exit(0 if ergebnis.get("alle_bestanden_ohne_import") else 1)


if __name__ == "__main__":
    main()
