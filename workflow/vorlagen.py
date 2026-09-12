"""
VORLAGEN-LADER -- die 2.012 gueltigen n8n-Vorlagen als Rohmaterial.

Liest dieselbe Datenbank wie daten/vorlagen_messen.py und wendet dieselbe
Regel an: eine Vorlage zaehlt nur, wenn JEDER ihrer Node-Typen in der
825er-Grundwahrheit steht. Community-Pakete bleiben draussen (lokal nicht
installiert, Tor 4 koennte sie nie importieren).

Gibt je Vorlage zurueck: id, name, metadata (use_cases, required_services,
key_features -- fuer instruktionen.py), und den auf das Noetige reduzierten
Workflow (name, nodes[name/type/typeVersion], connections).
"""

import base64
import gzip
import json
import sqlite3
from pathlib import Path

from .katalog import DB_STANDARD, lade_echte_typen


def _entpacke(s: str):
    try:
        return json.loads(gzip.decompress(base64.b64decode(s)))
    except Exception:
        return None


def reduziere(wf: dict) -> dict | None:
    """Nur die Felder, die die Kurzschrift traegt. None bei kaputter Struktur."""
    nodes = wf.get("nodes")
    if not isinstance(nodes, list) or not isinstance(wf.get("connections"), dict):
        return None
    aus, namen = [], set()
    for n in nodes:
        if not isinstance(n, dict) or not n.get("type") or not n.get("name"):
            return None
        # Haftnotizen sind Kommentare im Editor, keine Arbeitsschritte -- sie
        # haben nie Kanten und wuerden dem Modell nur beibringen, "Sticky
        # Note" als Baustein zu nennen. Raus, bevor irgendetwas sie sieht.
        if n["type"] == "n8n-nodes-base.stickyNote":
            continue
        # Namen normalisieren (Mehrfach-Leerzeichen, Zeilenumbrueche) -- die
        # Kurzschrift ist zeilenbasiert, und ein Name mit Umbruch braeche sie.
        name = " ".join(str(n["name"]).split())
        if not name or name in namen:
            return None
        namen.add(name)
        aus.append({"name": name, "type": n["type"],
                    "typeVersion": n.get("typeVersion", 1)})
    # Verbindungen muessen die Form haben, die Tor 3 prueft -- alles andere
    # ist keine Vorlage, sondern ein kaputter Export, und faellt raus.
    saeuber = lambda x: " ".join(str(x).split())
    connections = {}
    for quelle, typen in wf["connections"].items():
        quelle = saeuber(quelle)
        if quelle not in namen or not isinstance(typen, dict):
            return None
        neu_typen = {}
        for vtyp, ausgaenge in typen.items():
            if not isinstance(ausgaenge, list):
                return None
            neu_ausgaenge = []
            for ausgang in ausgaenge:
                if not isinstance(ausgang, list):
                    return None
                neu_ausgang = []
                for ziel in ausgang:
                    if not isinstance(ziel, dict):
                        return None
                    zn = saeuber(ziel.get("node"))
                    if zn not in namen:
                        return None
                    neu_ausgang.append({"node": zn, "type": vtyp, "index": 0})
                neu_ausgaenge.append(neu_ausgang)
            neu_typen[vtyp] = neu_ausgaenge
        connections[quelle] = neu_typen
    if not aus:
        return None
    return {"name": saeuber(wf.get("name") or "Workflow"), "nodes": aus,
            "connections": connections}


def lade(db: Path = DB_STANDARD, hoechstens: int | None = None) -> list[dict]:
    if not Path(db).exists():
        raise SystemExit(f"Vorlagen-Datenbank fehlt: {db}")
    echte = lade_echte_typen()
    con = sqlite3.connect(db)
    aus = []
    for tid, name, meta, comp in con.execute(
        "select id, name, metadata_json, workflow_json_compressed from templates "
        "where workflow_json_compressed is not null order by id"
    ):
        wf = _entpacke(comp)
        if wf is None:
            continue
        typen = {n.get("type") for n in wf.get("nodes", []) if isinstance(n, dict)}
        if not typen or not typen <= echte:
            continue
        red = reduziere(wf)
        if red is None:
            continue
        try:
            metadata = json.loads(meta) if meta else {}
        except json.JSONDecodeError:
            metadata = {}
        aus.append({"id": int(tid), "name": name, "metadata": metadata, "wf": red})
        if hoechstens and len(aus) >= hoechstens:
            break
    return aus
