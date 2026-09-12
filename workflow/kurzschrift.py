"""
WORKFLOW-KURZSCHRIFT -- die Textform, die das Modell lernt.

Warum nicht das rohe n8n-JSON? Gemessen ueber die 2.012 gueltigen Vorlagen
(daten/vorlagen_messen.py): selbst OHNE Parameter hat ein Workflow im
JSON median 3.156 Zeichen, p90 7.522 -- das sind mit einem BPE-Vokabular
von 4096 grob 900 bis 2.100 Token je Workflow. Ein kleines Modell mit
Kontext 512 saehe die Haelfte der Workflows nie am Stueck. Und der
groesste Teil dieser Zeichen ist Wiederholung: Anfuehrungszeichen,
Schluesselnamen, UUIDs, Positionen -- Dinge, an denen laut Stufe 3 KEIN
einziges Frontier-Modell scheitert (Tor 1 und Tor 3 bei ~99 %).

Die Kurzschrift behaelt genau das, woran sie scheitern -- den exakten
Typnamen -- und die Struktur, an der es zu messen gilt: Knoten, Versionen,
Kanten. Alles andere rechnet `rendere()` deterministisch zurueck:

    wf Lead-Erinnerung
    n1 n8n-nodes-base.formTrigger@2.2 Formular
    n2 n8n-nodes-base.wait@1.1 Warte 24h
    n3 n8n-nodes-base.emailSend@2.1 Erinnerung senden
    n1 > n2
    n2 > n3
    n4 n8n-nodes-base.slack@2.2 Slack
    n2 >1 n4                      (Ausgang 1, z. B. false-Zweig eines IF)
    n5 @n8n/n8n-nodes-langchain.lmChatOpenAi@1 Modell
    n5 ai_languageModel> n6       (Verbindungstyp, wenn nicht 'main')

DIE TYPNAMEN WERDEN AUSGESCHRIEBEN, nicht als ein Sondertoken kodiert.
Das ist Absicht und der Kern der Wette: 89 % der Fehler grosser Modelle
sind `sendEmail` statt `emailSend` -- ein Rechtschreibfehler in einem
geschlossenen Vokabular. Ein Modell, das die Typen Byte fuer Byte
schreiben muss, muss sie wirklich auswendig koennen. Ein Sondertoken je
Typ haette den Fehler konstruktionsbedingt unmoeglich gemacht und damit
die Messung entwertet (steht als Ablationsarm offen, siehe README).

Was die Kurzschrift bewusst NICHT traegt (und `rendere()` erfindet):
  - Node-IDs (uuid5 aus dem Namen -- deterministisch, kein Zufall)
  - Positionen (Raster nach topologischer Reihenfolge)
  - Parameter (immer {} -- Tor 2 verlangt nur, dass das Feld ein Objekt ist)
Ein gerenderter Workflow ist damit ein gueltiges Geruest, kein fertig
konfigurierter Workflow. Das steht so im README; die Parameter sind eine
eigene, spaetere Stufe.
"""

import re
import uuid
from collections import defaultdict, deque

NAMENSRAUM = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # fest -> reproduzierbare IDs

_KNOTEN = re.compile(r"^n(\d+) (\S+)@(\S+)(?: (.*))?$")
# "n1 > n2", "n1 >2 n4", "n5 ai_tool> n6", "n5 ai_tool>1 n6"
_KANTE = re.compile(r"^n(\d+) ([A-Za-z_]*)>(\d*) n(\d+)$")


class KurzschriftFehler(ValueError):
    pass


# ---------------------------------------------------------------------------
# JSON -> Kurzschrift
# ---------------------------------------------------------------------------

def _version_text(v) -> str:
    if isinstance(v, bool) or v is None:
        return "1"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v) if v != int(v) else str(int(v))
    return str(v)


def _saeubere(s) -> str:
    return " ".join(str(s).split()) or "Knoten"


_KATALOG = None


def _anzeigename(typ: str) -> str:
    """Katalog-Anzeigename des Typs ("Send Email"); Rueckfall: letzter Teil des Typs."""
    global _KATALOG
    if _KATALOG is None:
        try:
            from .katalog import lade
            _KATALOG = lade()
        except Exception:
            _KATALOG = {}
    e = _KATALOG.get(typ)
    return e["anzeige"] if e else typ.rsplit(".", 1)[-1]


def serialisiere(wf: dict, mit_namen: bool = False) -> str:
    """
    n8n-Workflow (dict) -> Kurzschrift-Text. Knoten in der Reihenfolge der
    `nodes`-Liste, Kanten in der Reihenfolge von `connections`.

    FASSUNG 2 (Vorgabe, `mit_namen=False`): OHNE Knotennamen und ohne
    Workflow-Namen. Gemessen an Fassung 1 (mit Namen, 7M-Modell, 2.400
    Schritte): das Modell schrieb die Typen fehlerfrei -- 0 erfundene Typen
    ueber alle Bewertungen -- und scheiterte fast immer an den NAMEN:
    frei getippte Beschriftungen wie "Answer- Get Flag" sind Rauschen, das
    kein Modell dieser Groesse vorhersagen kann; es verlor darin die
    Nummerierung und lief in Wiederholungen ("Knoten n6 doppelt", "n87").
    Der Name traegt fuer Tor 1-4 nichts -- `rendere()` setzt stattdessen
    den Katalog-Anzeigenamen des Typs, eindeutig gemacht.
    `mit_namen=True` bleibt fuer den Vergleich (Ablationsarm) erhalten.
    """
    nodes = wf.get("nodes", [])
    nummer = {}
    zeilen = [f"wf {_saeubere(wf.get('name') or 'Workflow')}" if mit_namen else "wf"]
    for i, n in enumerate(nodes, 1):
        name = n.get("name")
        if name in nummer:
            raise KurzschriftFehler(f"doppelter Node-Name {name!r}")
        nummer[name] = i
        kopf = f"n{i} {n['type']}@{_version_text(n.get('typeVersion'))}"
        zeilen.append(f"{kopf} {_saeubere(name)}" if mit_namen else kopf)
    for quelle, typen in (wf.get("connections") or {}).items():
        if quelle not in nummer:
            raise KurzschriftFehler(f"Kante von unbekanntem Node {quelle!r}")
        for vtyp, ausgaenge in (typen or {}).items():
            for idx, ausgang in enumerate(ausgaenge or []):
                for ziel in ausgang or []:
                    zn = ziel.get("node")
                    if zn not in nummer:
                        raise KurzschriftFehler(f"Kante zu unbekanntem Node {zn!r}")
                    t = "" if vtyp == "main" else vtyp
                    i = "" if idx == 0 else str(idx)
                    zeilen.append(f"n{nummer[quelle]} {t}>{i} n{nummer[zn]}")
    return "\n".join(zeilen)


# ---------------------------------------------------------------------------
# Kurzschrift -> JSON
# ---------------------------------------------------------------------------

def parse(text: str) -> dict:
    """
    Kurzschrift -> Zwischenform {name, knoten: [(nr, typ, version, name)],
    kanten: [(von, vtyp, idx, nach)]}. Wirft KurzschriftFehler bei jeder
    Zeile, die keinem der drei Muster entspricht -- das ist Tor 0 des
    eigenen Modells: "hat es ueberhaupt Kurzschrift geschrieben?".
    """
    knoten, kanten, name = [], [], None
    for roh in text.strip().splitlines():
        zeile = roh.strip()
        if not zeile:
            continue
        if zeile == "wf" or zeile.startswith("wf "):
            if name is not None:
                raise KurzschriftFehler("zweite wf-Zeile")
            name = zeile[3:].strip() or "Workflow"
            continue
        m = _KNOTEN.match(zeile)
        if m:
            knoten.append((int(m.group(1)), m.group(2), m.group(3), (m.group(4) or "").strip()))
            continue
        m = _KANTE.match(zeile)
        if m:
            kanten.append((int(m.group(1)), m.group(2) or "main",
                           int(m.group(3) or 0), int(m.group(4))))
            continue
        raise KurzschriftFehler(f"unlesbare Zeile: {zeile!r}")
    if name is None:
        raise KurzschriftFehler("wf-Zeile fehlt")
    return {"name": name, "knoten": knoten, "kanten": kanten}


def _version_wert(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            raise KurzschriftFehler(f"ungueltige Version {v!r}")


def rendere(text: str) -> dict:
    """
    Kurzschrift -> vollstaendiger n8n-Workflow (dict), so wie ihn Tor 2-4
    erwarten. Alles Erfundene ist deterministisch: gleicher Text, gleiches
    JSON, Byte fuer Byte.
    """
    z = parse(text)
    nr_zu_name, nodes = {}, []
    for nr, typ, version, name in z["knoten"]:
        if nr in nr_zu_name:
            raise KurzschriftFehler(f"Knoten n{nr} doppelt")
        name = name or _anzeigename(typ)
        # doppelte Namen eindeutig machen -- n8n verlangt Eindeutigkeit
        basis, k = name, 2
        while name in nr_zu_name.values():
            name = f"{basis} {k}"
            k += 1
        nr_zu_name[nr] = name
        nodes.append({
            "id": str(uuid.uuid5(NAMENSRAUM, f"{nr}:{typ}:{name}")),
            "name": name,
            "type": typ,
            "typeVersion": _version_wert(version),
            "position": [0, 0],
            "parameters": {},
        })
    for von, _, _, nach in z["kanten"]:
        for x in (von, nach):
            if x not in nr_zu_name:
                raise KurzschriftFehler(f"Kante nutzt unbekannten Knoten n{x}")

    # Positionen: Raster nach topologischer Tiefe (Kahn), Breite nach Reihenfolge
    nachfolger, eingang = defaultdict(list), {nr: 0 for nr in nr_zu_name}
    for von, _, _, nach in z["kanten"]:
        nachfolger[von].append(nach)
        eingang[nach] += 1
    tiefe = {nr: 0 for nr in nr_zu_name}
    schlange = deque(nr for nr in nr_zu_name if eingang[nr] == 0)
    while schlange:
        nr = schlange.popleft()
        for f in nachfolger[nr]:
            tiefe[f] = max(tiefe[f], tiefe[nr] + 1)
            eingang[f] -= 1
            if eingang[f] == 0:
                schlange.append(f)
    spalte_belegt = defaultdict(int)
    for node, nr in zip(nodes, nr_zu_name):
        t = tiefe[nr]
        node["position"] = [240 * t, 200 * spalte_belegt[t]]
        spalte_belegt[t] += 1

    connections = {}
    for von, vtyp, idx, nach in z["kanten"]:
        q = connections.setdefault(nr_zu_name[von], {}).setdefault(vtyp, [])
        while len(q) <= idx:
            q.append([])
        q[idx].append({"node": nr_zu_name[nach], "type": vtyp, "index": 0})

    return {
        "id": str(uuid.uuid5(NAMENSRAUM, "wf:" + z["name"])),
        "name": z["name"],
        "nodes": nodes,
        "connections": connections,
        "settings": {"executionOrder": "v1"},
    }


def kanten_index(wf: dict) -> set[tuple]:
    """(Quell-Index, Verbindungstyp, Ausgang, Ziel-Index) ueber die Knotenreihenfolge --
    namensunabhaengig, also der richtige Vergleich fuer die Kurzschrift ohne Namen."""
    idx = {n.get("name"): i for i, n in enumerate(wf.get("nodes", []))}
    return {(idx.get(q), t, a, idx.get(z)) for q, t, a, z in kanten_menge(wf)}


def kanten_menge(wf: dict) -> set[tuple]:
    """(Quellname, Verbindungstyp, Ausgang, Zielname) -- fuer Vergleiche im Test."""
    m = set()
    for q, typen in (wf.get("connections") or {}).items():
        for vtyp, ausgaenge in (typen or {}).items():
            for idx, ausgang in enumerate(ausgaenge or []):
                for ziel in ausgang or []:
                    m.add((q, vtyp, idx, ziel.get("node")))
    return m
