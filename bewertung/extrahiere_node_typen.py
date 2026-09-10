"""
ERZEUGT bewertung/echte_node_typen.json -- die Grundwahrheit fuer Tor 2.

Vorher gab es diese Datei nur als einmal von Hand erzeugtes Artefakt ohne
Erzeuger-Skript im Repo -- dieselbe Fehlerklasse wie daten/tokenizer.pkl
(Commit 3e7d1ce): etwas, das "in Sekunden neu erzeugbar" aussieht, aber von
keinem Skript neu erzeugt wird. Und sie war UNVOLLSTAENDIG, gemessen mit
daten/vorlagen_messen.py gegen 2.352 echte Vorlagen (10.09.2026):

  - 38,4 % der Vorlagen nutzen @n8n/n8n-nodes-langchain (Agent, LLM-Chat,
    Output-Parser, ...) -- das Paket ist Teil jeder n8n-Installation, stand
    aber nicht in der Liste, weil nur n8n-nodes-base extrahiert worden war.
  - 601 Vorlagen nutzen n8n-nodes-base.*Tool-Varianten (httpRequestTool,
    gmailTool, googleSheetsTool, ...). Die stehen in KEINER known/nodes.json:
    n8n erzeugt sie zur Laufzeit fuer jeden Node, dessen Beschreibung
    `usableAsTool: true` traegt. Der Validator haette jede davon als
    "halluziniert" abgelehnt -- ein Massstab, der echte Typen verwirft, ist
    kein Massstab.

Drei Quellen, alle aus der LOKAL INSTALLIERTEN n8n-Version (nicht aus dem
Netz, nicht aus einer Vorlage -- sonst wuerde die Grundwahrheit aus genau
den Daten stammen, die sie spaeter bewerten soll):

  1. n8n-nodes-base/dist/known/nodes.json              -> n8n-nodes-base.<key>
  2. @n8n/n8n-nodes-langchain/dist/known/nodes.json    -> @n8n/n8n-nodes-langchain.<key>
  3. fuer jeden Node aus 1 und 2: traegt eine seiner Klassendateien
     `usableAsTool: true`, gibt es zusaetzlich <typ>Tool.
     Dateizuordnung ueber den Klassennamen: <ClassName>.node.js oder
     <ClassName>V<n>.node.js im Ordner der sourcePath -- die Version steht bei
     versionierten Nodes (HttpRequest -> HttpRequestV3) in der Versions-
     datei, nicht in der Hauptdatei. Ein blosses "irgendwo im Ordner" waere
     zu grosszuegig: im Ordner Google/Sheet liegt auch der Trigger, und
     Trigger sind nie Tools.

Gegenprobe am Ende, gegen die Vorlagen-DB (falls vorhanden): jeder
*Tool-Typ, der in einer echten Vorlage vorkommt, muss in der erzeugten Liste
stehen (Recall). Fehlt einer, ist die Dateizuordnung oben zu eng -- dann
laut sagen, nicht still weglassen.

    python bewertung/extrahiere_node_typen.py            # schreibt die Datei
    python bewertung/extrahiere_node_typen.py --pruefen  # nur vergleichen
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HIER = Path(__file__).parent
ZIEL = HIER / "echte_node_typen.json"

USABLE_RE = re.compile(r"usableAsTool\s*:\s*(true|!0)")


def n8n_node_modules() -> Path:
    """Der node_modules-Ordner der global installierten n8n."""
    try:
        wurzel = subprocess.run(["npm", "root", "-g"], capture_output=True,
                                text=True, check=True, shell=True).stdout.strip()
    except Exception as e:
        sys.exit(f"npm root -g fehlgeschlagen: {e}")
    p = Path(wurzel) / "n8n" / "node_modules"
    if not p.is_dir():
        sys.exit(f"n8n nicht global installiert? {p} fehlt")
    return p


def n8n_version(nm: Path) -> str:
    return json.loads((nm.parent / "package.json").read_text(encoding="utf8"))["version"]


def lade_known(paket_dir: Path) -> dict:
    return json.loads((paket_dir / "dist" / "known" / "nodes.json").read_text(encoding="utf8"))


def ist_usable_as_tool(paket_dir: Path, eintrag: dict) -> bool:
    quelle = paket_dir / eintrag["sourcePath"]
    klasse = eintrag["className"]
    ordner = quelle.parent
    muster = re.compile(rf"^{re.escape(klasse)}(V\d+)?\.node\.js$")
    for datei in ordner.rglob("*.js"):
        # Zwei Orte, an denen die Beschreibung liegen kann -- gemessen, nicht
        # vermutet (erste Fassung kannte nur den ersten und verfehlte damit
        # googleSheetsTool, postgresTool, googleDriveTool, notionTool):
        #   a) <ClassName>.node.js oder <ClassName>V<n>.node.js
        #   b) versionDescription.js in versionierten Nodes (v2/actions/...)
        ist_klasse = bool(muster.match(datei.name))
        ist_version = datei.name.lower() == "versiondescription.js"
        if not (ist_klasse or ist_version):
            continue
        try:
            if USABLE_RE.search(datei.read_text(encoding="utf8", errors="ignore")):
                return True
        except OSError:
            pass
    return False


def sonderfaelle_aus_core(nm: Path) -> set[str]:
    """Typen, die n8n NICHT ueber usableAsTool erzeugt, sondern in n8n-core
    hart verdrahtet -- gemessen: httpRequestTool steht in KEINER Node-Datei,
    sondern als HTTP_REQUEST_AS_TOOL_NODE_TYPE in n8n-core/dist/constants.js.
    Wird von dort GELESEN, nicht abgetippt, damit es bei einem n8n-Update
    mitwandert."""
    datei = nm / "n8n-core" / "dist" / "constants.js"
    if not datei.exists():
        return set()
    text = datei.read_text(encoding="utf8", errors="ignore")
    return set(re.findall(r"_AS_TOOL_NODE_TYPE\s*=\s*['\"]([^'\"]+)['\"]", text))


def erzeuge(nm: Path) -> dict:
    pakete = {
        "n8n-nodes-base": nm / "n8n-nodes-base",
        "@n8n/n8n-nodes-langchain": nm / "@n8n" / "n8n-nodes-langchain",
    }
    typen: set[str] = set()
    zaehler = {}
    for prefix, pdir in pakete.items():
        known = lade_known(pdir)
        basis = 0
        tools = 0
        for key, eintrag in known.items():
            typen.add(f"{prefix}.{key}")
            basis += 1
            if not key.endswith("Trigger") and ist_usable_as_tool(pdir, eintrag):
                typen.add(f"{prefix}.{key}Tool")
                tools += 1
        zaehler[prefix] = {"basis": basis, "tool_varianten": tools}
    sonder = sonderfaelle_aus_core(nm)
    typen |= sonder
    zaehler["n8n-core Sonderfaelle"] = sorted(sonder)
    return {
        "n8n_version": n8n_version(nm),
        "extrahiert_am": date.today().isoformat(),
        "quelle": ("lokal installierte n8n: n8n-nodes-base + @n8n/n8n-nodes-langchain "
                   "(dist/known/nodes.json) + *Tool-Varianten fuer usableAsTool:true"),
        "erzeugt_von": "bewertung/extrahiere_node_typen.py",
        "je_paket": zaehler,
        "anzahl": len(typen),
        "node_typen": sorted(typen),
    }


def gegenprobe(typen: set[str]):
    """Recall gegen die Vorlagen-DB: welche in echten Vorlagen benutzten Typen
    der beiden Pakete fehlen in der Liste? (Community-Pakete bewusst nicht.)"""
    try:
        sys.path.insert(0, str(HIER.parent / "daten"))
        from vorlagen_messen import DB_STANDARD
    except Exception:
        return
    if not DB_STANDARD.exists():
        print("(Gegenprobe uebersprungen: Vorlagen-DB nicht gefunden)")
        return
    import sqlite3
    from collections import Counter
    con = sqlite3.connect(DB_STANDARD)
    benutzt = Counter()
    for (nodes_used,) in con.execute("select nodes_used from templates"):
        try:
            for t in set(json.loads(nodes_used or "[]")):
                if t.startswith("n8n-nodes-base.") or t.startswith("@n8n/n8n-nodes-langchain."):
                    benutzt[t] += 1
        except Exception:
            pass
    fehlend = {t: c for t, c in benutzt.items() if t not in typen}
    abgedeckt = sum(c for t, c in benutzt.items() if t in typen)
    gesamt = sum(benutzt.values())
    print(f"\nGegenprobe gegen Vorlagen-DB: {len(benutzt)} verschiedene Typen der zwei "
          f"Pakete in Benutzung, {abgedeckt}/{gesamt} Vorkommen abgedeckt "
          f"({100*abgedeckt/gesamt:.1f} %).")
    if fehlend:
        print(f"FEHLEN in der Liste ({len(fehlend)} Typen) -- Vorkommen in Vorlagen:")
        for t, c in sorted(fehlend.items(), key=lambda x: -x[1])[:30]:
            print(f"  {c:5d}  {t}")
    else:
        print("Kein benutzter Typ der zwei Pakete fehlt.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pruefen", action="store_true",
                   help="nur erzeugen und mit der bestehenden Datei vergleichen, nichts schreiben")
    args = p.parse_args()

    nm = n8n_node_modules()
    neu = erzeuge(nm)
    print(f"n8n {neu['n8n_version']}: {neu['anzahl']} Typen  {neu['je_paket']}")

    if ZIEL.exists():
        alt = set(json.loads(ZIEL.read_text(encoding="utf8"))["node_typen"])
        neu_set = set(neu["node_typen"])
        print(f"bisher: {len(alt)}  neu: {len(neu_set)}  "
              f"dazu: {len(neu_set - alt)}  weg: {len(alt - neu_set)}")
        if alt - neu_set:
            print("  WEGGEFALLEN (pruefen!):", sorted(alt - neu_set)[:20])

    gegenprobe(set(neu["node_typen"]))

    if args.pruefen:
        return
    ZIEL.write_text(json.dumps(neu, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"\ngeschrieben: {ZIEL}")


if __name__ == "__main__":
    main()
