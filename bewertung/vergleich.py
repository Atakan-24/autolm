"""
STUFE 3 -- DER VERGLEICH: FUEHRT ALLE VIER TORE UEBER EINEN STAPEL VON
KANDIDATEN, EGAL VON WELCHEM MODELL SIE STAMMEN.

Zweck: dieselbe Frage an unterschiedliche Modelle stellen (GPT-4, Claude,
spaeter das eigene trainierte Modell) und ALLE mit demselben Massstab
messen.

ABLAUF, PRO KANDIDAT:
    Ausgabe des Modells (roher Text)
        -> Tor 1 (gueltiges JSON?)          -- tore.py
        -> Tor 2 (Struktur + echte Node-Typen?)  -- tore.py
        -> Tor 3 (Verbindungen konsistent?) -- tore.py
    Nur wer alle drei besteht, geht weiter zu:
        -> Tor 4 (importiert n8n es wirklich?) -- importtest.py, GESAMMELT
           als EIN Stapel, nicht einzeln (siehe importtest.py-Docstring:
           ein einziges kaputtes JSON wuerde sonst den ganzen Stapelimport
           zum Absturz bringen -- deshalb laufen nur Tor-1-3-Bestandene mit).

ERGEBNIS: eine JSON-Datei pro Lauf unter bewertung/ergebnisse/, damit JEDE
Zahl im README auf einen echten, nachvollziehbaren Lauf zeigt statt auf
eine Behauptung.

WAS DIESE DATEI NICHT TUT: keine LLM-Aufrufe. Die Eingaben (Instruktion ->
rohe Modellantwort) muessen VORHER gesammelt und als JSONL uebergeben
werden -- eine Zeile je Kandidat:
    {"modell": "gpt-4o", "instruktion": "...", "antwort_roh": "..."}

Der GPT-4/Claude-Vergleich selbst kostet echtes Geld (~5 EUR laut Plan) --
das ist ein bewusster STOP-IF-Punkt, kein technischer. Dieses Skript
bewertet nur, es ruft nie selbst eine bezahlte API auf.

    python bewertung/vergleich.py --kandidaten <antworten.jsonl> --out <ergebnisse.json>
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HIER = Path(__file__).parent
sys.path.insert(0, str(HIER))
import tore
import importtest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _extrahiere_json(text: str) -> str:
    """
    LLM-Antworten kommen selten als reines JSON -- oft mit Codeblock-
    Markierungen (```json ... ```) oder erklaerendem Text drumherum. Diese
    Funktion versucht, den ERSTEN vollstaendigen JSON-Block herauszuloesen,
    OHNE ihn zu reparieren -- ein Modell, das kein sauberes JSON liefert,
    soll das auch als Fehler zu spueren bekommen, nicht durch Grosszuegigkeit
    beim Parsen belohnt werden. Nur Codeblock-Markierungen werden entfernt,
    das ist reine Formatierung, kein inhaltlicher Unterschied.
    """
    t = text.strip()
    if t.startswith("```"):
        zeilen = t.split("\n")
        if zeilen[0].startswith("```"):
            zeilen = zeilen[1:]
        if zeilen and zeilen[-1].strip() == "```":
            zeilen = zeilen[:-1]
        t = "\n".join(zeilen)
    return t.strip()


def bewerte_stapel(kandidaten: list[dict]) -> dict:
    """
    kandidaten: Liste von {"modell": str, "instruktion": str, "antwort_roh": str, ["id": str]}

    Gibt ein vollstaendiges Ergebnis-Dict zurueck: pro Kandidat die
    Tor-Ergebnisse, plus aggregierte Quoten je Modell.
    """
    for i, k in enumerate(kandidaten):
        k.setdefault("id", f"k{i:04d}")

    # --- Tore 1-3, lokal, kein n8n noetig ---
    einzel = {}
    tor123_bestanden_ids = []
    tempordner = HIER / "_stapel_fuer_import"
    tempordner.mkdir(exist_ok=True)
    for f in tempordner.glob("*.json"):
        f.unlink()

    for k in kandidaten:
        text = _extrahiere_json(k["antwort_roh"])
        ergebnis = tore.pruefe_alle_tore(text)
        einzel[k["id"]] = {
            "modell": k["modell"],
            "instruktion": k["instruktion"],
            "tor1_json": ergebnis["tor1_json"],
            "tor2_struktur": ergebnis["tor2_struktur"],
            "tor3_verbindungen": ergebnis["tor3_verbindungen"],
            "tor4_importiert": None,  # erst unten befuellt
            "probleme": ergebnis["probleme"],
        }
        if ergebnis["alle_bestanden_ohne_import"]:
            tor123_bestanden_ids.append(k["id"])
            # Fuer Tor 4 braucht n8n eine gueltige "id" -- falls das
            # Modell selbst keine (oder eine unbrauchbare) mitgeliefert
            # hat, wird der Dateiname als Ersatz-ID verwendet, damit der
            # Import nicht an einer fehlenden ID scheitert, waehrend der
            # eigentliche INHALT unveraendert bleibt.
            obj = json.loads(text)
            # IMMER ueberschreiben, nicht setdefault: das Modell kann seine
            # EIGENE "id" im JSON mitliefern (Kandidat k0000 hatte z.B.
            # "id":"a1"). Ohne das Ueberschreiben landet der Workflow unter
            # "a1" in der DB, waehrend spaeter nach "k0000" gesucht wird --
            # der Treffer bleibt aus, Tor 4 wird faelschlich als False
            # gemeldet, obwohl der Import erfolgreich war. Gefunden beim
            # ersten Testlauf mit synthetischen Kandidaten.
            obj["id"] = k["id"]
            obj.setdefault("name", k["id"])
            (tempordner / f"{k['id']}.json").write_text(
                json.dumps(obj), encoding="utf8"
            )

    # --- Tor 4, nur fuer die, die 1-3 bestanden haben ---
    if tor123_bestanden_ids:
        if not importtest.DB_ORDNER.exists():
            print("Wegwerf-n8n-DB fehlt -- initialisiere einmalig (dauert laenger) ...")
            importtest.init_db(zeige_ausgabe=False)
        import_ergebnis = importtest.teste_ordner(tempordner)
        importierte_ids = set(import_ergebnis["gefundene_ids"])
        for kid in tor123_bestanden_ids:
            einzel[kid]["tor4_importiert"] = kid in importierte_ids
    else:
        import_ergebnis = {"dateien": 0, "importiert_laut_n8n": 0}

    for f in tempordner.glob("*.json"):
        f.unlink()
    tempordner.rmdir()

    # --- Aggregation je Modell ---
    je_modell = defaultdict(lambda: {"n": 0, "tor1": 0, "tor2": 0, "tor3": 0,
                                       "tor4": 0, "alle_vier": 0})
    for kid, e in einzel.items():
        m = je_modell[e["modell"]]
        m["n"] += 1
        m["tor1"] += bool(e["tor1_json"])
        m["tor2"] += bool(e["tor2_struktur"])
        m["tor3"] += bool(e["tor3_verbindungen"])
        m["tor4"] += bool(e["tor4_importiert"])
        m["alle_vier"] += bool(
            e["tor1_json"] and e["tor2_struktur"]
            and e["tor3_verbindungen"] and e["tor4_importiert"]
        )

    zusammenfassung = {}
    for modell, m in je_modell.items():
        n = m["n"]
        zusammenfassung[modell] = {
            "n": n,
            "tor1_json_quote": m["tor1"] / n,
            "tor2_struktur_quote": m["tor2"] / n,
            "tor3_verbindungen_quote": m["tor3"] / n,
            "tor4_import_quote": m["tor4"] / n,
            "gueltigkeitsquote_alle_vier": m["alle_vier"] / n,
        }

    return {
        "zeit": time.strftime("%Y-%m-%d %H:%M:%S"),
        "anzahl_kandidaten": len(kandidaten),
        "je_modell": zusammenfassung,
        "einzelergebnisse": einzel,
    }


def bootstrap_konfidenzintervall(erfolge: int, n: int, wiederholungen: int = 10000,
                                  seed: int = 0) -> tuple[float, float]:
    """
    95-%-Bootstrap-Konfidenzintervall auf einer Quote -- dieselbe Methodik
    wie tgcrm/scripts/ml/baseline.py. Bei kleinem n (z.B. 20 Testkandidaten)
    ist eine nackte Prozentzahl irrefuehrend praezise; das Intervall zeigt,
    wie viel Unsicherheit tatsaechlich drinsteckt.
    """
    import random
    random.seed(seed)
    proben = [1] * erfolge + [0] * (n - erfolge)
    quoten = []
    for _ in range(wiederholungen):
        stich = [random.choice(proben) for _ in range(n)]
        quoten.append(sum(stich) / n)
    quoten.sort()
    unten = quoten[int(0.025 * wiederholungen)]
    oben = quoten[int(0.975 * wiederholungen)]
    return unten, oben


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kandidaten", type=Path, required=True,
                    help="JSONL mit {modell, instruktion, antwort_roh}")
    p.add_argument("--out", type=Path, default=HIER / "ergebnisse" / "lauf.json")
    args = p.parse_args()

    kandidaten = [json.loads(z) for z in args.kandidaten.read_text(encoding="utf8").splitlines() if z.strip()]
    print(f"{len(kandidaten)} Kandidaten geladen aus {args.kandidaten}")

    ergebnis = bewerte_stapel(kandidaten)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"\nErgebnis gespeichert: {args.out}\n")

    print(f"{'Modell':<20} | {'n':>4} | {'Tor1':>6} | {'Tor2':>6} | {'Tor3':>6} | "
          f"{'Tor4':>6} | {'Gueltig':>8} | 95%-KI")
    print("-" * 95)
    for modell, z in ergebnis["je_modell"].items():
        erfolge = round(z["gueltigkeitsquote_alle_vier"] * z["n"])
        unten, oben = bootstrap_konfidenzintervall(erfolge, z["n"])
        print(f"{modell:<20} | {z['n']:>4} | {z['tor1_json_quote']*100:>5.0f}% | "
              f"{z['tor2_struktur_quote']*100:>5.0f}% | {z['tor3_verbindungen_quote']*100:>5.0f}% | "
              f"{z['tor4_import_quote']*100:>5.0f}% | {z['gueltigkeitsquote_alle_vier']*100:>7.0f}% | "
              f"[{unten*100:.0f}%, {oben*100:.0f}%]")


if __name__ == "__main__":
    main()
