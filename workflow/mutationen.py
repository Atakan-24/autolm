"""
STRUKTURERHALTENDE MUTATIONEN -- aus 1.974 Vorlagen werden zehntausende.

1.974 gueltige Vorlagen sind fuer ein Sprachmodell zu wenig -- das Modell
wuerde sie auswendig lernen, nicht die Regeln dahinter. Der Hebel aus dem
Plan: Mutationen, die einen gueltigen Workflow in einen ANDEREN gueltigen
Workflow ueberfuehren, und JEDE Mutante laeuft durch denselben Validator
(bewertung/tore.py), der spaeter auch das Modell bewertet. Datengenerator
und Bewertungsinstanz sind dasselbe Orakel -- kein Beispiel im Trainings-
satz, das der Validator nicht abgesegnet hat.

Vier Operationen, alle deterministisch bei gleichem Seed:

  permutiere      Reihenfolge der Knoten mischen. Die Kanten haengen an
                  Namen, nicht an Positionen -- der Graph bleibt identisch,
                  die Kurzschrift-Nummerierung aendert sich. Lehrt: n3
                  ist keine Eigenschaft eines Knotens.
  umbenenne       Knoten bekommen den Katalog-Anzeigenamen ihres Typs
                  ("Send Email", "Slack 2"). Lehrt die Bruecke Typ <->
                  Anzeigename, ueber die eine Instruktion in normaler
                  Sprache zum Typ findet.
  tausche_typ     ein Typ wird gegen einen GLEICHARTIGEN getauscht: gleiche
                  Kategorie (trigger/input/output/transform), gleiche
                  Ausloeser- und Tool-Eigenschaft, gleiches Paket. Die
                  Version kommt aus den in den Vorlagen beobachteten
                  Versionen des neuen Typs, der Name wird zum Anzeigenamen.
                  Lehrt: 825 Typen, nicht nur die 60 haeufigsten.
  entferne_blatt  ein Knoten ohne ausgehende Kanten (und kein Ausloeser)
                  faellt weg samt seiner eingehenden Kanten. Lehrt:
                  kuerzere Workflows sind auch Workflows.

Was NICHT mutiert wird: Kanten einfuegen oder umhaengen. Eine Kante von
einem Slack-Knoten in einen KI-Agenten ueber `ai_languageModel` waere
strukturell "gueltig" (Tor 3 prueft nur, ob beide Namen existieren) und
inhaltlich Unsinn -- der Validator kann semantische Vertraeglichkeit nicht
pruefen, also darf die Mutation sie nicht zufaellig erzeugen.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bewertung"))
from tore import pruefe_alle_tore  # noqa: E402

from . import kurzschrift as ks  # noqa: E402


def versionen_je_typ(vorlagen: list[dict]) -> dict[str, str]:
    """Haeufigste typeVersion je Typ, als Kurzschrift-Text ('4.2')."""
    z = defaultdict(Counter)
    for v in vorlagen:
        for n in v["wf"]["nodes"]:
            z[n["type"]][ks._version_text(n.get("typeVersion"))] += 1
    return {t: c.most_common(1)[0][0] for t, c in z.items()}


def tauschgruppen(katalog: dict) -> dict[tuple, list[str]]:
    g = defaultdict(list)
    for typ, e in katalog.items():
        if e["kategorie"] == "unbekannt":
            continue
        g[(e["kategorie"], e["ausloeser"], e["tool"], e["paket"])].append(typ)
    return {k: sorted(v) for k, v in g.items() if len(v) >= 2}


def _kopie(wf: dict) -> dict:
    return json.loads(json.dumps(wf))


def _benenne_um(wf: dict, alt: str, neu: str) -> None:
    for n in wf["nodes"]:
        if n["name"] == alt:
            n["name"] = neu
    conns = wf["connections"]
    if alt in conns:
        conns[neu] = conns.pop(alt)
    for typen in conns.values():
        for ausgaenge in typen.values():
            for ausgang in ausgaenge:
                for ziel in ausgang:
                    if ziel["node"] == alt:
                        ziel["node"] = neu


def _eindeutig(name: str, belegt: set[str]) -> str:
    kandidat, k = name, 2
    while kandidat in belegt:
        kandidat = f"{name} {k}"
        k += 1
    return kandidat


# ---------------------------------------------------------------------------
# die vier Operationen -- jede gibt einen NEUEN Workflow zurueck oder None
# ---------------------------------------------------------------------------

def permutiere(wf: dict, rng: random.Random, **_) -> dict | None:
    if len(wf["nodes"]) < 2:
        return None
    neu = _kopie(wf)
    rng.shuffle(neu["nodes"])
    return neu


def umbenenne(wf: dict, rng: random.Random, katalog: dict, **_) -> dict | None:
    neu = _kopie(wf)
    kandidaten = [n for n in neu["nodes"]
                  if n["name"] != katalog.get(n["type"], {}).get("anzeige")]
    if not kandidaten:
        return None
    anzahl = rng.randint(1, min(4, len(kandidaten)))
    belegt = {n["name"] for n in neu["nodes"]}
    for n in rng.sample(kandidaten, anzahl):
        anzeige = katalog.get(n["type"], {}).get("anzeige") or n["type"].rsplit(".", 1)[-1]
        ziel = _eindeutig(anzeige, belegt - {n["name"]})
        if ziel == n["name"]:
            continue
        belegt.discard(n["name"])
        belegt.add(ziel)
        _benenne_um(neu, n["name"], ziel)
    return neu


def tausche_typ(wf: dict, rng: random.Random, katalog: dict, gruppen: dict,
                versionen: dict, **_) -> dict | None:
    neu = _kopie(wf)
    tauschbar = []
    for n in neu["nodes"]:
        e = katalog.get(n["type"])
        if not e or e["kategorie"] == "unbekannt":
            continue
        schl = (e["kategorie"], e["ausloeser"], e["tool"], e["paket"])
        if schl in gruppen:
            tauschbar.append((n, schl))
    if not tauschbar:
        return None
    n, schl = rng.choice(tauschbar)
    alternativen = [t for t in gruppen[schl] if t != n["type"]]
    # bevorzugt Typen, deren Version aus den Vorlagen bekannt ist -- fuer die
    # anderen bleibt Version 1, was bei n8n meist existiert, aber nicht immer
    bekannt = [t for t in alternativen if t in versionen]
    neuer_typ = rng.choice(bekannt or alternativen)
    n["type"] = neuer_typ
    n["typeVersion"] = ks._version_wert(versionen.get(neuer_typ, "1"))
    anzeige = katalog[neuer_typ]["anzeige"]
    belegt = {x["name"] for x in neu["nodes"]} - {n["name"]}
    _benenne_um(neu, n["name"], _eindeutig(anzeige, belegt))
    return neu


def entferne_blatt(wf: dict, rng: random.Random, katalog: dict, **_) -> dict | None:
    if len(wf["nodes"]) < 3:
        return None
    quellen = set(wf["connections"].keys())
    blaetter = [n["name"] for n in wf["nodes"]
                if n["name"] not in quellen
                and not katalog.get(n["type"], {}).get("ausloeser")]
    if not blaetter:
        return None
    weg = rng.choice(blaetter)
    neu = _kopie(wf)
    neu["nodes"] = [n for n in neu["nodes"] if n["name"] != weg]
    for quelle in list(neu["connections"]):
        typen = neu["connections"][quelle]
        for vtyp in list(typen):
            typen[vtyp] = [[z for z in ausgang if z["node"] != weg]
                           for ausgang in typen[vtyp]]
        # leere Ausgangslisten am Ende abschneiden, leere Typen entfernen
        for vtyp in list(typen):
            while typen[vtyp] and not typen[vtyp][-1]:
                typen[vtyp].pop()
            if not typen[vtyp]:
                del typen[vtyp]
        if not typen:
            del neu["connections"][quelle]
    return neu


OPERATIONEN = {
    "permutiere": permutiere,
    "umbenenne": umbenenne,
    "tausche_typ": tausche_typ,
    "entferne_blatt": entferne_blatt,
}


def ist_gueltig(wf: dict, mit_namen: bool = False) -> bool:
    """Kurzschrift-Rundlauf + Tor 1-3 -- exakt der Weg, den spaeter das Modell geht."""
    try:
        text = ks.serialisiere(wf, mit_namen)
        gerendert = ks.rendere(text)
    except ks.KurzschriftFehler:
        return False
    return bool(pruefe_alle_tore(json.dumps(gerendert))["alle_bestanden_ohne_import"])


def mutiere(wf: dict, rng: random.Random, katalog: dict, gruppen: dict,
            versionen: dict, hoechstens_ops: int = 3,
            mit_namen: bool = False) -> tuple[dict, list[str]] | None:
    """1..hoechstens_ops zufaellige Operationen; None, wenn nichts Gueltiges entsteht.
    Ohne Namen in der Kurzschrift ist `umbenenne` wirkungslos und wird ausgelassen."""
    aktuell = wf
    angewandt = []
    ops = list(OPERATIONEN) if mit_namen else [o for o in OPERATIONEN if o != "umbenenne"]
    for _ in range(rng.randint(1, hoechstens_ops)):
        name = rng.choice(ops)
        ergebnis = OPERATIONEN[name](aktuell, rng, katalog=katalog,
                                     gruppen=gruppen, versionen=versionen)
        if ergebnis is None:
            continue
        aktuell = ergebnis
        angewandt.append(name)
    if not angewandt or not ist_gueltig(aktuell, mit_namen):
        return None
    return aktuell, angewandt


def erzeuge_mutanten(vorlage: dict, anzahl: int, seed: int, katalog: dict,
                     gruppen: dict, versionen: dict, mit_namen: bool = False) -> list[dict]:
    """
    Bis zu `anzahl` VERSCHIEDENE gueltige Mutanten einer Vorlage. Verschieden
    heisst: andere Kurzschrift als das Original und als jede andere Mutante --
    sonst zaehlt das Modell dasselbe Beispiel mehrfach.
    """
    rng = random.Random(f"{seed}:{vorlage['id']}")
    gesehen = {ks.serialisiere(vorlage["wf"], mit_namen)}
    aus, versuche = [], 0
    while len(aus) < anzahl and versuche < anzahl * 6:
        versuche += 1
        m = mutiere(vorlage["wf"], rng, katalog, gruppen, versionen, mit_namen=mit_namen)
        if m is None:
            continue
        wf, ops = m
        text = ks.serialisiere(wf, mit_namen)
        if text in gesehen:
            continue
        gesehen.add(text)
        aus.append({"quelle_id": vorlage["id"], "ops": ops, "wf": wf})
    return aus
