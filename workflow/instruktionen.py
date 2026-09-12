"""
INSTRUKTIONEN -- deterministisch aus den Metadaten, kein LLM, 0 Euro.

Jedes Trainingsbeispiel ist ein Paar (Instruktion in normaler Sprache,
Workflow in Kurzschrift). Die Instruktion wird NICHT von einem grossen
Modell geschrieben (das waere ein zweites Orakel, teuer und nicht
reproduzierbar), sondern aus drei Quellen zusammengesetzt, die alle im
Repo bzw. in der Vorlagen-Datenbank liegen:

  use_cases          aus metadata_json der Vorlage ("automate lead
                     discovery") -- nur fuer ORIGINALE, eine Mutante mit
                     getauschtem Typ hat diesen Anwendungsfall nicht mehr
  Ausloeser          Anzeigename des ersten Trigger-Knotens aus dem Katalog
  Bausteine          Anzeigenamen aller uebrigen Typen, in Knotenreihenfolge

Der Anzeigename ist die Bruecke: "Google Sheets Trigger", "Slack", "Send
Email" sind die Woerter, mit denen ein Mensch die Aufgabe beschreibt --
und genau die muss das Modell auf `n8n-nodes-base.emailSend` abbilden.

Mehrere Formulierungsvarianten (deutsch und englisch, mit und ohne
Anwendungsfall), gewaehlt ueber einen aus (Vorlagen-ID, Variante)
abgeleiteten Seed -- derselbe Datensatz-Bau ergibt Byte fuer Byte
dieselben Instruktionen.

Grenze, die man kennen muss: die 15 Test-Instruktionen in
bewertung/instruktionen.jsonl sind frei formulierte Saetze ("Wenn ein
neuer Lead in einem Formular ankommt, warte 24 Stunden ..."). Die hier
erzeugten sind schematischer. Der Abstand zwischen beiden ist Teil der
Messung -- ein Modell, das nur die Schablone kann, faellt im Test auf.
"""

import random

MAX_BAUSTEINE = 12


def _anzeige(typ: str, katalog: dict) -> str:
    e = katalog.get(typ)
    return e["anzeige"] if e else typ.rsplit(".", 1)[-1]


def bausteine(wf: dict, katalog: dict) -> tuple[str | None, list[str]]:
    """(Anzeigename des ersten Ausloesers oder None, eindeutige Anzeigenamen der uebrigen)."""
    ausloeser, rest, gesehen = None, [], set()
    for n in wf["nodes"]:
        e = katalog.get(n["type"], {})
        a = _anzeige(n["type"], katalog)
        if e.get("ausloeser") and ausloeser is None:
            ausloeser = a
            continue
        if a not in gesehen:
            gesehen.add(a)
            rest.append(a)
    return ausloeser, rest


def _liste(teile: list[str], und: str) -> str:
    if len(teile) > MAX_BAUSTEINE:
        rest = len(teile) - MAX_BAUSTEINE
        teile = teile[:MAX_BAUSTEINE] + [f"{rest} weitere" if und == "und" else f"{rest} more"]
    if len(teile) <= 1:
        return "".join(teile)
    return ", ".join(teile[:-1]) + f" {und} " + teile[-1]


def erzeuge(wf: dict, metadata: dict | None, katalog: dict, seed: str,
            ist_original: bool) -> str:
    rng = random.Random(seed)
    ausloeser, rest = bausteine(wf, katalog)
    use_cases = [u for u in (metadata or {}).get("use_cases", []) if isinstance(u, str) and u.strip()]
    use_case = rng.choice(use_cases).strip().rstrip(".") if (ist_original and use_cases) else None

    deutsch = rng.random() < 0.6
    saetze = []
    if deutsch:
        if use_case:
            saetze.append(rng.choice([
                f"Baue einen n8n-Workflow, der Folgendes erledigt: {use_case}.",
                f"Erstelle einen Workflow fuer: {use_case}.",
                f"Ich brauche eine Automation, die {use_case}.",
            ]))
        else:
            saetze.append(rng.choice([
                "Baue einen n8n-Workflow.",
                "Erstelle einen Workflow mit den folgenden Bausteinen.",
                "Ich brauche eine Automation aus diesen Schritten.",
            ]))
        if ausloeser:
            saetze.append(rng.choice([
                f"Ausloeser: {ausloeser}.",
                f"Er startet mit {ausloeser}.",
                f"Trigger ist {ausloeser}.",
            ]))
        if rest:
            saetze.append(rng.choice([
                f"Verwende {_liste(rest, 'und')}.",
                f"Bausteine: {_liste(rest, 'und')}.",
                f"Danach: {' -> '.join(rest[:MAX_BAUSTEINE])}.",
            ]))
    else:
        if use_case:
            saetze.append(rng.choice([
                f"Build an n8n workflow that does the following: {use_case}.",
                f"Create a workflow to {use_case}.",
                f"I need an automation that {use_case}.",
            ]))
        else:
            saetze.append(rng.choice([
                "Build an n8n workflow.",
                "Create a workflow from the following building blocks.",
            ]))
        if ausloeser:
            saetze.append(rng.choice([
                f"Trigger: {ausloeser}.",
                f"It starts with {ausloeser}.",
            ]))
        if rest:
            saetze.append(rng.choice([
                f"Use {_liste(rest, 'and')}.",
                f"Steps: {' -> '.join(rest[:MAX_BAUSTEINE])}.",
            ]))
    return " ".join(saetze)
