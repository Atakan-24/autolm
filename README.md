# AutoLM

**Ein Sprachmodell von Grund auf — von zufälligen Gewichten bis zu einem
Modell, das gegen GPT-4 und Claude auf einer maschinell prüfbaren Aufgabe
gemessen wird.**

Kein fertiges Modell feingetunt. Kein API-Wrapper. Der Transformer, der
Tokenizer, die Trainingsschleife: selbst gebaut, Schritt für Schritt.

> Status: **Stufe 0 abgeschlossen** — Fundament fertig, Pretraining folgt.

---

## Das Ziel

Ein kleines, spezialisiertes Modell, das aus einer Beschreibung in normaler
Sprache einen lauffähigen [n8n](https://n8n.io)-Workflow erzeugt.

```
"Wenn ein neuer Lead reinkommt, warte 24 Stunden, dann schick eine Mail"
                              ↓
        { "nodes": [ { "type": "n8n-nodes-base.webhook", ... } ] }
```

## Warum diese Aufgabe

Die meisten From-Scratch-LLM-Projekte enden bei „schreibt halbwegs Text" —
und ob das gut ist, kann niemand objektiv sagen.

Hier ist die Bewertung **eine Maschine, kein Gefühl**, über vier Tore:

| Tor | Frage | Prüfung |
|---|---|---|
| 1 | Ist die Ausgabe gültiges JSON? | `json.loads()` |
| 2 | Hat sie die richtige Struktur? | Schema-Prüfung, reines Python |
| 3 | Validiert n8n den Workflow? | `n8n-mcp`s `validate_workflow` |
| 4 | Lässt er sich importieren und starten? | `n8n import:workflow`, echter Exit-Code |

**Offene, noch nicht gemessene Frage — bewusst als Frage formuliert, nicht
als Behauptung:** wie nah kommt ein kleines, für ~0 € trainiertes
Spezialmodell an GPT-4/Claude auf dieser engen Aufgabe, und auf welchen
Achsen (Gültigkeitsquote, Latenz, Kosten je 1.000 Anfragen, Offline-Betrieb)
gewinnt es, auf welchen verliert es? Die einzige Achse, auf der ein kleines
Modell plausibel gewinnen kann, ist die **Gültigkeitsquote** — große Modelle
erfinden zuverlässig Node-Typen, die es nicht gibt. Das wird gemessen,
nicht behauptet, sobald der Eval-Harness steht (Stufe 3 unten).

## Aufbau

| | Stufe | Ergebnis |
|---|---|---|
| 1–4 | Backprop · Tokenizer · Attention · Transformer, alles von Hand | ✅ `schritte/01`–`04` |
| **0** | **Modell aus 04 herausgelöst, schneller Attention-Pfad, gegen Lehrpfad bewiesen gleich** | ✅ `kern/` |
| 1 | Datenpipeline: Tokenizer-Durchsatz messen, TinyStories vortokenisieren | offen |
| 2 | Pretraining, 17M Parameter, absturzsicheres Checkpointing | offen |
| 3 | **Eval-Harness — vor dem eigenen Modell.** GPT-4/Claude/Groq gegen die vier Tore | offen |
| 4 | Workflow-Modell: validator-gesicherte synthetische Trainingsdaten | offen |
| 5 | Eingeschränkte Dekodierung (falls nötig) | offen |
| 6 | Ablation (3 Seeds), Skalierungskurve, Interpretierbarkeit gegen den echten Parse-Baum | offen |
| 7 | Quantisierung, Hugging-Face-Demo | offen |

**Warum der Eval-Harness (Stufe 3) vor dem eigenen Modell kommt:** er
liefert ein eigenständig veröffentlichbares Ergebnis, selbst wenn danach
nichts mehr klappt — „wie oft erzeugen Frontier-Modelle tatsächlich
importierbare n8n-Workflows?" ist bisher, soweit bekannt, nirgends gemessen.

**Warum Pretraining (Stufe 2) auf Kindergeschichten (TinyStories) läuft und
nicht direkt auf Workflow-Daten:** ein selbstgebauter Transformer muss erst
auf einem Datensatz laufen, bei dem bewiesen ist, dass es funktioniert —
sonst ist bei einem Fehlschlag nie unterscheidbar, ob der Code falsch ist
oder die Aufgabe zu schwer war. Das darauf vortrainierte Modell wird
**nicht** ins Workflow-Training übernommen: Kindergeschichten liefern
keinen nützlichen Prior für `n8n-nodes-base.httpRequest`. Stufe 4 trainiert
ein frisches Modell — mit der in Stufe 1–2 gebauten Infrastruktur, aber
eigenem Tokenizer und eigenen Daten.

## Stufe 0 — was gerade fertig wurde

`kern/modell.py` — das Modell aus `schritte/04_transformer.py`
herausgelöst, damit Pretraining, Ablation und Interpretierbarkeit später
garantiert dasselbe Modell verwenden.

Dazu ein zweiter Attention-Pfad: `MehrKoepfeSchnell` fasst die
Query/Key/Value-Berechnung aller Köpfe in **einem** Matrixprodukt zusammen
und nutzt `F.scaled_dot_product_attention` statt der Formel von Hand.

**Der wichtigste Test im Repo ist nicht der schnellste Code, sondern der
Beweis, dass beide Pfade dasselbe rechnen** — eine Optimierung, die
nebenbei das Ergebnis ändert, ist kein schnelleres Modell, sondern ein
kaputtes mit Stoppuhr, und das fällt nicht auf: der Loss sinkt trotzdem.

```bash
cd kern
python test_qkv_gleichheit.py   # 6 Tests: Ausgabe, Gradienten, ganzes Modell
python mess_speedup.py          # Geschwindigkeit, getrennt vom Korrektheitstest
```

**Gemessen auf diesem Server (3 CPU-Kerne, 2 GHz):**

```
Lehrpfad (Schleife)     343.7 ms/Schritt
fusioniertes QKV        186.8 ms/Schritt
Faktor: 1.84x
```

Auf CPU ist der Gewinn moderat, weil dort ohnehin ein Kern nach dem
anderen rechnet. Auf GPU ist der Aufruf-Overhead selbst der Engpass — die
GPU-Zahl steht nach dem ersten Colab-Lauf hier.

## Schritt 1 — was drinsteht

`schritte/01_wie_lernt_ein_computer.py` — **keine Bibliothek, nur Python.**
Kein PyTorch, kein numpy. Ein Autograd-System in unter 150 Zeilen, das die
vier Begriffe zeigt, aus denen jedes Training besteht:

```
LOSS       eine Zahl: wie falsch war ich
GRADIENT   welche Stellschraube ist wie sehr schuld
BACKPROP   das Rückwärts-Verteilen dieser Schuld
LERNRATE   wie große Schritte gemacht werden
```

Der Lauf lässt einen Computer eine Regel finden, die ihm niemand gesagt hat
(`2x + 3y`), allein aus Beispielen — und zeigt am Ende, wie dieselbe Aufgabe
bei zu großer Lernrate explodiert.

```bash
python schritte/01_wie_lernt_ein_computer.py
```

## Nicht-Ziele

- **Kein Konkurrent zu GPT.** Das Modell wird klein sein. Der Anspruch ist
  Verständnis und ein prüfbares Nischenergebnis, nicht Allzwecksprache.
- **Kein Milliarden-Parameter-Lauf.** Viel GPU-Geld auf ein großes Modell zu
  werfen beweist nicht, dass man Training verstanden hat.
- **Keine Kundendaten.** Läuft ausschließlich auf öffentlichen Datensätzen
  (TinyStories, öffentliche n8n-Workflow-Vorlagen).
- **Kein DPO.** Die Belohnung hier ist binär und maschinell prüfbar
  (validiert / validiert nicht) — wo ein verifizierbares Signal vorliegt,
  ist DPO das falsche Werkzeug. Stattdessen, falls überhaupt: Rejection
  Sampling gegen den Validator.
- **Kein Distributed Training.** Bei diesem Modell und dieser Netzwerk-
  bandbreite wäre die Kommunikation langsamer als eine einzelne Maschine —
  das zu bauen würde belegen, dass dieses Verhältnis nicht abgeschätzt
  werden kann, nicht das Gegenteil.

## Kosten

0 € bis auf einen bewusst gewählten Posten in Stufe 3 (~5 € API-Kosten für
den fairen Vergleich gegen GPT-4/Claude — wird im README genannt, sobald
er anfällt). Training auf Google Colab (kostenlose GPU-Kontingente,
Verfügbarkeit nicht garantiert).
