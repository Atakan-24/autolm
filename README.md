# AutoLM

**Ein Sprachmodell von Grund auf — von zufälligen Gewichten bis zum
deployten Modell, das Automations-Workflows schreibt.**

Kein fertiges Modell feingetunt. Kein API-Wrapper. Der Transformer, der
Tokenizer, die Trainingsschleife: selbst gebaut.

> Status: **Schritt 4 von 7** — in Arbeit, öffentlich ab Schritt 3.

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

Hier ist die Bewertung **eine Maschine, kein Gefühl**:

| Frage | messbar |
|---|---|
| Ist die Ausgabe gültiges JSON? | ja/nein |
| Validiert n8n den Workflow? | ja/nein, über die n8n-API |
| Lässt er sich importieren und starten? | ja/nein |

Damit ist die zentrale Behauptung des Projekts überhaupt prüfbar:
**ein winziges Spezialmodell schlägt ein großes Allzweckmodell auf dessen
eigener Aufgabe** — gemessen mit demselben Validator, gegen GPT-4 und Claude.

## Aufbau

| | Schritt | Ergebnis |
|---|---|---|
| 1 | Backpropagation von Hand | ✅ `schritte/01_wie_lernt_ein_computer.py` |
| 2 | Eigener Tokenizer (BPE) | ✅ `schritte/02_tokenizer.py` |
| 3 | Attention von Hand | ✅ `schritte/03_attention.py` |
| 4 | Decoder-only Transformer (PyTorch) | ✅ `schritte/04_transformer.py` |
| 5 | Pretraining — **Funktionstest** auf öffentlichem Korpus | offen |
| 6 | Training auf Workflow-Daten + Instruction-Tuning | offen |
| 7 | Evaluation, Quantisierung, Web-Demo | offen |

**Schritt 5 ist bewusst ein Funktionstest, kein Selbstzweck.** Ein
selbstgebauter Transformer muss erst auf einem Datensatz laufen, bei dem
bewiesen ist, dass es klappt — sonst ist bei einem Fehlschlag nie
unterscheidbar, ob der Code falsch ist oder die Daten zu schwer sind.

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

- **Kein Konkurrent zu GPT.** Das Modell wird klein und dumm sein.
  Der Anspruch ist Verständnis und ein prüfbares Nischenergebnis.
- **Kein Milliarden-Parameter-Lauf.** Viel GPU-Geld auf ein großes Modell zu
  werfen beweist nicht, dass man Training verstanden hat.
- **Keine Kundendaten.** Das Projekt läuft auf öffentlichen Datensätzen.

## Kosten

0 € bis Schritt 7. Training auf Google Colab (kostenlose GPU-Kontingente,
Verfügbarkeit nicht garantiert).
