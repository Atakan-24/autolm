# AutoLM

**Ein Sprachmodell von Grund auf — von zufälligen Gewichten bis zu einem
Modell, das gegen GPT-4 und Claude auf einer maschinell prüfbaren Aufgabe
gemessen wird.**

Kein fertiges Modell feingetunt. Kein API-Wrapper. Der Transformer, der
Tokenizer, die Trainingsschleife: selbst gebaut, Schritt für Schritt.

> Status: **Stufe 0–3 abgeschlossen, Stufe 4 vermessen** — 384,8 Mio. echte
> Trainings-Token vorbereitet (über dem Chinchilla-optimalen Ziel),
> Eval-Harness gegen echtes n8n läuft, fünf Modelle durchgemessen (alle
> 100 % — der leichte Modus ist zu leicht, siehe Stufe 3), 2.012 gültige
> Vorlagen ≈ 7,3 Mio. Token als Rohmaterial für Stufe 4 gezählt. Fehlt: der
> GPU-Trainingslauf (Google Colab, läuft) und der Bau des Workflow-Modells.

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
| **0** | Modell aus 04 herausgelöst, schneller Attention-Pfad, gegen Lehrpfad bewiesen gleich | ✅ `kern/` |
| **1** | Datenpipeline: BPE-Encoder + -Trainer 39× beschleunigt, **384.762.231 echte TinyStories-Token** vortokenisiert (über dem 340M-Chinchilla-Ziel) | ✅ `kern/bpe_*`, `daten/vortokenisiere.py` |
| **2** | Absturzsicheres Checkpointing (echter Kill-und-Resume-Beweis) + Trainingsloop fertig. **GPU-Lauf selbst noch offen** — Colab-Notebook liegt bereit | 🔶 Infrastruktur fertig, Training offen |
| **3** | **Eval-Harness gegen echtes n8n 2.25.6** — vier Tore, Vergleichs-Orchestrator. Vergleich gegen GPT-4/Claude noch nicht gefahren (kostet ~5 €) | ✅ `bewertung/` |
| 4 | Workflow-Modell: validator-gesicherte synthetische Trainingsdaten | offen — Community-Node-Scope noch zu klären |
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

## Stufe 1 — die Datenpipeline, und warum sie zweimal gebaut wurde

Der naive BPE-Encoder/-Trainer aus Schritt 2 schafft 0,018 MB/s — auf den
2-GB-TinyStories-Korpus hochgerechnet wären das **31 Stunden**. Gemessen,
nicht geraten, bevor irgendetwas anderes gebaut wurde.

Der Fix ändert den Algorithmus nicht (Byte Pair Encoding bleibt exakt
dasselbe Verfahren), nur die Datenstruktur: eine verkettete Liste + ein
Min-Heap statt „nach jedem Merge die ganze Folge neu durchsuchen".
`kern/bpe_schnell.py` (Encoder) und `kern/bpe_training_schnell.py`
(Trainer) — beide mit eigenen Gleichheitstests gegen den langsamen
Referenz-Encoder bewiesen, nicht nur behauptet schneller.

```
Encoder:  13,3× (Heap) × 2,97× (4 Prozesse parallel) ≈ 39× gesamt
Trainer:  711.000 Zeichen bei Vokabular 4096 — 28,5 s statt >5 Minuten
```

**Ergebnis: 384.762.231 echte Token** aus 2.119.719 TinyStories-Geschichten
(alle 4 Shards), über zwei Server verteilt geladen (dieser Server + ein
zweiter mit mehr freier Platte) — über dem für ein 17M-Modell
Chinchilla-optimalen Ziel von 340 Mio. Token.

```bash
python daten/vortokenisiere.py --hoechstens-token 340000000
```

## Stufe 3 — der Eval-Harness, und ein Fund dabei

Vier Tore, `bewertung/tore.py` + `bewertung/importtest.py`:

| Tor | Prüfung |
|---|---|
| 1 | Gültiges JSON? |
| 2 | Struktur korrekt + **echte** Node-Typen (gegen 825 aus der lokalen n8n-Installation extrahierte Typen — `bewertung/extrahiere_node_typen.py`, siehe Stufe 4) |
| 3 | Verbindungen konsistent? |
| 4 | Importiert das echte, installierte n8n (2.25.6) den Workflow wirklich? |

**Fund beim Bauen:** `n8n import:workflow` prüft nicht, ob die verwendeten
Node-Typen überhaupt existieren — ein Workflow mit einem erfundenen Typ
wurde anstandslos importiert. Tor 4 allein hätte also genau die Fälle
übersehen, in denen ein großes Modell am ehesten halluziniert. Bewiesen mit
einem Testfall: Tor 4 lässt ihn durch, Tor 2 verwirft ihn korrekt. Beide
Tore sind nötig, keines ist redundant.

```bash
python bewertung/vergleich.py --kandidaten <antworten.jsonl> --out <ergebnis.json>
```

### Fünf Modelle, identischer Prompt, alle vier Tore (Stand 10.09.2026)

15 handgeschriebene Test-Instruktionen (`bewertung/instruktionen.jsonl`),
derselbe Systemprompt (`bewertung/systemprompt.txt`) für alle, kostenlose
Modelle über OpenRouter, alle vier Tore inklusive echtem n8n-Import
(`bewertung/sammle_antworten.sh <modell>` → `bewertung/vergleich.py`):

| Modell | n | Tor 1 | Tor 2 | Tor 3 | Tor 4 | Gültig | 95-%-KI |
|---|---|---|---|---|---|---|---|
| Nemotron-3-Ultra 550B | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |
| DeepSeek-V4-Flash | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |
| Llama-4-Maverick | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |
| Gemma-4-31B-it | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |
| MiniMax-M3 | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |

Rohdaten und Ergebnisse liegen versioniert (`bewertung/nemotron_antworten.jsonl`,
`bewertung/gratis_modelle_antworten.jsonl`, `bewertung/ergebnisse/*.json`) —
jede Zahl zeigt auf einen nachvollziehbaren Lauf, 75 echte Importe in n8n.

**Was fünfmal 100 % wirklich bedeutet — Deckeneffekt, nicht Gleichstand:**
Der Systemprompt gibt bewusst eine **enge Auswahl von 36 gängigen
Node-Typen** vor, nicht die vollen 825. In dieser Form ist die Aufgabe für
aktuelle Modelle offenbar gelöst — fünf Modelle von 31B bis 550B Parametern
liefern ohne einen einzigen Fehler importierbare Workflows. Ein Benchmark,
auf dem alle 100 % erreichen, **unterscheidet nichts**. Die Messung ist damit
nicht wertlos, sie hat eine klare Aussage: *der leichte Modus ist zu leicht.*
Der nächste Lauf braucht die schwere Einstellung — freie Wahl aus allen 825
Typen oder gar keine Liste — bevor ein Vergleich mit dem eigenen kleinen
Modell (Stufe 4) etwas aussagen kann. Der GPT-4/Claude-Lauf (~5 € API-Kosten)
lohnt erst dann.

**Zweiter Fund, diesmal im eigenen Harness:** beim ersten Lauf mit vier
Modellen in einer Datei zeigte die Tabelle nur *ein* Modell mit n = 15
statt vier mit n = 60. Ursache: alle Modelle bekamen dieselben
Instruktions-IDs (`i01`…`i15`), und `vergleich.py` benutzte die ID als
alleinigen Schlüssel — jedes Modell überschrieb das vorige, in der
Ergebnistabelle **und** im Import-Ordner für Tor 4. Die drei überschriebenen
Modelle hätten „importiert" gemeldet, obwohl ihre Workflows nie bei n8n
ankamen. Ein Fehler, der ein falsches 100 % erzeugt, ist der teuerste, den
ein Messgerät haben kann. Behoben (Schlüssel = Modell + ID), Gegenprobe im
Ergebnis-JSON: 60 eindeutige Kandidaten, 60 echte Importe.

## Stufe 4 — erst gemessen, dann gebaut (Stand 10.09.2026)

Bevor die Mutations-Pipeline für das Workflow-Modell entsteht, war eine
Frage offen: *„Viele echte n8n-Vorlagen nutzen Node-Typen außerhalb unserer
Liste — wie viele?"* Bis dahin eine Vermutung aus einer Metadaten-Stichprobe.
`daten/vorlagen_messen.py` beantwortet sie gegen den vollen Bestand: die
2.352 Vorlagen (mit komplettem Workflow-JSON) aus der Datenbank des
`n8n-mcp`-Pakets.

| Typenliste des Validators | Vorlagen, die Tor 2 vollständig annehmen würde |
|---|---|
| vorher: 439 Typen (nur `n8n-nodes-base`) | **682 / 2.352 = 29 %** |
| jetzt: 825 Typen (`extrahiere_node_typen.py`) | **2.012 / 2.352 = 85,5 %** |

**Die alte Liste war nicht nur klein, sie war falsch** — sie hätte echte
Typen als „halluziniert" verworfen:

- **38 % der Vorlagen nutzen `@n8n/n8n-nodes-langchain`** (Agent, LLM-Chat,
  Output-Parser …). Das Paket ist Teil jeder n8n-Installation, stand aber
  nicht in der Liste, weil nur `n8n-nodes-base` extrahiert worden war.
- **601 Vorlagen nutzen `*Tool`-Varianten** (`httpRequestTool`, `gmailTool`,
  `googleSheetsTool` …). Die stehen in **keiner** `known/nodes.json`: n8n
  erzeugt sie zur Laufzeit für jeden Node mit `usableAsTool: true`. Und
  `httpRequestTool` — der häufigste davon, 139 Vorlagen — kommt nicht einmal
  daher, sondern ist in `n8n-core/dist/constants.js` hart verdrahtet.
  Das Skript liest alle drei Quellen aus der **lokalen Installation**, nicht
  aus den Vorlagen — sonst käme die Grundwahrheit aus genau den Daten, die
  sie später bewerten soll.

Gegenprobe, im Skript eingebaut: von 21.721 Typ-Vorkommen der beiden Pakete
in echten Vorlagen deckt die neue Liste **21.713 (99,96 %)** ab. Die
fehlenden 8 sind ein abgeschaffter Node (`start`) und drei Typen aus anderen
n8n-Versionen — im Bericht benannt, nicht stillschweigend weggelassen.

Die restlichen **14,5 % (340 Vorlagen)** nutzen echte Community-Pakete
(`n8n-nodes-mcp`, `@apify/…`, `@tavily/…`, 40+ weitere). Die bleiben draußen:
sie sind lokal nicht installiert, also kann Tor 4 sie nicht importieren —
und ein Validator darf nichts als „echt" annehmen, was er nicht prüfen kann.

Was das für den Bau bedeutet: **2.012 gültige Vorlagen ≈ 7,3 Mio. Token**
Rohmaterial (Median 17 Knoten pro Workflow) — die Schätzung aus dem Plan
(≈ 7 Mio.) hält. Und die `echte_node_typen.json` hat jetzt ein
Erzeuger-Skript im Repo; vorher war sie ein einmal von Hand erzeugtes
Artefakt — dieselbe Fehlerklasse wie der `tokenizer.pkl`, der beim ersten
echten Colab-Lauf fehlte (Commit `3e7d1ce`).

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
