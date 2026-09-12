# AutoLM

**Ein Sprachmodell von Grund auf — von zufälligen Gewichten bis zu einem
Modell, das gegen GPT-4 und Claude auf einer maschinell prüfbaren Aufgabe
gemessen wird.**

Kein fertiges Modell feingetunt. Kein API-Wrapper. Der Transformer, der
Tokenizer, die Trainingsschleife: selbst gebaut, Schritt für Schritt.

> Status: **Stufe 0–3 abgeschlossen, Stufe 4 gebaut und erstmals gemessen (12.09.2026)** — ein 7-Mio.-Parameter-Modell, auf CPU trainiert, erzeugt aus einer Beschreibung in normaler Sprache in 70 % der Fälle (Best-of-4) ein n8n-Workflow-Gerüst, das dieselben Tore passiert wie die Antworten der neun Frontier-Modelle — mit einem erfundenen Node-Typ. Drei Fassungen der Textform an einem Tag, jede aus einer gemessenen Schwäche der vorigen. Was fehlt: GPU, ein größeres Modell, Parameter im Gerüst. Frühere Angabe: — 384,8 Mio. echte
> Trainings-Token vorbereitet (über dem Chinchilla-optimalen Ziel),
> Eval-Harness gegen echtes n8n läuft, **neun Modelle gemessen (fünf gratis,
> vier bezahlt): 60–100 % gültige Workflows, und 89 % aller Fehler sind
> derselbe Fehlertyp** — erfundene Node-Typnamen (`sendEmail` statt
> `emailSend`). Ein kostenloses Modell schlägt dabei zwei bezahlte
> Frontier-Modelle. Genau die Lücke, die ein kleines Spezialmodell schließen
> können sollte. 2.012 gültige Vorlagen ≈ 7,3 Mio. Token für Stufe 4.
> Weiterhin offen: der GPU-Trainingslauf (Stufe 2).

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

**Die Frage, die das Projekt trägt — und die halbe Antwort, die inzwischen
gemessen ist:** wie nah kommt ein kleines, für ~0 € trainiertes
Spezialmodell an große Allzweckmodelle auf dieser engen Aufgabe? Die einzige
Achse, auf der ein kleines Modell plausibel gewinnen kann, ist die
**Gültigkeitsquote** — die Vermutung war, dass große Modelle zuverlässig
Node-Typen erfinden, die es nicht gibt.

**Gemessen (12.09.2026, neun Modelle, 135 Antworten, ohne Typenliste im
Prompt): die Vermutung stimmt, und zwar schärfer als erwartet.** 60–100 %
gültige Workflows — aber **89 % aller Fehlschläge sind genau ein Fehlertyp**:
ein erfundener Bezeichner bei ansonsten korrekter Struktur (`sendEmail`
statt `emailSend`, `hubSpotTrigger` statt `hubspotTrigger`). Struktur und
Verdrahtung sitzen bei ~99 %. Damit ist die Zielmarke für Stufe 4 keine
Vermutung mehr, sondern eine Zahl. Details: [Stufe 3](#schwerer-modus--und-damit-der-fund-der-das-ganze-projekt-begründet).

**Zweite Messung, noch überraschender:** vier bezahlte Frontier-Modelle
(Claude Opus 5, Claude Sonnet 5, GPT-5.6-terra-pro, GPT-5.2-chat) im selben
Test — sie gewinnen **nicht** zuverlässig. Ein kostenloses Modell schlägt
zwei bezahlte; das teure Flaggschiff einer Familie verliert gegen das
günstigere Modell derselben Familie. Nur Claude Sonnet 5 erreicht 100 %.

Offen bleibt der Rest der Achsen (Latenz, Kosten je 1.000 Anfragen,
Offline-Betrieb).

## Aufbau

| | Stufe | Ergebnis |
|---|---|---|
| 1–4 | Backprop · Tokenizer · Attention · Transformer, alles von Hand | ✅ `schritte/01`–`04` |
| **0** | Modell aus 04 herausgelöst, schneller Attention-Pfad, gegen Lehrpfad bewiesen gleich | ✅ `kern/` |
| **1** | Datenpipeline: BPE-Encoder + -Trainer 39× beschleunigt, **384.762.231 echte TinyStories-Token** vortokenisiert (über dem 340M-Chinchilla-Ziel) | ✅ `kern/bpe_*`, `daten/vortokenisiere.py` |
| **2** | Absturzsicheres Checkpointing (echter Kill-und-Resume-Beweis) + Trainingsloop fertig. **GPU-Lauf selbst noch offen** — `kern/gpu_bootstrap.sh` startet ihn auf einer Miet-GPU ohne Browser | 🔶 Infrastruktur fertig, Training offen |
| **3** | **Eval-Harness gegen echtes n8n 2.25.6** — vier Tore, Vergleichs-Orchestrator, **neun Modelle gemessen (leicht + schwer, gratis + bezahlt), 2,41 $ tatsächliche API-Kosten** | ✅ `bewertung/` |
| **4** | Workflow-Modell: Kurzschrift, Mutations-Pipeline (jede Mutante durch den Validator), Split nach Vorlagen-ID mit hartem Leck-Abbruch, eigener Tokenizer, **drei Fassungen auf CPU trainiert und auf 98 ungesehenen Vorlagen gemessen** | 🔶 `workflow/` — gebaut und gemessen (7M, CPU); GPU-Lauf und größeres Modell offen |
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

### Schwerer Modus — und damit der Fund, der das ganze Projekt begründet

Konsequenz aus dem Deckeneffekt: derselbe Lauf nochmal, aber mit
`bewertung/systemprompt_schwer.txt` — **gar keine Typenliste**. Das Modell
muss die echten n8n-Typnamen selbst kennen. Fünf Modelle, dieselben 15
Instruktionen, 75 Antworten, wieder alle vier Tore:

| Modell | n | Tor 1 (JSON) | Tor 2 (Typen) | Tor 3 (Graph) | Tor 4 (Import) | Gültig | 95-%-KI |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 15 | 100 % | 80 % | 100 % | 80 % | **80 %** | [60 %, 100 %] |
| Nemotron-3-Ultra 550B | 15 | 100 % | 73 % | 100 % | 73 % | **73 %** | [47 %, 93 %] |
| Gemma-4-31B-it | 15 | 100 % | 67 % | 100 % | 67 % | **67 %** | [40 %, 87 %] |
| Llama-4-Maverick | 15 | 100 % | 60 % | 100 % | 60 % | **60 %** | [33 %, 87 %] |
| MiniMax-M3 | 15 | 93 % | 60 % | 93 % | 60 % | **60 %** | [33 %, 87 %] |

**Der Benchmark unterscheidet jetzt** — 60 % bis 80 % statt fünfmal 100 %.
Aber das Interessante ist nicht die Rangfolge, sondern **woran** sie
scheitern. Von 24 Fehlschlägen sind **23 (96 %) ein und derselbe Fehlertyp**:

| erfunden | Vorkommen | tatsächlich heißt es |
|---|---|---|
| `n8n-nodes-base.sendEmail` | 17 | `n8n-nodes-base.emailSend` |
| `n8n-nodes-base.imapTrigger` | 4 | `n8n-nodes-base.emailReadImap` |
| `n8n-nodes-base.imap` | 1 | `n8n-nodes-base.emailReadImap` |
| `n8n-nodes-base.hubSpotTrigger` | 1 | `n8n-nodes-base.hubspotTrigger` |

Das sind **vier** verschiedene Fehlgriffe, mehr nicht. `sendEmail` statt
`emailSend` — dieselben zwei Wörter, vertauscht. `hubSpotTrigger` statt
`hubspotTrigger` — **ein einziger Großbuchstabe**.

**Warum das die Kernthese des Projekts stützt:** Tor 1 (gültiges JSON) und
Tor 3 (konsistenter Verbindungsgraph) liegen bei ~99 %. Die Modelle verstehen
die Aufgabe, bauen die richtige Struktur und verdrahten die Knoten korrekt.
Sie scheitern ausschließlich am **exakten Abruf eines geschlossenen
Vokabulars** — und genau das ist keine Denkleistung, sondern Auswendiglernen.
Ein 550-Milliarden-Parameter-Modell hat keinen strukturellen Vorteil beim
Auswendiglernen von 825 Bezeichnern; ein 17-Mio.-Parameter-Modell, das auf
nichts anderem als echten n8n-Workflows trainiert wurde, sollte hier
prinzipiell **100 %** erreichen können.

Das ist die Achse, auf der ein winziges Spezialmodell ein großes
Allzweckmodell schlagen kann — und sie ist jetzt **gemessen statt behauptet**.
Genau diese Behauptung stand ursprünglich ungedeckt im README und wurde vor
der Messung zu einer Frage umformuliert (siehe *Nicht-Ziele*); sie hat jetzt
eine Zahl, gegen die Stufe 4 antreten muss: **68 % über alle fünf Modelle.**

Rohdaten: `bewertung/schwer_antworten.jsonl`,
`bewertung/ergebnisse/schwer-2026-09-12.json`.

### Und jetzt die bezahlten Spitzenmodelle — sie gewinnen nicht

Derselbe schwere Modus, dieselben 15 Instruktionen, vier kostenpflichtige
Frontier-Modelle über OpenRouter. **Tatsächliche Kosten: 2,41 $** (am
Guthaben vorher/nachher gemessen, nicht geschätzt).

| Modell | Preis (in/out je Mio. Token) | Gültig | 95-%-KI |
|---|---|---|---|
| **Claude Sonnet 5** | 2 $ / 10 $ | **100 %** | [100 %, 100 %] |
| GPT-5.6-terra-pro | 2 $ / 12 $ | 80 % | [60 %, 100 %] |
| **DeepSeek-V4-Flash** | **kostenlos** | **80 %** | [60 %, 100 %] |
| Nemotron-3-Ultra 550B | kostenlos | 73 % | [47 %, 93 %] |
| Claude Opus 5 | 5 $ / 25 $ | 67 % | [40 %, 87 %] |
| Gemma-4-31B-it | kostenlos | 67 % | [40 %, 87 %] |
| GPT-5.2-chat | 1,75 $ / 14 $ | 60 % | [33 %, 87 %] |
| Llama-4-Maverick | kostenlos | 60 % | [33 %, 87 %] |
| MiniMax-M3 | kostenlos | 60 % | [33 %, 87 %] |

**Zwei Ergebnisse, die man nicht erwarten würde:**

1. **Bezahlen hilft nicht zuverlässig.** Ein *kostenloses* Modell
   (DeepSeek-V4-Flash, 80 %) schlägt zwei bezahlte Frontier-Modelle —
   Claude Opus 5 (67 %) und GPT-5.2-chat (60 %). Preis und Gültigkeitsquote
   sind auf dieser Aufgabe **nicht korreliert**.
2. **Das teurere Modell derselben Familie verliert.** Claude Sonnet 5 (2 $/Mio.)
   erreicht 100 %, Claude Opus 5 (5 $/Mio., das Flaggschiff) nur 67 % — bei
   identischem Prompt. Opus erfand unter anderem `readWriteFromFile`
   (echt: `readWriteFile`) und `csv` (echt: `spreadsheetFile`).

**Über alle neun Modelle, 135 Antworten:** 38 Fehlschläge, davon **34 (89 %)
ausschließlich am Node-Typnamen** — bei korrektem JSON und korrektem
Verbindungsgraph. Und es sind nur **sechs** verschiedene erfundene
Bezeichner im ganzen Datensatz, angeführt von 24× `sendEmail`.

**Was das für Stufe 4 bedeutet:** Die Messlatte ist nicht mehr „irgendwas
mit 68 %", sondern **Claude Sonnet 5 mit 100 %**. Ein 17-Mio.-Parameter-
Modell kann das nur schaffen, wenn es das Vokabular wirklich auswendig
kennt — genau das ist die Wette. Und selbst wenn es nur 90 % erreicht, hätte
es vier von neun getesteten Modellen geschlagen, darunter zwei bezahlte.

Rohdaten: `bewertung/premium_antworten.jsonl`,
`bewertung/ergebnisse/premium-schwer-2026-09-12.json`.

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

## Stufe 4 — das Workflow-Modell: gebaut, trainiert, gemessen (12.09.2026)

Code in `workflow/` (neun Dateien, je eine Aufgabe), 16 Tests in
`workflow/test_workflow.py`. Alles Erzeugte liegt unter `daten/workflow/`
(gitignored, in drei Minuten neu erzeugbar) — die Zahlen daraus in
`bewertung/ergebnisse/stufe4-*.json`.

```bash
python workflow/katalog.py            # 825 Typen mit Anzeigename/Kategorie aus der lokalen DB
python workflow/baue_datensatz.py     # Vorlagen → Split → Mutanten → Tokenizer → uint16-Strom
python -m unittest workflow.test_workflow
python kern/trainiere.py --daten daten/workflow/train.bin --meta daten/workflow/meta.json \
    --val-daten daten/workflow/val.bin --checkpoint-ordner daten/workflow/ckpt \
    --dim 256 --schichten 6 --koepfe 4 --block 512 --batch 16 --schritte 800
python workflow/erzeuge.py --checkpoints daten/workflow/ckpt --test daten/workflow/test.jsonl
```

### Drei Entscheidungen, die man kennen muss

**1. Das Modell lernt eine Kurzschrift, nicht das rohe JSON.** Gemessen
über die gültigen Vorlagen: n8n-JSON hat selbst *ohne* Parameter median
3.156 Zeichen (p90 7.522) — bei Kontext 512 sähe ein kleines Modell die
Hälfte der Workflows nie am Stück. Und fast alle dieser Zeichen sind
Wiederholung (Anführungszeichen, UUIDs, Positionen), an der laut Stufe 3
kein einziges Frontier-Modell scheitert. Die Kurzschrift behält, woran sie
scheitern — den exakten Typnamen — und die Struktur:

```
wf Lead-Erinnerung
n1 n8n-nodes-base.formTrigger@2.2 Formular
n2 n8n-nodes-base.wait@1.1 Warte 24h
n3 n8n-nodes-base.emailSend@2.1 Erinnerung senden
n1 > n2
n2 > n3
```

`kurzschrift.rendere()` rechnet daraus deterministisch das vollständige
n8n-JSON (IDs per uuid5, Positionen im Raster, `parameters: {}`), und
dieses JSON geht durch **dieselben vier Tore** wie die Antworten der neun
Frontier-Modelle. Rundlauf über alle 1.978 Vorlagen: Typen und Kanten
exakt erhalten, 0 Fehler. Median 802 Zeichen statt 3.156; nach BPE median
118 Token, p90 290.

**2. Die Typnamen werden ausgeschrieben, nicht als ein Sondertoken kodiert.**
Das wäre die naheliegende Abkürzung: 825 Sondertoken, und `sendEmail` statt
`emailSend` ist konstruktionsbedingt unmöglich. Genau deshalb nicht — es
hätte die Kernmessung entwertet. 89 % der Frontier-Fehler sind
Rechtschreibung in einem geschlossenen Vokabular; ein Modell, das die
Typen Byte für Byte schreiben muss, muss sie *wirklich* auswendig können.
Der eigene BPE-Tokenizer (4.096 Token, 7,6 Zeichen/Token auf dem Korpus)
lernt häufige Typen als wenige Token, seltene bleiben Stückwerk. Die
Sondertoken-Variante bleibt als Ablationsarm für Stufe 6 offen.

**3. Split nach Vorlagen-ID, nicht nach Zeile — mit hartem Abbruch.** 90 %
Training, 5 % Validierung, 5 % Test, gezogen über die Vorlagen-ID. Mutanten
entstehen **nur** im Trainings-Split; Validierung und Test sind
unveränderte echte Vorlagen. `pruefe_split()` beendet den Bau, wenn eine ID
in zwei Splits liegt oder eine Test-Kurzschrift wortgleich im Training
vorkommt — eine Warnung würde überlesen. Das ist dieselbe Fehlerklasse wie
das `ANSAGE:`-Leck im Anruf-Klassifikator von `tgcrm`: dort stand die
Antwort im Eingabetext, hier stünde die Mutante im Training und ihr Original
im Test.

### Der Datensatz — Generator und Bewerter sind dasselbe Orakel

| | |
|---|---|
| gültige Vorlagen (ohne Haftnotizen, ohne kaputte Exporte) | **1.978** → 1.782 / 98 / 98 |
| Mutanten im Training (4 Operationen, jede durch `tore.py`) | **35.640** |
| Trainingsbeispiele (3 Instruktions-Varianten je Original + Mutanten) | **40.986** |
| Trainings-Token | **6,54 Mio.** |
| Instruktionen | deterministisch aus `use_cases` + Katalog-Anzeigenamen, kein LLM, 0 € |

Die vier Mutationen (`workflow/mutationen.py`): Knoten permutieren (Kanten
hängen an Namen — der Graph bleibt gleich, die Nummerierung nicht),
Knoten auf ihren Katalog-Anzeigenamen umbenennen (die Brücke Typ ↔
„Send Email"), **Typ gegen einen gleichartigen tauschen** (gleiche
Kategorie, gleiche Auslöser- und Tool-Eigenschaft, gleiches Paket — der
Weg zu allen 825 Typen statt der 60 häufigsten), Blattknoten entfernen.
Kanten werden bewusst **nicht** erfunden: eine `ai_languageModel`-Kante von
Slack in einen Agenten wäre für Tor 3 gültig und inhaltlich Unsinn; was der
Validator nicht prüfen kann, darf der Generator nicht zufällig erzeugen.

Was das Geruest **nicht** enthält: Parameter. Ein gerenderter Workflow ist
importierbar, aber nicht konfiguriert (`httpMethod`, `channel`, `path` fehlen).
Das ist eine bewusste Grenze dieser Fassung, keine Nebensache — sie steht
hier, damit niemand „100 % gültig" für „fertig einsetzbar" hält.

### Kontrolle vor dem Modell: was Abschreiben schafft

`workflow/baseline_naechster_nachbar.py`: für jede Test-Instruktion das
Trainingsbeispiel mit der größten Wort-Überlappung, dessen Kurzschrift
unverändert abgegeben. Gültigkeit per Konstruktion 100 % (sagt nichts);
**Typen-Überdeckung gegen die Referenz (Jaccard): 0,51**. Unter diesem Wert
hat ein Modell nichts gelernt, was Nachschlagen nicht könnte
(`bewertung/ergebnisse/stufe4-baseline-nn-2026-09-12.json`).

### Drei Fassungen an einem Tag — jede aus einer gemessenen Schwäche der vorigen

Ein 7-Mio.-Parameter-Modell (dim 256, 6 Schichten, 4 Köpfe, Kontext 512),
trainiert auf CPU (4 Kerne, 3,5 s/Schritt, 2.400 Schritte ≈ 2–3,7 h je Lauf),
identische Konfiguration für alle drei Fassungen — es ändert sich **nur die
Textform**, die das Modell lernt. Bewertet auf 98 Vorlagen, die das Modell
nie gesehen hat, Best-of-4 gegen den Validator (Temperatur 0,7, top-k 40).

| Fassung | was das Modell schreibt | Val-Verlust @2400 | gültig@1 | gültig@4 | Typen-Jaccard | erfundene Typen |
|---|---|---|---|---|---|---|
| 1 — mit Knotennamen | `n3 n8n-nodes-base.emailSend@2.1 Erinnerung senden` | 3,83 | 20 %* | 40 %* | 0,26* | **0** |
| 2 — ohne Namen | `n3 n8n-nodes-base.emailSend@2.1` | 3,31 | 19 % | 42 % | 0,35 | 1 (`boxTool`) |
| 3 — ohne Namen, ohne Nummern, Verlust nur auf der Kurzschrift | `n8n-nodes-base.emailSend@2.1` | 4,62† | 36 % | 70 % | 0,41 · Kanten 0,04 | 1 Formatfehler (`set@3.3@…`) |
| Kontrolle: nächster Nachbar | Trainingsbeispiel abschreiben | — | 100 % (Konstruktion) | — | 0,52 | 0 |

† Fassung 3 misst den Verlust nur auf Kurzschrift-Token (Maske) — nicht mit den
Zeilen darüber vergleichbar, dort zählt die leichter vorhersagbare Instruktion mit.

\* Fassung 1 bei Schritt 2.400 nur auf n = 30 gemessen (Erzeugung ohne
KV-Cache ist bei den langen Fassung-1-Ausgaben zu langsam für alle 98).
Rohdaten: `bewertung/ergebnisse/stufe4-v{1,2,3}-*.json`.

**Was in allen drei Läufen gleich war — und der eigentliche Befund:** in
**keiner** Bewertung, auch nicht in den unlesbarsten Ausgaben, hat das
Modell je einen Node-Typ erfunden. `sendEmail`, `imapTrigger`,
`hubSpotTrigger` — die Fehler, an denen 89 % der Frontier-Fehlschläge
hängen — kommen bei einem 7-Mio.-Modell praktisch nicht vor: über alle
Bewertungen (rund 700 erzeugte Workflows) genau **ein** erfundener Typ,
`boxTool` — `box` gibt es, nur ohne Tool-Variante. Selbst dieser Fehler ist
ein Regelfehler (Tool-Suffix auf einen Typ, der keins hat), kein
Buchstabendreher. Das Vokabular auswendig zu können ist die leichte Hälfte der Aufgabe.

**Woran es stattdessen scheitert, Fassung für Fassung:**

- **Fassung 1:** die frei getippten Knotennamen („Answer- Get Flag", „GET -
  DB - STE KE KE KE …") sind Rauschen, das kein Modell dieser Größe
  vorhersagen kann. Greedy-Dekodierung lief darin in Wiederholungsschleifen
  und verlor die Nummerierung; erst Sampling brachte 20 % / 40 %. Die Namen
  tragen für Tor 1–4 nichts — weg damit, `rendere()` setzt den
  Katalog-Anzeigenamen.
- **Fassung 2:** ohne Namen sank der Val-Verlust schneller (3,31 statt 3,83)
  und die Typen-Überdeckung stieg auf 0,35 — aber auch bei Schritt 2.400
  scheitern 57 von 98 Ausgaben, davon 28 an einer **doppelten Knotennummer**
  und 9 an einer Kante zu einer Nummer, die es nicht gibt; die Kanten-
  Überdeckung liegt bei 0,03. Schon bei Schritt 800 war das Muster sichtbar — `n7 … n7 … n7`, `n11 n12 n11`: das Modell
  kopiert die letzte Nummer statt hochzuzählen. Und es ignorierte die
  Instruktion: für „Extract from File → Set → Code" kam eine Agenten-Kette.
  Zwei Befunde, zwei Änderungen: Knotenzeilen ohne Nummer (die Nummer ist
  die Zeilenposition; Kanten behalten absolute Nummern, ein Zählfehler dort
  ist ein falscher Verweis, kein Abbruch — deshalb misst Fassung 3 zusätzlich
  die **Kanten-Überdeckung**), und der Verlust zählt nur noch auf der
  Kurzschrift, nicht auf der Instruktion (`--maske`; vorher kam ein Drittel
  des Verlusts aus dem Vorhersagen der Instruktion selbst).
- **Und ein Fund im eigenen Werkzeug:** beim ersten Bau von Fassung 2 hat
  der harte Split-Check abgebrochen — 7 Validierungs-Kurzschriften standen
  wortgleich im Training. Ohne Namen sind verschiedene Vorlagen oft
  strukturgleich (1.978 Vorlagen = 1.927 verschiedene Texte). Der Split
  zieht seitdem je Text-Gruppe, und Trainingszeilen, die textgleich mit
  Validierung/Test sind, fliegen raus (3). Ein Leck, das eine Warnung
  überlesen hätte — der Abbruch hat es erzwungen.


### Gegen die Frontier-Modelle — dieselben 15 Instruktionen, alle vier Tore, echter n8n-Import

Fassung 3, Schritt 2.400, die 15 deutschen Instruktionen aus Stufe 3 (die
das Modell nie gesehen hat, und die anders formuliert sind als alles im
Training), Ausgabe durch `bewertung/vergleich.py` — also derselbe Weg
inklusive Tor 4 wie bei den neun großen Modellen:

| | n | Tor 1 | Tor 2 | Tor 3 | Tor 4 | gültig | 95-%-KI |
|---|---|---|---|---|---|---|---|
| autolm 7M, **ein Versuch** (k = 1) | 15 | 67 % | 47 % | 67 % | 47 % | **47 %** | [20 %, 73 %] |
| autolm 7M, Best-of-8 gegen den Validator | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |

**Die ehrliche Lesart, beide Zeilen:** die Frontier-Modelle hatten *einen*
Versuch — die Vergleichszahl ist also **47 %**, und damit liegt das
7-Mio.-Modell **unter allen neun** (60–100 %). Die Best-of-8-Zeile ist kein
Sieg, sondern die Messung aus dem Plan (Gültigkeit@1 gegen Gültigkeit@k):
mit dem Validator in der Schleife — und der ist zugleich das Orakel, das
die Trainingsdaten erzeugt hat — kommen alle 15 durch, inklusive Import in
das echte n8n 2.25.6. Das ist die Bauart, die ein kleines Modell praktisch
nutzbar macht, und sie steht den großen Modellen genauso offen.

**Und der Befund, der die eigene These einschränkt:** auf diesen 15
*fremd formulierten* Instruktionen hat das Modell **dreimal einen Typ
erfunden** — `s@2Tool`, `herMapTool`, `googleAdsTrigger` (es gibt
`googleAds`, aber keinen Trigger). Auf den 98 Test-Vorlagen, deren
Instruktionen nach derselben Schablone gebaut sind wie das Training, war es
einmal in rund 400 Ausgaben. Auswendigkönnen reicht also **innerhalb** der
Verteilung; sobald die Formulierung fremd wird, rät auch ein Modell, das das
Vokabular kennt. Genau die Lücke, die Stufe 5 (eingeschränkte Dekodierung
über den 825 echten Typen) schließen würde — jetzt mit einer gemessenen
Fehlerquote, gegen die sie antreten muss.

### Was das Modell heute nicht kann — gemessen, nicht vermutet

- **Es schlägt Nachschlagen nicht.** Typen-Überdeckung gegen die Referenz
  0,41 (Fassung 3), die Nächster-Nachbar-Kontrolle liegt bei 0,52; bei den
  Kanten 0,04 gegen 0,20. Das Modell erzeugt gültige Gerüste, aber noch
  nicht *die* Gerüste, die die Instruktion meint. Ein Modell, das unter der
  Abschreib-Kontrolle liegt, hat auf dieser Achse nichts gelernt, was ein
  Index nicht könnte — das steht hier, damit es nicht in einer Fußnote
  verschwindet.
- **Es überpasst sich an die Mutanten.** Trainingsverlust 3,19, Validierung
  4,62 bei Schritt 2.400, Plateau ab 1.600 — 20 Mutanten je Vorlage sind
  offenbar zu ähnlich. Weniger Mutanten mit mehr Varianz, oder mehr echte
  Vorlagen, ist der nächste Hebel, nicht mehr Schritte.
- **Kanten sind das schwächste Glied.** Absolute Nummern in `n3 > n5`
  verlangen Zählen, und Zählen ist genau das, was das Modell in Fassung 2
  nicht konnte. Die Kanten-Überdeckung von 0,04 sagt: die Knoten stimmen
  oft, ihre Verdrahtung fast nie. Eine relative Kantenschreibweise
  (Verweis „k Zeilen darüber") wurde gemessen und verworfen — 26 % der
  Vorlagen haben Zyklen, und nur 59 % der Kanten liegen innerhalb von drei
  Zeilen.
- **7 Mio. Parameter auf CPU** sind die Untergrenze, nicht die Wahl. Die
  14-Mio.-Konfiguration braucht 6,3 s/Schritt auf vier Kernen; der
  GPU-Lauf (Stufe 2) hängt weiter an einem Miet-GPU-Zugang.

Alle Zahlen: `bewertung/ergebnisse/stufe4-*.json`, die 15 gerenderten
Kandidaten in `bewertung/eigenes_modell_v3_antworten.jsonl` (Best-of-8) und
`bewertung/eigenes_modell_v3_k1_antworten.jsonl` (ein Versuch).

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
