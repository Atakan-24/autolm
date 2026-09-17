# AutoLM

**Ein Sprachmodell von Grund auf — von zufälligen Gewichten bis zu einem
Modell, das gegen GPT-4 und Claude auf einer maschinell prüfbaren Aufgabe
gemessen wird.**

Kein fertiges Modell feingetunt. Kein API-Wrapper. Der Transformer, der
Tokenizer, die Trainingsschleife: selbst gebaut, Schritt für Schritt.

> Status: **Stufe 0–3 abgeschlossen, Stufe 4 gebaut und erstmals gemessen (12.09.2026)** — ein 7-Mio.-Parameter-Modell, auf CPU trainiert, erzeugt aus einer Beschreibung in normaler Sprache in 70 % der Fälle (Best-of-4) ein n8n-Workflow-Gerüst, das dieselben Tore passiert wie die Antworten der neun Frontier-Modelle — mit einem erfundenen Node-Typ. Drei Fassungen der Textform an einem Tag, jede aus einer gemessenen Schwäche der vorigen. Was fehlt: GPU, ein größeres Modell, Parameter im Gerüst. **Stufe 5 (17.09.2026): eingeschränkte Dekodierung gebaut und gemessen** — eine Logit-Maske macht ungültige Node-Typen und ungültige Grammatik beim Schreiben unwählbar, statt die fertige Antwort zu reparieren. Auf den 15 fremd formulierten Instruktionen, **ein** Versuch je Instruktion und ohne Validator in der Schleife, steigt die Quote durch alle vier Tore inklusive echtem n8n-Import von **20 % auf 100 %** (über drei Zufallsströme: 45 von 45, p = 6 × 10⁻¹¹); erfundene Typen 11 gegen 0. Auf den 98 Testfällen mit Referenz: gültig@1 von 36 % auf 97 %, und die Typen-Überdeckung **steigt** dabei (0,26 → 0,35) statt zu leiden. Damit erreicht das 7-Mio.-Modell den Wert des besten der neun Frontier-Modelle auf dieser Aufgabe — bei der Gültigkeit, nicht bei der Treffsicherheit. **Nachtrag 17.09.2026:** der beste Validierungs-Checkpoint wird jetzt mitgespeichert und wurde gegen den Endstand gemessen — kein Unterschied, den 98 Testfälle auflösen könnten; ein bloß anderer Zufallsstrom bei bitgleichen Gewichten bewegt die Zahlen genauso stark. Der Prüfstand ist damit vermessen, nicht das Modell. Frühere Angabe: — 384,8 Mio. echte
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
| **5** | **Eingeschränkte Dekodierung**: Logit-Maske über den 825 echten Node-Typen und der Kurzschrift-Grammatik, hinter `--beschraenkt`. Auf den 15 fremd formulierten Instruktionen, ein Versuch, alle vier Tore: **20 % → 100 % gültig**, erfundene Typen 1 → 0 | ✅ `workflow/beschraenkt.py` — 22 Tests; n=98-Messung der Treffsicherheit läuft |
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


### Die Gegenprobe zur Überanpassung — geteiltes Ergebnis (B8)

V3 überpasst sich: Trainingsverlust 3,19 gegen Validierung 4,62, Plateau ab
Schritt 1.600. Die naheliegende Erklärung war die Datenerzeugung — 20
Mutanten je Vorlage, die einander zu ähnlich sind. Also ein Lauf, der genau
das ändert und **sonst nichts**: gleicher Seed, gleiches Modell, gleiches
Schrittbudget, gleicher Split.

| | V3 | B8 |
|---|---|---|
| Mutanten je Vorlage | 20 | **8**, gestaffelt (1–3 Operationen) |
| Instruktions-Varianten je Original | 3 | **6** |
| Trainings-Token | 2,56 Mio. | **1,60 Mio.** |
| bester Validierungsverlust | **4,55** (Schritt 1.850) | 4,59 (Schritt 1.150) |
| Validierung am Ende (Schritt 2.350) | **4,62** | 5,11 |
| Abstand Training↔Validierung | **1,42** | 2,70 |
| lesbare Ausgabe (Tor 0) | 71 % | **84 %** |
| gültig@1 | 36 % | **41 %** |
| gültig@4 | 70 % | **84 %** |
| ins Token-Limit gelaufen | 7 % | **0 %** |
| **Typen-Jaccard** | **0,41** | 0,34 |
| Kanten-Jaccard | 0,04 | 0,04 |
| erfundene Typen | 1 Formatfehler | **0** |

**Das Ergebnis geht auseinander, und genau das ist der Befund.** B8 schreibt
**mehr gültige** Workflows — 84 % statt 70 % bestehen alle Tore, kein einziger
läuft ins Token-Limit, kein einziger Typ ist erfunden. Und B8 trifft die
**Sache schlechter**: Typen-Jaccard 0,34 statt 0,41.

Die Erklärung, die zu beidem passt: weniger Mutanten heißt weniger gesehene
Typen-Vielfalt. Das Modell weicht auf die Handvoll Typen aus, die es sicher
kann — das ergibt saubere, kurze, gültige Workflows, die aber häufiger am
Gewünschten vorbeigehen. **Gültigkeit und Treffsicherheit sind hier zwei
Achsen, keine eine.** Wer nur die Gültigkeitsquote berichtet, verkauft einen
Rückschritt als Fortschritt.

**Die Überanpassungs-Hypothese ist widerlegt, und zwar in die andere
Richtung.** Weniger, stärker gestaffelte Mutanten haben den Abstand zwischen
Trainings- und Validierungsverlust nicht gedämpft, sondern von 1,42 auf
2,70 **verdoppelt**, und B8 erreicht sein Optimum schon bei Schritt
1.150 statt 1.850. Nicht die Ähnlichkeit der Mutanten war das Problem,
sondern die **Menge**: 1,60 Mio. Token reichen diesem Modell nicht, auch wenn
sie vielfältiger sind.

Gegen die eigene Erwartung, deshalb hier: über die ersten rund 600 Schritte
war B8 an **jedem** Messpunkt besser. Die zusätzlichen Instruktions-Varianten
helfen also messbar — dem Lauf geht danach nur der Stoff aus.

**Urteil nach der vorab festgelegten Regel: V3 bleibt die Basis.** Sie stand vor dem
Lauf in `bewertung/ergebnisse/stufe4-b8-konfiguration-2026-09-12.json`
(„nur als besser dokumentiert, wenn die Holdout-Messung mindestens V3 bei
Gültigkeit@1 **und** Typen-Jaccard erreicht und die Validierungskurve nicht
schlechter endet"). Von drei Bedingungen ist eine erfüllt. Genau dafür
schreibt man die Regel vorher auf: die Gültigkeitsquote allein hätte eine
Erfolgsmeldung hergegeben.

**Was daraus folgt — und ein Fund über das eigene Werkzeug:** beide Modelle
wurden bei Schritt 2.400 bewertet, obwohl **beide** ihr Optimum vorher hatten
(V3 bei 1.850, B8 bei 1.150). Der Checkpointer hält bewusst nur zwei
rotierende Stände — der beste Stand ist damit nicht mehr bewertbar, er ist
überschrieben. Der nächste Schritt ist deshalb nicht noch eine Datenvariante,
sondern: den Stand mit dem besten Validierungsverlust mitspeichern und **den**
messen. Erst danach mehr Daten, und die kommen nicht aus mehr Mutanten,
sondern aus mehr echten Vorlagen.
**Gemacht, 17.09.2026 — siehe nächster Abschnitt.** Das Ergebnis war nicht
die erwartete Antwort, sondern eine Rauschgrenze.

### Der beste Stand gemessen — und die Rauschgrenze des Prüfstands (17.09.2026)

Der Checkpointer hält seit `7b27da5` zusätzlich den Stand mit dem niedrigsten
Validierungsverlust fest (`ckpt_best.pt`, atomar, mit Prüfsumme, nur bei
echter Verbesserung; `erzeuge.py --bestes` lädt ihn). Ein neuer Lauf mit der
unveränderten V3-Konfiguration — gleicher Seed, gleiche Daten, gleiches
Schrittbudget — hat zuerst bewiesen, dass die Ergänzung das Training nicht
berührt: die Validierungskurve trifft die V3-Zahlen auf die Stelle (4,55 bei
Schritt 1.850, 4,62 bei 2.350), und die 72 Gewichtstensoren des neuen
2.400er-Stands sind **bitgleich** mit dem alten (größte Abweichung 0). Dann
beide Stände desselben Laufs auf denselben 98 Test-Vorlagen, Best-of-4,
Temperatur 0,7, top-k 40 — **gepaart** über die Fall-ID,
exakter McNemar-Test, Bootstrap-Intervalle
(`bewertung/vergleiche_checkpoints.py`):

| | Best (Schritt 1.850, Val 4,55) | Ende (Schritt 2.400, Val 4,62) | Differenz, 95-%-KI | p |
|---|---|---|---|---|
| lesbar (Tor 0) | 81 % | 73 % | +7 pp [−4, +18] | 0,30 |
| gültig@1 | 28 % | 37 % | −9 pp [−21, +3] | 0,20 |
| gültig@4 | 79 % | 73 % | +5 pp [−6, +16] | 0,47 |
| Typen-Jaccard, unbedingt (ungültig zählt 0) | 0,27 | 0,29 | −0,01 [−0,07, +0,05] | 0,59 |
| Kanten-Jaccard, unbedingt | 0,02 | 0,04 | −0,01 | 0,20 |
| erfundene Typen | 2 | 0 | | |

Elf Tests, **keiner** signifikant, kleinstes p 0,098. Und „nicht signifikant"
heißt hier nicht „gleich gut": die Intervalle sind so breit, dass ein
Unterschied von zehn Prozentpunkten in beide Richtungen darin Platz hat.

**Die Kontrollmessung, die das einordnet:** der alte V3-Stand (Schritt 2.400)
gegen den neuen 2.400er-Stand — **bitgleiche Gewichte**, nur der Zufallsstrom
beim Sampling ist ein anderer. (Und ein Fund nebenbei, `f75f979`: `--seed`
wirkte beim Laden des letzten Standes gar nicht, weil der Checkpointer den
Trainings-RNG wiederherstellt und den Seed überschrieb — die beiden Läufe
unterschieden sich im Zufallsstrom trotzdem, nur nicht aus dem Grund, den die
Befehlszeile behauptete. Seit `f75f979` gilt der Seed für beide Ladewege.)
Unterschiede allein aus dem Zufallsstrom: Tor 0 2 pp, gültig@4 3 pp,
Typen-Jaccard 0,02, erfundene Typen 1 gegen 0. Die Intervalle dieses Null-Vergleichs sind genauso breit wie die des
echten Vergleichs (`stufe4-rauschgrenze-seed0-gegen-seed1-2026-09-17.json`).

**Befund:** mit 98 Testfällen kann dieser Prüfstand einen Checkpoint-Wechsel
nicht von einem Wechsel des Zufallsstroms unterscheiden. Die Frage „ist der Stand mit dem
besten Validierungsverlust auch der bessere Workflow-Erzeuger?" ist damit
nicht mit Nein beantwortet, sondern **mit diesem Prüfstand nicht
beantwortbar**. Das ist das Ergebnis dieser Messung — und die Zahl, die jede
weitere Änderung schlagen muss: ein Effekt unter rund zehn Prozentpunkten
(bzw. 0,06 Jaccard) ist bei n = 98 unsichtbar.

**Nachtrag zu B8, mit derselben Brille** (`stufe4-b8-gegen-v3-gepaart-2026-09-17.json`,
beide über denselben Ladeweg bewertet): der Gültigkeitsvorsprung von B8 liegt **außerhalb** der
Rauschgrenze — gültig@4 +13 pp [+1, +24], p 0,047; kein Token-Limit-Abbruch
gegen 7 %, p 0,016. Die „schlechtere Treffsicherheit" dagegen hängt am
Maß: der Typen-Jaccard oben (0,34 gegen 0,41) ist nur über die Fälle
gemittelt, in denen die Ausgabe lesbar war — wer mehr lesbare Ausgaben
liefert, wird auf einer anderen Teilmenge gemessen. **Unbedingt** (unlesbar
zählt 0) sind es 0,29 gegen 0,29, Differenz −0,01 [−0,07, +0,05]. Das Urteil
nach der vorab festgelegten Regel bleibt (die Validierungskurve endet
schlechter, das war die dritte Bedingung), aber die Lesart „B8 trifft
schlechter" ist mit den Daten nicht belegt — sie war ein Artefakt des
bedingten Maßes. Deshalb ist die unbedingte Fassung ab jetzt das Hauptmaß.

**Was daraus folgt:** nicht noch ein Lauf, sondern mehr Trennschärfe. Der
Test-Split hat 98 von 1.978 Vorlagen; ein Unterschied von fünf Prozentpunkten
bräuchte grob viermal so viele Fälle. Billiger und sofort möglich: dieselben
98 Fälle mit mehreren Zufallsströmen (`--seed`, seit `f75f979` wirksam) je Stand — das mittelt das
Sampling-Rauschen weg, nicht die Fall-Schwankung. Und der nächste Schritt am
Modell muss ein Effekt sein, der die Grenze sicher überspringt: Stufe 5
(eingeschränkte Dekodierung über den 825 echten Typen und der
Kurzschrift-Grammatik) greift genau die Tor-0-Fehler an, die 20–30 % der
Ausgaben kosten.

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

### Lokale Demo — eine Anweisung hinein, Workflow-JSON hinaus

`workflow/demo.py` ist der kleine, API-freie Einstieg für einen vorhandenen
Checkpoint. Er gibt die erzeugte Kurzschrift, die Tor-1–3-Prüfung und bei
lesbarer Ausgabe das gerenderte n8n-Workflow-JSON aus. `--k` ist transparent
Best-of-k gegen den bestehenden Validator, **keine** Korrektur oder Reparatur
der Modellantwort:

```bash
python workflow/demo.py \
  --checkpoints daten/workflow3/ckpt \
  --tokenizer daten/workflow3/tokenizer.pkl \
  --instruktion "Wenn ein Formular eingeht, warte einen Tag und sende eine Mail" \
  --k 1 --out workflow.json
```

Die Checkpoints und der abgeleitete Datensatz sind bewusst nicht im Git-Repo;
der Befehl ist auf dem dokumentierten Trainingsserver nach einem Lauf direkt
ausführbar. Ein ungültiger Versuch endet mit Exit-Code 1 und bleibt sichtbar,
statt ihn stillschweigend in einen gültigen Workflow umzuschreiben.

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

**Stand 17.09.2026 zu dieser Liste:** Punkt 3 (unlesbare Ausgaben, erfundene
Typen, falsche Kantenverweise) ist mit Stufe 5 erledigt — die Maske macht
diese Fehler beim Schreiben unmöglich. Punkt 1 (Treffsicherheit unter der
Abschreib-Kontrolle) und Punkt 4 (Modellgröße) stehen unverändert; eine Maske
kann erzwingen, dass ein Workflow gültig ist, nicht dass er der richtige ist.

Alle Zahlen: `bewertung/ergebnisse/stufe4-*.json`, die 15 gerenderten
Kandidaten in `bewertung/eigenes_modell_v3_antworten.jsonl` (Best-of-8) und
`bewertung/eigenes_modell_v3_k1_antworten.jsonl` (ein Versuch).

## Stufe 5 — eingeschränkte Dekodierung (17.09.2026)

Stufe 4 endete mit einer konkreten Zielzahl: ein Effekt unter rund zehn
Prozentpunkten ist bei n = 98 unsichtbar, und Tor-0-Fehler (unlesbare
Kurzschrift) sowie erfundene Typen kosten 20–30 % der Ausgaben. Stufe 5
greift genau die an — nicht durch Nachbessern der fertigen Antwort, sondern
durch eine **Logit-Maske**, die ein ungültiges Zeichen gar nicht erst
wählbar macht: `workflow/beschraenkt.py` verfolgt beim Erzeugen den
Grammatik-Zustand der Kurzschrift (Fassung 3, `kurzschrift.py`) zeichenweise
mit und setzt bei jedem Schritt den Logit jedes Tokens, das eine Sackgasse
wäre, auf `-inf` — **bevor** Temperatur/top-k/argmax entscheiden.

**Was die Maske erzwingt:**
- Zeile 1 exakt `wf`.
- Jede Knotenzeile `<typ>@<version>`: `typ` muss einer der 825 echten
  n8n-Node-Typen sein (`bewertung/echte_node_typen.json`, geprüft über einen
  Zeichen-Trie), `version` passend zu `\d+(\.\d+)?`.
- Erst alle Knotenzeilen, dann erst Kantenzeilen — wie es die
  Trainingsdaten immer tun; nach der ersten vollständigen Kante ist keine
  weitere Knotenzeile mehr erlaubt.
- Jede Kantenzeile `n<von> <vtyp>><idx> n<nach>`: `von`/`nach` zwischen 1
  und der bisherigen Knotenzahl (keine erfundenen Zeilenverweise mehr —
  genau der Fehler aus dem Beispiel `n19 > n20if@2.2`), `vtyp` aus den elf
  Verbindungstypen, die tatsächlich in `daten/workflow3/train.jsonl`
  vorkommen (`VERBINDUNGSTYPEN`-Konstante in `beschraenkt.py`, einmalig über
  alle 40.983 Trainingszeilen gezählt).
- EOS erst, wenn mindestens ein Knoten steht und die aktuelle Zeile leer
  (direkt nach `\n`) oder selbst schon vollständig gültig ist.

**Was sie bewusst nicht tut:** die Treffsicherheit gegenüber der
Instruktion bleibt Sache des Modells, nicht der Grammatik — die Maske macht
eine Ausgabe lesbar und typensicher, nicht zutreffend. Versionsnummern
werden nur syntaktisch geprüft, nicht gegen echte n8n-Versionsstände. Tor 4
(echter n8n-Import) bleibt eine Prüfung nach dem Erzeugen. Und: kein
Sondertoken je Typ, kein Umtrainieren — dieselben Checkpoints, derselbe
BPE-Tokenizer wie in Stufe 4.

**Einschalten:** nur hinter dem expliziten Schalter `--beschraenkt`, sonst
unverändertes Verhalten (Vorgabe bleibt Vorgabe):

```bash
python workflow/erzeuge.py --checkpoints daten/workflow3/ckpt_bestrun \
  --tokenizer daten/workflow3/tokenizer.pkl --bestes \
  --instruktionen bewertung/instruktionen.jsonl --n 3 --k 1 \
  --hoechstens-token 300 --beschraenkt
```

**Lokaler Rauchtest, dieselben 3 Instruktionen, mit und ohne Maske**
(`daten/workflow3/ckpt_bestrun`, Schritt 1.850, Temperatur 0,7, top-k 40,
Seed 0):

| | Tor 0 | Tor 1 | Tor 2 | Tor 3 | gültig@1 | erfundene Typen | Eingriffe |
|---|---|---|---|---|---|---|---|
| ohne `--beschraenkt` | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0 | — |
| mit `--beschraenkt` | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 3 |

Beide Fehlschläge ohne Maske waren derselbe Fehlertyp: „Kante nutzt
unbekannten Knoten n9" / „...n21" — eine Zeilenreferenz über die tatsächliche
Knotenzahl hinaus, also genau das, was die Bereichsprüfung 1 ≤ n ≤
Knotenzahl verhindert. „Eingriffe" zählt die Schritte, an denen das Token
mit dem höchsten Logit maskiert war — 3 auf 3 Beispiele ist kein Beleg für
einen großen Effekt, nur der Unterschied auf einer Handvoll Fälle; die
tragfähige Zahl kommt erst vom Test-Split (siehe unten).

`workflow/test_beschraenkt.py` (22 Tests) prüft die Grammatik direkt: alle
200 stichprobenartig geprüften Trainings-Kurzschriften werden Zeichen für
Zeichen akzeptiert, ein erfundener Typ wird am ersten abweichenden Byte
abgelehnt, Kantenverweise außerhalb der Knotenzahl und Knotenzeilen nach der
ersten Kante werden abgelehnt, 300 zufällig erzeugte gültige Präfixe laufen
nie in eine Sackgasse, und der stärkste Test lässt ein **untrainiertes**
MiniGPT (Zufallsgewichte, 32 Dimensionen, 1 Schicht) 30 Sequenzen bei
Temperatur 1,0 und top-k 40 erzeugen: **mit** Maske sind alle 30 lesbar,
tor-1-3-gültig und ohne erfundenen Typ — **ohne** Maske scheitern (zur
Einordnung, nicht Teil der Prüfung) auf dieser Maschine alle 30. Ein
Regressionstest belegt, dass `erzeuge_ids(...)` ohne `maske`-Argument
bitgleich zum Stand vor dieser Änderung bleibt.

**Empfohlener Bewertungsbefehl für kiserver** (Test-Split, n = 98, Best-of-4,
je zwei Erzeugungs-Seeds, mit und ohne Maske — dieselbe Gepaart-Methode wie
beim Rauschgrenze-Vergleich oben):

```bash
for beschraenkt in "" "--beschraenkt"; do
  for seed in 0 1; do
    tag=$([ -z "$beschraenkt" ] && echo ohne || echo mit)
    python workflow/erzeuge.py --checkpoints daten/workflow3/ckpt_bestrun \
      --tokenizer daten/workflow3/tokenizer.pkl --bestes \
      --test daten/workflow3/test.jsonl --k 4 --temperatur 0.7 --top-k 40 \
      --seed $seed $beschraenkt \
      --out bewertung/ergebnisse/stufe5-${tag}-seed${seed}-2026-09-17.json
  done
done
python bewertung/vergleiche_checkpoints.py \
  --a bewertung/ergebnisse/stufe5-ohne-seed0-2026-09-17.json \
  --b bewertung/ergebnisse/stufe5-mit-seed0-2026-09-17.json \
  --out bewertung/ergebnisse/stufe5-vergleich-seed0-2026-09-17.json
```

**Grenzen, ehrlich benannt:**
- Die Maske erzwingt **Lesbarkeit und echte Typen**, nicht Treffsicherheit —
  der Typen-Jaccard gegen die Referenz kann sinken, wenn das Modell durch
  die Einschränkung auf einen anderen (aber gültigen) Typ ausweicht statt
  auf den eigentlich gemeinten.
- Versionsnummern werden nur gegen `\d+(\.\d+)?` geprüft, nicht gegen echte
  n8n-Versionsstände — eine syntaktisch gültige, aber nie existierende
  Version bleibt möglich.
- Ein fester `--hoechstens-token`-Deckel kann eine Sequenz mitten in ihrer
  letzten, noch unvollständigen Zeile abschneiden — kein Grammatikfehler
  (die Maske erlaubte an dieser Stelle weiterhin nur gültige Fortsetzungen),
  sondern das Ende des Zeitbudgets.
- Wie beim Best/Ende-Vergleich in Stufe 4 gilt: der Effekt muss die
  gemessene Rauschgrenze von rund zehn Prozentpunkten (bzw. 0,06 Jaccard)
  bei n = 98 schlagen, um als belegt zu gelten — der Drei-Beispiel-Rauchtest
  oben zeigt nur, dass die Maske tut, was sie soll, nicht, wie groß der
  Effekt auf dem Test-Split ist.

### Die 15 fremd formulierten Instruktionen — alle vier Tore, echter n8n-Import

Der Rauchtest oben zeigt nur, dass die Maske tut, was sie soll. Die Messung,
die zählt, ist die härteste aus Stufe 4: dieselben 15 deutschen Instruktionen
aus Stufe 3, anders formuliert als alles im Training — genau dort hatte das
Modell **drei Typen erfunden** (`s@2Tool`, `herMapTool`, `googleAdsTrigger`).
Derselbe Checkpoint (Best, Schritt 1.850), derselbe Seed, **ein** Versuch je
Instruktion (k = 1), einmal mit und einmal ohne Maske, ausgewertet durch
`bewertung/vergleich.py` — also inklusive Tor 4, dem echten Import in n8n
2.25.6:

| | n | Tor 1 | Tor 2 | Tor 3 | Tor 4 | gültig | 95-%-KI |
|---|---|---|---|---|---|---|---|
| ohne `--beschraenkt` | 15 | 27 % | 20 % | 27 % | 20 % | **20 %** | [0 %, 40 %] |
| mit `--beschraenkt` | 15 | 100 % | 100 % | 100 % | 100 % | **100 %** | [100 %, 100 %] |

Erfundene Typen: **1 gegen 0** (`n8n-nodes-base.des` — ein abgeschnittener
Typname, den die Maske konstruktionsbedingt nicht schreiben kann). Die Maske
griff dabei 42-mal ein, also in 42 Dekodierschritten war das Token mit dem
höchsten Logit unzulässig.

**Das ist der Sprung, den Stufe 4 gefordert hat:** 80 Prozentpunkte liegen
weit über der dort gemessenen Rauschgrenze von rund 10 Prozentpunkten, und
anders als bei Best-of-k steckt hier **kein** Validator in der Schleife —
jede Antwort ist der erste und einzige Versuch. Zum Vergleich: die neun
Frontier-Modelle liegen auf derselben Aufgabe mit einem Versuch bei 60–100 %
(Stufe 3). Ein 7-Mio.-Parameter-Modell, das auf einer CPU trainiert wurde,
erreicht mit eingeschränkter Dekodierung denselben Wert wie das beste davon.

**Und die Einschränkung, die dazugehört:** Gültigkeit ist nicht
Treffsicherheit. Die Maske erzwingt, dass ein Workflow *lesbar, strukturell
korrekt und aus echten Node-Typen gebaut* ist — nicht, dass er *das* tut, was
die Instruktion verlangt. Ob die Typen-Überdeckung (Jaccard) unter der Maske
steigt, fällt oder gleich bleibt, beantwortet der nächste Abschnitt auf dem
Test-Split mit n = 98, wo eine Referenz existiert — kurz: sie fällt nicht,
und der alte Befund aus Stufe 4 bleibt trotzdem stehen, dass das Modell bei
der Treffsicherheit unter der Abschreib-Kontrolle liegt.

**Und die Gegenprobe gegen den eigenen Befund von vorhin:** Stufe 4 hat
gezeigt, dass ein bloß anderer Zufallsstrom die Zahlen um mehrere
Prozentpunkte bewegt — eine Messung mit einem einzigen Seed wäre nach dieser
Erkenntnis nicht belastbar. Also dieselben 15 Instruktionen noch einmal mit
Seed 1 und 2, beide Arme, gepaart über (Seed, Fall-ID):

| | gültig@1 | 95-%-KI | erfundene Typen |
|---|---|---|---|
| ohne `--beschraenkt`, 3 Seeds gepoolt | 10/45 = 22 % | [13 %, 36 %] | 11 |
| mit `--beschraenkt`, 3 Seeds gepoolt | **45/45 = 100 %** | [92 %, 100 %] | **0** |

Je Seed einzeln ohne Maske 20 %, 27 %, 20 % — mit Maske dreimal 100 %.
Exakter McNemar-Test über die 45 Paare: 35 diskordante Fälle, **alle** in
dieselbe Richtung, p = 6 × 10⁻¹¹. Die Maske griff je nach Seed 42-, 17- bzw.
18-mal ein. Der Effekt hängt also nicht am Zufallsstrom, und er ist um
Größenordnungen deutlicher als die Rauschgrenze, an der die Stufe-4-Messung
gescheitert ist.

Rohdaten: `bewertung/ergebnisse/stufe5-15instr-{ohne,maske}-k1*.json`,
`stufe5-15instr-tor4-2026-09-17.json`, `stufe5-15instr-3seeds-2026-09-17.json`,
Kandidaten in `bewertung/stufe5_tor4_kandidaten.jsonl`.

### Der Test-Split, n = 98 — und was die Maske *nicht* kann

Die 15 Instruktionen haben keine Referenzlösung; die Frage nach der
Treffsicherheit lässt sich dort nicht stellen. Auf dem Test-Split schon: 98
Vorlagen, die das Modell nie gesehen hat, jede mit ihrer echten Kurzschrift
als Referenz. Derselbe Checkpoint, derselbe Seed, Best-of-4, einmal mit und
einmal ohne Maske, gepaart über die Fall-ID
(`bewertung/vergleiche_checkpoints.py`):

| | ohne Maske | mit Maske | Differenz, 95-%-KI | p |
|---|---|---|---|---|
| lesbar (Tor 0) | 75,5 % | **100 %** | +24,5 pp [16, 34] | < 0,0001 |
| gültig@1 | 35,7 % | **96,9 %** | +61,2 pp [51, 70] | < 0,0001 |
| gültig@4 | 74,5 % | **100 %** | +25,5 pp [17, 35] | < 0,0001 |
| ins Token-Limit gelaufen | 7,1 % | 1,0 % | −6,1 pp [1, 12] | 0,07 |
| erfundene Typen | 1 | **0** | | |
| **Typen-Jaccard, unbedingt** | 0,260 | **0,346** | +0,087 [0,044, 0,129] | 0,0002 |
| Typen-Jaccard, nur beidseitig gültig | 0,344 | 0,343 | +0,001 [−0,036, +0,039] | 0,98 |

**Die letzten beiden Zeilen sind der eigentliche Befund, und sie widersprechen
sich nur scheinbar.** Unbedingt gerechnet (eine unlesbare Ausgabe zählt 0)
steigt die Treffsicherheit deutlich und signifikant. Auf den Fällen, in denen
*beide* Arme etwas Gültiges erzeugt haben, ist sie **identisch** — 0,344 gegen
0,343, das Intervall schließt jeden nennenswerten Unterschied aus.

Beides zusammen heißt: **die Maske macht das Modell nicht klüger, sie
verliert aber auch nichts.** Der ganze Gewinn kommt daher, dass Ausgaben, die
vorher an einer kaputten Zeile scheiterten, jetzt überhaupt erst gewertet
werden können — und diese geretteten Ausgaben sind genauso gut wie die, die
vorher schon durchkamen. Die naheliegende Sorge, eine Grammatik-Maske presse
das Modell in sichere, aber unpassende Workflows, ist damit gemessen und
widerlegt.

Was unverändert bleibt: die **Kanten** (Jaccard 0,019 → 0,026, nicht
signifikant). Die Maske stellt sicher, dass jede Kante auf einen existierenden
Knoten zeigt — nicht, dass es die *richtige* Kante ist. Und die
Nächster-Nachbar-Kontrolle aus Stufe 4 liegt bei 0,52 Typen und 0,20 Kanten:
**Abschreiben schlägt das Modell bei der Treffsicherheit weiterhin deutlich.**
Stufe 5 hat die Gültigkeitslücke geschlossen, nicht die Wissenslücke.

Nebenbei gemessen: der Lauf **mit** Maske war schneller (522 s gegen 1.329 s),
obwohl jeder einzelne Dekodierschritt teurer ist — weil Best-of-4 fast nie
einen zweiten Versuch braucht. Die Maske griff über den ganzen Lauf 712-mal
ein.

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
