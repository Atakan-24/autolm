"""
STUFE 5 -- EINGESCHRAENKTE DEKODIERUNG (constrained decoding), nur hinter `--beschraenkt`.

Stufe 4 hat zwei Fehlerquellen gemessen, die NACH dem Training noch offen
sind: unlesbare Kurzschrift (Tor 0, 19-27 % je nach Stand) und erfundene
Node-Typen (auf fremd formulierten Instruktionen dreimal in 15 Antworten,
siehe README "Und der Befund, der die eigene These einschraenkt"). Beides
ist beim Erzeugen bereits sichtbar: das Modell hat gerade ein Zeichen
geschrieben, das keine gueltige Kurzschrift-Zeile mehr fortsetzen kann
(z. B. `n19 > n20if@2.2` -- ein Bewertungsbeispiel, in dem nach der
Kantenzeile `n19 > n20` unvermittelt ein Typname weitergeschrieben wurde,
statt einer neuen Kantenzeile oder EOS).

DIESES MODUL REPARIERT NICHTS NACHTRAEGLICH. Es macht das ungueltige
Zeichen gar nicht erst waehlbar: `maskiere_logits()` setzt den Logit jedes
Tokens, das den bisherigen Text in eine Sackgasse fuehren wuerde, auf
-inf, BEVOR Temperatur/top-k/argmax entscheiden. Das Modell entscheidet
weiterhin selbst -- nur aus einer kleineren, garantiert fortsetzbaren
Menge.

WIE DIE PRUEFUNG FUNKTIONIERT -- eine zeichenweise Praefixpruefung ueber
genau die Grammatik aus kurzschrift.py (Fassung 3, siehe dort fuer die
Regexe, an denen sich hier jede Grenze orientiert):

    Zeile 1                 exakt "wf"
    Knotenzeilen (0..n)     "<typ>@<version>"   typ aus den 825 echten
                            Node-Typen (bewertung/echte_node_typen.json),
                            version passend zu \\d+(\\.\\d+)?
    Kantenzeilen (0..n)     "n<von> [vtyp]>[idx] n<nach>", von/nach
                            zwischen 1 und der bisherigen Knotenzahl,
                            vtyp aus VERBINDUNGSTYPEN (siehe unten)

Die Trainingsdaten schreiben IMMER erst alle Knoten, dann alle Kanten
(kurzschrift.serialisiere()) -- das erzwingt dieses Modul mit: sobald
eine vollstaendige Kantenzeile abgeschlossen ist, sind Knotenzeilen fuer
den Rest der Antwort nicht mehr erlaubt. Vor dem ersten Knoten ist auch
"n1 > n2" nicht erlaubt (Kante ohne Referenzziel), und EOS ist erst
erlaubt, wenn mindestens ein Knoten steht und die aktuelle Zeile entweder
leer (direkt nach einem \\n) oder selbst schon vollstaendig gueltig ist.

WAS DIE MASKE BEWUSST NICHT PRUEFT:
  - ob der erzeugte Graph zur Instruktion passt (Treffsicherheit bleibt
    Sache des Modells, nicht der Grammatik)
  - ob eine Versionsnummer bei n8n wirklich existiert (nur die Syntax
    \\d+(\\.\\d+)? wird erzwungen, nicht der Wertebereich)
  - Tor 4 (echter n8n-Import) -- das bleibt eine Pruefung nach dem
    Erzeugen, keine, die sich in eine Logit-Maske uebersetzen liesse

KEINE SACKGASSE MOEGLICH: die Byte-Tokens 0..255 sind immer im
BPE-Vokabular (tokenizer_workflow.py). Jedes Zeichen, das die Grammatik an
irgendeiner Stelle braucht (Ziffern, Punkt, "n", ">", "@", "\\n" ...), ist
also mindestens als Ein-Byte-Token erreichbar -- die Praefixpruefung muss
nur an jeder Stelle mindestens EIN gueltiges Zeichen offenlassen, nie ein
mehrbytiges Token.
"""

import re

VERBINDUNGSTYPEN = [
    # Alle Verbindungstypen, die in daten/workflow3/train.jsonl vorkommen
    # (vollstaendig gezaehlt ueber alle 40.983 Zeilen, 17.09.2026 -- siehe
    # test_beschraenkt.py::test_vtyp_liste_deckt_stichprobe_ab). "" ist
    # `main`, das kurzschrift.serialisiere() als leeren String schreibt
    # (t = "" if vtyp == "main" else vtyp).
    "",
    "ai_document",
    "ai_embedding",
    "ai_languageModel",
    "ai_memory",
    "ai_outputParser",
    "ai_reranker",
    "ai_retriever",
    "ai_textSplitter",
    "ai_tool",
    "ai_vectorStore",
]

_VERSION_VOLL = re.compile(r"^\d+(\.\d+)?$")
_VERSION_PRAEFIX = re.compile(r"^\d+\.?\d*$")


def _version_praefix_ok(v: str) -> bool:
    return v == "" or bool(_VERSION_PRAEFIX.fullmatch(v))


def _version_voll_ok(v: str) -> bool:
    return bool(_VERSION_VOLL.fullmatch(v))


def _vtyp_praefix_ok(v: str) -> bool:
    return any(t.startswith(v) for t in VERBINDUNGSTYPEN)


def _vtyp_voll_ok(v: str) -> bool:
    return v in VERBINDUNGSTYPEN


class TypenTrie:
    """Zeichenweiser Trie ueber die 825 echten Node-Typen -- Praefix- und
    Wort-Pruefung je in O(Laenge des Typnamens), nicht O(825)."""

    # `None` als Wortende-Markerschluessel, NICHT ein String wie "$": ein
    # BPE-Token kann buchstaeblich jedes druckbare ASCII-Zeichen enthalten
    # (auch "$", "#", "~", ...), und ein Markerschluessel, der selbst ein
    # gueltiges Zeichen ist, wuerde mit einem Typnamen kollidieren, der
    # zufaellig genau bis dorthin passt -- `knoten.get(ch)` traefe dann auf
    # den Marker (`True`) statt auf ein Unterwoerterbuch und stuerzte ab.
    # `None` kann als Zeichen aus `s` nie vorkommen (s ist immer ein `str`).
    _ENDE = None

    def __init__(self, typen):
        self._wurzel: dict = {}
        for t in typen:
            knoten = self._wurzel
            for ch in t:
                knoten = knoten.setdefault(ch, {})
            knoten[self._ENDE] = True

    def _knoten(self, s: str):
        knoten = self._wurzel
        for ch in s:
            knoten = knoten.get(ch)
            if knoten is None:
                return None
        return knoten

    def ist_praefix(self, s: str) -> bool:
        """True, wenn s Praefix (oder selbst) mindestens eines Typs ist."""
        return self._knoten(s) is not None

    def ist_wort(self, s: str) -> bool:
        """True, wenn s EXAKT einem vollstaendigen Typ entspricht."""
        knoten = self._knoten(s)
        return knoten is not None and self._ENDE in knoten


def _ziffern_lesen(s: str, i: int, ober_grenze: int | None,
                    keine_fuehrende_null: bool = False) -> tuple[int, int, bool]:
    """
    Liest eine Ziffernfolge ab Index i (0..n Ziffern, wie \\d*).

    Gibt (neuer_index, anzahl_gelesener_ziffern, gueltig) zurueck.
    `gueltig=False` heisst Sackgasse: entweder eine fuehrende Null bei
    einer Knotennummer (die Trainingsdaten schreiben Nummern nie mit
    fuehrender Null -- str(i) in kurzschrift.serialisiere()) oder ein Wert,
    der `ober_grenze` bereits ueberschritten hat und durch weitere Ziffern
    nur noch groesser werden kann.
    """
    wert = 0
    anzahl = 0
    n = len(s)
    while i < n and s[i].isdigit():
        if anzahl == 0 and keine_fuehrende_null and s[i] == "0":
            return i, anzahl, False
        wert = wert * 10 + int(s[i])
        anzahl += 1
        if ober_grenze is not None and wert > ober_grenze:
            return i, anzahl, False
        i += 1
    return i, anzahl, True


def kante_praefix_status(s: str, anzahl_knoten: int) -> tuple[bool, bool]:
    """
    (ist_gueltiger_praefix, ist_vollstaendige_kante) fuer den Zeileninhalt
    (ohne \\n) einer Kante "n<von> <vtyp>><idx> n<nach>" -- die Grammatik
    aus kurzschrift._KANTE, zeichenweise nachgebaut, mit den zusaetzlichen
    Bereichspruefungen 1 <= von, nach <= anzahl_knoten und vtyp aus
    VERBINDUNGSTYPEN.
    """
    n = len(s)
    if n == 0:
        return True, False
    if s[0] != "n":
        return False, False
    i = 1
    if i == n:
        return True, False
    i, anz, ok = _ziffern_lesen(s, i, anzahl_knoten, keine_fuehrende_null=True)
    if not ok:
        return False, False
    if anz == 0:
        return False, False  # naechstes Zeichen existiert, ist aber keine Ziffer -- "von" braucht >=1
    if i == n:
        return True, False
    if s[i] != " ":
        return False, False
    i += 1
    if i == n:
        return True, False

    vtyp_start = i
    while i < n and s[i] != ">":
        i += 1
    vtyp_so_weit = s[vtyp_start:i]
    if not _vtyp_praefix_ok(vtyp_so_weit):
        return False, False
    if i == n:
        return True, False
    if not _vtyp_voll_ok(vtyp_so_weit):
        return False, False  # ">" getippt, aber vtyp bisher kein bekannter Verbindungstyp
    i += 1  # ">" konsumiert
    if i == n:
        return True, False

    i, _, ok = _ziffern_lesen(s, i, None)  # Ausgang-Index: \d*, keine Obergrenze
    if not ok:
        return False, False
    if i == n:
        return True, False
    if s[i] != " ":
        return False, False
    i += 1
    if i == n:
        return True, False
    if s[i] != "n":
        return False, False
    i += 1
    if i == n:
        return True, False

    i, anz2, ok = _ziffern_lesen(s, i, anzahl_knoten, keine_fuehrende_null=True)
    if not ok:
        return False, False
    if anz2 == 0:
        return False, False
    if i == n:
        return True, True
    return False, False  # Zeichen nach "nach" -- die Zeile ist regex-verankert ($)


class KurzschriftMaske:
    """
    Grammatik-Zustand + Logit-Maske fuer die eingeschraenkte Dekodierung
    einer Kurzschrift-Antwort (Fassung 3, kurzschrift.py). Ein Objekt lebt
    ueber eine ganze Erzeugung; `zuruecksetzen()` startet neu (z. B. fuer
    jeden Best-of-k-Versuch in erzeuge.py).
    """

    def __init__(self, tok, echte_typen):
        self.tok = tok
        self.trie = TypenTrie(echte_typen)
        self.zuruecksetzen()

    def zuruecksetzen(self) -> None:
        self.wf_fertig = False
        self.anzahl_knoten = 0
        self.hat_kante = False
        self.zeile = ""
        self.eingriffe = 0

    # ---- Zeilenpruefung ------------------------------------------------

    def _knoten_praefix(self, s: str) -> bool:
        if self.trie.ist_praefix(s):
            return True
        for i, ch in enumerate(s):
            if ch == "@":
                t, v = s[:i], s[i + 1:]
                if self.trie.ist_wort(t) and _version_praefix_ok(v):
                    return True
        return False

    def _knoten_vollstaendig(self, s: str) -> bool:
        for i, ch in enumerate(s):
            if ch == "@":
                t, v = s[:i], s[i + 1:]
                if self.trie.ist_wort(t) and _version_voll_ok(v):
                    return True
        return False

    def _zeile_praefix(self, s: str) -> bool:
        ok = False
        if not self.hat_kante:
            ok = ok or self._knoten_praefix(s)
        if self.anzahl_knoten >= 1:
            p, _ = kante_praefix_status(s, self.anzahl_knoten)
            ok = ok or p
        return ok

    def _zeile_vollstaendig(self, s: str) -> bool:
        if not self.hat_kante and self._knoten_vollstaendig(s):
            return True
        if self.anzahl_knoten >= 1:
            _, voll = kante_praefix_status(s, self.anzahl_knoten)
            if voll:
                return True
        return False

    # ---- Zeichenebene ----------------------------------------------------

    def _zeichen_erlaubt(self, ch: str) -> bool:
        if not self.wf_fertig:
            if ch == "\n":
                return self.zeile == "wf"
            return (self.zeile + ch) in ("w", "wf")
        if ch == "\n":
            return self._zeile_vollstaendig(self.zeile)
        if ord(ch) > 127:
            return False  # Grammatik und Typen sind reines ASCII
        return self._zeile_praefix(self.zeile + ch)

    def _schreibe_zeichen(self, ch: str) -> None:
        if not self.wf_fertig:
            if ch == "\n":
                self.wf_fertig = True
                self.zeile = ""
            else:
                self.zeile += ch
            return
        if ch == "\n":
            if not self.hat_kante and self._knoten_vollstaendig(self.zeile):
                self.anzahl_knoten += 1
            else:
                self.hat_kante = True
            self.zeile = ""
        else:
            self.zeile += ch

    def _eos_erlaubt(self) -> bool:
        if not self.wf_fertig:
            return False
        if self.zeile == "":
            return self.anzahl_knoten >= 1
        if not self._zeile_vollstaendig(self.zeile):
            return False
        if not self.hat_kante and self._knoten_vollstaendig(self.zeile):
            return True
        return self.anzahl_knoten >= 1

    # ---- Tokenebene ------------------------------------------------------

    def erlaubt(self, token_id: int) -> bool:
        """Darf `token_id` als naechstes Token geschrieben werden, ohne den
        bisherigen Text in eine Sackgasse zu fuehren?"""
        if token_id == self.tok.eos_id:
            return self._eos_erlaubt()
        if token_id in (self.tok.instr_id, self.tok.wf_id):
            return False
        b = self.tok.bpe.vokabular.get(token_id)
        if b is None:
            return False
        try:
            text = b.decode("ascii")
        except UnicodeDecodeError:
            return False
        # Char fuer Char simulieren, danach den echten Zustand wiederherstellen --
        # billiger als eine tiefe Kopie fuer die paar Felder, die hier zaehlen.
        sicherung = (self.wf_fertig, self.anzahl_knoten, self.hat_kante, self.zeile)
        ok = True
        for ch in text:
            if not self._zeichen_erlaubt(ch):
                ok = False
                break
            self._schreibe_zeichen(ch)
        self.wf_fertig, self.anzahl_knoten, self.hat_kante, self.zeile = sicherung
        return ok

    def schreibe(self, token_id: int) -> None:
        """Schaltet den Zustand nach dem tatsaechlich gewaehlten Token fort."""
        if token_id in (self.tok.instr_id, self.tok.wf_id, self.tok.eos_id):
            return
        b = self.tok.bpe.vokabular.get(token_id)
        if b is None:
            return
        for ch in b.decode("ascii"):
            self._schreibe_zeichen(ch)

    def maskiere_logits(self, logits, top_k, temperatur):
        """
        logits: Tensor der Form (1, V) -- der letzte Zeitschritt, VOR
        Temperatur/top-k/argmax. Setzt jedes unzulaessige Token auf -inf.

        Effizienz: Logits absteigend sortieren, von oben pruefen, bis
        genug zulaessige gefunden sind (1 bei temperatur<=0, sonst top_k --
        ohne top_k das ganze Vokabular, weil dann die volle Verteilung
        zaehlt) -- statt alle ~4096 Tokens auf jedem Schritt zu pruefen.
        """
        row = logits[0]
        argmax_id = int(row.argmax().item())
        if not self.erlaubt(argmax_id):
            self.eingriffe += 1

        vokabular_groesse = row.shape[0]
        if temperatur <= 0:
            benoetigt = 1
        elif top_k:
            benoetigt = min(top_k, vokabular_groesse)
        else:
            benoetigt = vokabular_groesse

        ordnung = row.argsort(descending=True).tolist()
        erlaubte_ids = []
        for tid in ordnung:
            if self.erlaubt(tid):
                erlaubte_ids.append(tid)
                if len(erlaubte_ids) >= benoetigt:
                    break

        neue_row = row.new_full(row.shape, -float("inf"))
        for tid in erlaubte_ids:
            neue_row[tid] = row[tid]
        return neue_row.unsqueeze(0)
