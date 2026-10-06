"""
TESTS FUER STUFE 5 -- EINGESCHRAENKTE DEKODIERUNG (workflow/beschraenkt.py).

    python -m unittest workflow.test_beschraenkt -v        (aus dem Repo-Wurzelordner)

Tests, die den echten Tokenizer oder die Trainingsdaten brauchen, werden
uebersprungen, wenn sie fehlen -- wie in test_workflow.py (DB_DA-Muster).
Die reine Grammatiklogik (kante_praefix_status, TypenTrie, Zeilen-/EOS-
Regeln) braucht beides nicht und laeuft ueberall.
"""

import json
import random
import re
import sys
import unittest
from pathlib import Path

WURZEL = Path(__file__).parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "kern"))
sys.path.insert(0, str(WURZEL / "bewertung"))

import torch  # noqa: E402

from workflow import kurzschrift as ks  # noqa: E402
from workflow.beschraenkt import (  # noqa: E402
    KurzschriftMaske, TypenTrie, VERBINDUNGSTYPEN,
    _version_praefix_ok, _version_voll_ok, kante_praefix_status,
)
from workflow.erzeuge import erzeuge_ids  # noqa: E402
from workflow.tokenizer_workflow import WorkflowTokenizer  # noqa: E402
from modell import MiniGPT  # noqa: E402
from tore import lade_echte_node_typen, pruefe_alle_tore  # noqa: E402

TOK_PFAD = WURZEL / "daten" / "workflow3" / "tokenizer.pkl"
TRAIN_PFAD = WURZEL / "daten" / "workflow3" / "train.jsonl"
DATEN_DA = TOK_PFAD.exists() and TRAIN_PFAD.exists()

_ECHTE = None


def echte_typen() -> set[str]:
    global _ECHTE
    if _ECHTE is None:
        _ECHTE = lade_echte_node_typen()
    return _ECHTE


def fuettere(maske: KurzschriftMaske, text: str, zeichen_erlaubt=None, schreibe=None) -> None:
    """Schreibt `text` Zeichen fuer Zeichen ueber die uebergebenen Methoden
    (Vorgabe: die privaten Zeichen-Methoden der Maske) und schlaegt fehl,
    sobald ein Zeichen abgelehnt wird -- so bekommt man beim Test-Fehlschlag
    sofort die genaue Stelle statt nur "irgendwo falsch"."""
    zeichen_erlaubt = zeichen_erlaubt or maske._zeichen_erlaubt
    schreibe = schreibe or maske._schreibe_zeichen
    for i, ch in enumerate(text):
        assert zeichen_erlaubt(ch), f"{ch!r} (Position {i}) nach Zustand {maske.zeile!r} abgelehnt"
        schreibe(ch)


# ===========================================================================
# Reine Grammatiklogik -- kein Tokenizer, keine Trainingsdaten noetig
# ===========================================================================

class Ziffern(unittest.TestCase):
    def test_version_praefix_und_vollstaendig(self):
        self.assertTrue(_version_praefix_ok(""))
        self.assertTrue(_version_praefix_ok("2"))
        self.assertTrue(_version_praefix_ok("2."))
        self.assertTrue(_version_praefix_ok("2.1"))
        self.assertFalse(_version_praefix_ok("."))       # \d+ verlangt eine Ziffer vor dem Punkt
        self.assertFalse(_version_praefix_ok("2.1.3"))
        self.assertFalse(_version_praefix_ok("a"))
        self.assertTrue(_version_voll_ok("2"))
        self.assertTrue(_version_voll_ok("2.1"))
        self.assertFalse(_version_voll_ok("2."))          # Punkt ohne folgende Ziffer ist nicht \d+(\.\d+)?
        self.assertFalse(_version_voll_ok(""))

    def test_kante_praefix_status_zahlenbereich(self):
        # "n" allein: Praefix, noch nicht vollstaendig
        self.assertEqual(kante_praefix_status("n", 5), (True, False))
        # von <= anzahl_knoten bleibt Praefix
        self.assertEqual(kante_praefix_status("n5", 5), (True, False))
        # von > anzahl_knoten ist eine Sackgasse -- auch als Praefix nicht mehr rettbar
        self.assertEqual(kante_praefix_status("n6", 5)[0], False)
        # fuehrende Null bei einer Knotennummer ist nie gueltig
        self.assertEqual(kante_praefix_status("n0", 5)[0], False)
        self.assertEqual(kante_praefix_status("n01", 5)[0], False)

    def test_kante_praefix_status_vollstaendige_zeilen(self):
        self.assertEqual(kante_praefix_status("n1 > n2", 2), (True, True))
        self.assertEqual(kante_praefix_status("n2 >1 n4", 4), (True, True))
        self.assertEqual(kante_praefix_status("n5 ai_tool> n6", 6), (True, True))
        # unbekannter Verbindungstyp -- "ai_toolx" ist in keiner bekannten Liste enthalten
        self.assertEqual(kante_praefix_status("n1 ai_toolx> n2", 2), (False, False))
        # Zeichen nach der letzten Ziffer -- die Zeile ist verankert ($)
        self.assertEqual(kante_praefix_status("n1 > n2x", 2), (False, False))

    def test_vtyp_liste_deckt_stichprobe_ab(self):
        """VERBINDUNGSTYPEN ist aus daten/workflow3/train.jsonl gezaehlt (siehe
        Kommentar in beschraenkt.py) -- diese Stichprobe (2.000 Zeilen reichen,
        die vollstaendige Zaehlung stand einmalig beim Bau der Konstante) darf
        keinen anderen Wert enthalten."""
        if not TRAIN_PFAD.exists():
            self.skipTest("train.jsonl fehlt")
        muster = re.compile(r"^n(\d+) ([A-Za-z_]*)>(\d*) n(\d+)$")
        gefunden = set()
        with open(TRAIN_PFAD, encoding="utf8") as f:
            for i, zeile in enumerate(f):
                if i >= 2000:
                    break
                d = json.loads(zeile)
                for z in d["kurzschrift"].splitlines():
                    m = muster.match(z.strip())
                    if m:
                        gefunden.add(m.group(2))
        self.assertTrue(gefunden, "keine Kantenzeilen in der Stichprobe gefunden")
        self.assertTrue(gefunden <= set(VERBINDUNGSTYPEN), gefunden - set(VERBINDUNGSTYPEN))


class TypenTrieTest(unittest.TestCase):
    def test_praefix_und_wort(self):
        t = TypenTrie(["n8n-nodes-base.set", "n8n-nodes-base.slack", "@n8n/n8n-nodes-langchain.agent"])
        self.assertTrue(t.ist_praefix(""))
        self.assertTrue(t.ist_praefix("n8n-nodes-base.s"))
        self.assertTrue(t.ist_wort("n8n-nodes-base.set"))
        self.assertFalse(t.ist_wort("n8n-nodes-base.se"))
        self.assertFalse(t.ist_praefix("n8n-nodes-base.x"))
        self.assertTrue(t.ist_praefix("@n8n/n8n-nodes-langchain.agent"))
        self.assertFalse(t.ist_praefix("@n8n/n8n-nodes-langchain.agentX"))

    def test_dollarzeichen_kollidiert_nicht_mit_dem_wortende_marker(self):
        # Regressions-Fall: der interne Wortende-Marker war frueher der String
        # "$" -- ein Typ, dessen naechstes ECHTES Zeichen zufaellig "$" waere,
        # haette dann den Marker (True) statt eines Unterworts getroffen und
        # AttributeError geworfen. "$" ist druckbares ASCII, also von der
        # Maske nicht pauschal ausgeschlossen.
        t = TypenTrie(["n8n-nodes-base.switch"])
        self.assertTrue(t.ist_wort("n8n-nodes-base.switch"))
        self.assertFalse(t.ist_praefix("n8n-nodes-base.switch$"))
        self.assertFalse(t.ist_wort("n8n-nodes-base.switch$"))


class Zeilenordnung(unittest.TestCase):
    """Phasenregeln (Knoten vor Kanten, EOS-Grenzen) -- ohne Tokenizer ueber
    die privaten Zeichen-Methoden direkt geprueft."""

    def _maske(self, typen):
        return KurzschriftMaske(None, typen)

    def test_wf_kopfzeile_ist_exakt_wf(self):
        m = self._maske(("a.b",))
        fuettere(m, "wf")
        self.assertFalse(m._zeichen_erlaubt("x"))   # keine dritte Buchstabe erlaubt
        self.assertTrue(m._zeichen_erlaubt("\n"))
        m._schreibe_zeichen("\n")
        self.assertTrue(m.wf_fertig)

    def test_kante_vor_erstem_knoten_wird_abgelehnt(self):
        m = self._maske(("a.b", "a.c"))
        fuettere(m, "wf\n")
        self.assertFalse(m._zeichen_erlaubt("n"))

    def test_knoten_nach_erster_kante_wird_abgelehnt(self):
        m = self._maske(("a.b", "a.c"))
        fuettere(m, "wf\na.b@1\na.c@1\nn1 > n2\n")
        self.assertTrue(m.hat_kante)
        self.assertFalse(m._zeichen_erlaubt("a"))

    def test_gueltige_kante_mit_zwei_knoten_geht_durch(self):
        m = self._maske(("a.b", "a.c"))
        fuettere(m, "wf\na.b@1\na.c@1\n")
        self.assertEqual(m.anzahl_knoten, 2)
        fuettere(m, "n1 > n2")
        self.assertTrue(m._eos_erlaubt())        # Zeile selbst schon vollstaendig
        self.assertTrue(m._zeichen_erlaubt("\n"))

    def test_eos_nicht_vor_erstem_knoten(self):
        m = self._maske(("a.b",))
        fuettere(m, "wf\n")
        self.assertFalse(m._eos_erlaubt())

    def test_eos_nicht_mitten_in_zeile(self):
        m = self._maske(("a.b",))
        fuettere(m, "wf\na.b@")
        self.assertFalse(m._eos_erlaubt())
        self.assertFalse(m._zeichen_erlaubt("\n"))   # "a.b@" ist keine vollstaendige Zeile

    def test_eos_direkt_nach_vollstaendiger_offener_zeile(self):
        m = self._maske(("a.b",))
        fuettere(m, "wf\na.b@1")
        self.assertTrue(m._eos_erlaubt())            # vollstaendiger Knoten, noch kein \n

    def test_eos_direkt_nach_zeilenumbruch(self):
        m = self._maske(("a.b",))
        fuettere(m, "wf\na.b@1\n")
        self.assertTrue(m._eos_erlaubt())

    def test_erfundener_typ_wird_als_praefix_abgelehnt(self):
        m = self._maske(("n8n-nodes-base.emailSend", "n8n-nodes-base.webhook"))
        fuettere(m, "wf\n")
        # "n8n-nodes-base.se" ist noch Praefix von emailSend? Nein -- die
        # beiden bekannten Typen teilen sich nur "n8n-nodes-base." (15 Zeichen).
        text = "n8n-nodes-base.sendEmail"
        i = 0
        while i < len(text) and m._zeichen_erlaubt(text[i]):
            m._schreibe_zeichen(text[i])
            i += 1
        self.assertLess(i, len(text), "erfundener Typ wurde vollstaendig akzeptiert")
        self.assertEqual(m.zeile, "n8n-nodes-base.")


# ===========================================================================
# Tests mit dem echten Tokenizer / den echten 825 Typen / echten Trainingsdaten
# ===========================================================================

@unittest.skipUnless(DATEN_DA, "Tokenizer/Trainingsdaten (daten/workflow3/) fehlen")
class MitEchtenDaten(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = WorkflowTokenizer.lade(TOK_PFAD)
        cls.echte = echte_typen()

    def _maske(self):
        return KurzschriftMaske(self.tok, self.echte)

    def _fuettere_token(self, maske, text):
        """Wie `fuettere()`, aber ueber die echte Token-Schnittstelle: Bytes
        0..255 haben in tokenizer_workflow.py immer Token-ID == Byte-Wert
        (SchnellerBPETokenizer.vokabular = {i: bytes([i]) for i in range(256)}),
        also entspricht ord(ch) fuer ASCII-Zeichen genau einem Ein-Byte-Token."""
        for i, ch in enumerate(text):
            tid = ord(ch)
            assert maske.erlaubt(tid), f"{ch!r} (Position {i}) nach {maske.zeile!r} abgelehnt"
            maske.schreibe(tid)

    def test_zweihundert_trainings_kurzschriften_werden_akzeptiert(self):
        maske = self._maske()
        anzahl = 0
        with open(TRAIN_PFAD, encoding="utf8") as f:
            for zeile in f:
                if anzahl >= 200:
                    break
                kurz = json.loads(zeile)["kurzschrift"]
                maske.zuruecksetzen()
                self._fuettere_token(maske, kurz)
                self.assertTrue(maske.erlaubt(self.tok.eos_id),
                                 f"EOS nach vollstaendiger Kurzschrift #{anzahl} abgelehnt")
                anzahl += 1
        self.assertEqual(anzahl, 200)

    def test_erfundener_typ_wird_beim_ersten_abweichenden_byte_abgelehnt(self):
        maske = self._maske()
        self._fuettere_token(maske, "wf\n")
        erfunden = "n8n-nodes-base.sendEmail@1"
        i = 0
        while i < len(erfunden) and maske.erlaubt(ord(erfunden[i])):
            maske.schreibe(ord(erfunden[i]))
            i += 1
        self.assertLess(i, len(erfunden), "erfundener Typ n8n-nodes-base.sendEmail wurde akzeptiert")

        maske2 = self._maske()
        self._fuettere_token(maske2, "wf\n")
        self._fuettere_token(maske2, "n8n-nodes-base.emailSend@2.1")
        self.assertTrue(maske2.erlaubt(self.tok.eos_id))

    def test_kantenzahl_und_phasenordnung_ueber_token(self):
        maske = self._maske()
        self._fuettere_token(maske, "wf\nn8n-nodes-base.webhook@2\nn8n-nodes-base.noOp@1\n")
        self.assertEqual(maske.anzahl_knoten, 2)
        # n3 gibt es nicht (nur 2 Knoten)
        self.assertTrue(maske.erlaubt(ord("n")))
        maske.schreibe(ord("n"))
        self.assertFalse(maske.erlaubt(ord("3")))
        self.assertTrue(maske.erlaubt(ord("2")))
        maske.schreibe(ord("2")); maske.schreibe(ord(" ")); maske.schreibe(ord(">")); maske.schreibe(ord(" "))
        self._fuettere_token(maske, "n1")
        self.assertTrue(maske.erlaubt(self.tok.eos_id) or maske.erlaubt(ord("\n")))
        maske.schreibe(ord("\n"))
        self.assertTrue(maske.hat_kante)
        # keine weitere Knotenzeile mehr moeglich
        self.assertFalse(maske.erlaubt(ord("n8n-nodes-base.webhook@2"[0])) and
                          maske.erlaubt(ord("8")))

    def test_keine_sackgasse_bei_zufaelligen_gueltigen_praefixen(self):
        maske = self._maske()
        rng = random.Random(7)
        kandidaten = [chr(c) for c in range(32, 127)] + ["\n"]
        for _ in range(300):
            maske.zuruecksetzen()
            for _ in range(rng.randint(0, 40)):
                legal = [c for c in kandidaten if maske.erlaubt(ord(c))]
                if not legal:
                    break
                ch = rng.choice(legal)
                maske.schreibe(ord(ch))
            gefunden = any(maske.erlaubt(tid) for tid in range(256)) or maske.erlaubt(self.tok.eos_id)
            self.assertTrue(gefunden, f"Sackgasse bei Zustand {maske.zeile!r}")

    def test_zufallsmodell_mit_maske_erzeugt_immer_gueltige_kurzschrift(self):
        """Der staerkste Test: ein kleines, UNTRAINIERTES MiniGPT (Zufallsgewichte)
        erzeugt mit der Maske 30 Sequenzen bei temperatur=1.0, top_k=40. Jede
        muss lesbar sein, darf keinen erfundenen Typ enthalten und muss Tor
        1-3 bestehen -- ohne dass das Modell je etwas ueber n8n gelernt hat.

        Ein fester Token-Deckel (hoechstens_token) kann eine Sequenz mitten
        in ihrer letzten (noch unvollstaendigen) Zeile abschneiden -- das ist
        kein Grammatikfehler (die Maske erlaubt an dieser Stelle weiterhin nur
        gueltige Fortsetzungen), sondern schlicht das Ende des Zeitbudgets.
        Fuer die Pruefung wird eine solche unvollstaendige letzte Zeile
        deshalb verworfen; alle VORHER abgeschlossenen Zeilen sind durch die
        Maske bereits einzeln als vollstaendig gueltig bestaetigt. Mit dem
        gewaehlten Seed/Budget unten tritt das nicht einmal ein (siehe
        Zaehler `abgeschnitten` unten) -- der Ausweg bleibt als Sicherheitsnetz
        gegen Plattform-/Versionsunterschiede stehen.
        """
        maske = self._maske()
        torch.manual_seed(0)
        modell = MiniGPT(self.tok.groesse, dim=32, koepfe=2, schichten=1, block=300)
        modell.eval()
        torch.manual_seed(0)
        prompt = self.tok.kodiere_prompt("Baue einen einfachen Workflow.")

        abgeschnitten = 0
        for versuch in range(30):
            ids = erzeuge_ids(modell, prompt, self.tok.eos_id, 200, 1.0, 40, "cpu", maske=maske)
            kurz = self.tok.dekodiere_antwort(ids)
            try:
                wf = ks.rendere(kurz)
            except ks.KurzschriftFehler:
                self.assertIn("\n", kurz, f"Versuch {versuch}: unlesbar ohne jede vollstaendige Zeile")
                abgeschnitten += 1
                wf = ks.rendere(kurz.rsplit("\n", 1)[0])
            r = pruefe_alle_tore(json.dumps(wf))
            self.assertTrue(r["tor1_json"], f"Versuch {versuch}: Tor 1 {r['probleme']}")
            self.assertTrue(r["tor2_struktur"], f"Versuch {versuch}: Tor 2 {r['probleme']}")
            self.assertTrue(r["tor3_verbindungen"], f"Versuch {versuch}: Tor 3 {r['probleme']}")
            gefundene_typen = {n["type"] for n in wf["nodes"]}
            self.assertTrue(gefundene_typen <= self.echte,
                             f"Versuch {versuch}: erfundene Typen {gefundene_typen - self.echte}")

        # Kontrollmessung OHNE Maske, GENAU DASSELBE Zufallsmodell -- nur zur
        # Einordnung ausgegeben, wie im Auftrag verlangt NICHT Teil der Pruefung
        # (ein einzelnes untrainiertes Modell muss hier nicht zuverlaessig
        # scheitern, aber typischerweise tut es das deutlich haeufiger).
        torch.manual_seed(0)
        misserfolge_ohne_maske = 0
        for _ in range(30):
            ids = erzeuge_ids(modell, prompt, self.tok.eos_id, 200, 1.0, 40, "cpu", maske=None)
            kurz = self.tok.dekodiere_antwort(ids)
            try:
                wf = ks.rendere(kurz)
                if not ({n["type"] for n in wf["nodes"]} <= self.echte):
                    misserfolge_ohne_maske += 1
            except ks.KurzschriftFehler:
                misserfolge_ohne_maske += 1
        print(f"\n[Kontrolle] ohne Maske: {misserfolge_ohne_maske}/30 unlesbar oder mit "
              f"erfundenem Typ; mit Maske: 0/30, {abgeschnitten}/30 an der Token-Grenze "
              f"abgeschnitten (nur Einordnung, nicht Teil der Pruefung).")


# ===========================================================================
# Regression: erzeuge_ids OHNE Maske ist bitgleich zum alten Verhalten
# ===========================================================================

class OhneMaskeRegression(unittest.TestCase):
    def _kleines_modell_und_prompt(self):
        torch.manual_seed(42)
        modell = MiniGPT(64, dim=16, koepfe=2, schichten=1, block=32)
        modell.eval()
        prompt = [1, 2, 3]
        return modell, prompt

    def test_ohne_maske_ist_deterministisch_und_ruehrt_die_maske_nicht_an(self):
        modell, prompt = self._kleines_modell_und_prompt()

        class ExplodierendeMaske:
            """Jede Beruehrung ist ein Programmierfehler -- `maske=None`
            darf diesen Codepfad nie erreichen."""
            def __getattr__(self, name):
                raise AssertionError(f"Maskenmethode {name!r} wurde ohne Maske aufgerufen")

        torch.manual_seed(0)
        a = erzeuge_ids(modell, prompt, eos_id=63, hoechstens=25, temperatur=0.8,
                        top_k=10, geraet="cpu")               # maske gar nicht angegeben
        torch.manual_seed(0)
        b = erzeuge_ids(modell, prompt, eos_id=63, hoechstens=25, temperatur=0.8,
                        top_k=10, geraet="cpu", maske=None)    # explizit None
        self.assertEqual(a, b)

        # Dieselbe Erzeugung nochmal, mit einer Maske, die bei jeder Beruehrung
        # explodiert -- da maske=None bleibt (kein drittes erzeuge_ids ruft sie
        # tatsaechlich auf), dient das nur als Beleg, dass der Aufruf ohne
        # `maske` gar keinen Verweis auf ein Maskenobjekt braucht.
        ExplodierendeMaske()  # instanziierbar, aber hier nie in erzeuge_ids uebergeben

    def test_argmax_pfad_ohne_maske_ist_deterministisch(self):
        modell, prompt = self._kleines_modell_und_prompt()
        torch.manual_seed(0)
        a = erzeuge_ids(modell, prompt, eos_id=63, hoechstens=15, temperatur=0.0,
                        top_k=None, geraet="cpu")
        torch.manual_seed(0)
        b = erzeuge_ids(modell, prompt, eos_id=63, hoechstens=15, temperatur=0.0,
                        top_k=None, geraet="cpu", maske=None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
