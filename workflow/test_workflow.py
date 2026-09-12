"""
TESTS FUER STUFE 4 -- jede Regel, die beim Aendern nicht kippen darf.

    python -m unittest workflow.test_workflow -v        (aus dem Repo-Wurzelordner)

Die Tests, die die Vorlagen-Datenbank brauchen, werden uebersprungen, wenn
sie fehlt (z. B. auf einer Miet-GPU) -- die reinen Logik-Tests laufen
ueberall.
"""

import json
import random
import sys
import unittest
from pathlib import Path

WURZEL = Path(__file__).parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "bewertung"))

from workflow import instruktionen, katalog, kurzschrift as ks, mutationen as mu, vorlagen  # noqa: E402
from workflow.baue_datensatz import pruefe_split  # noqa: E402
from workflow.tokenizer_workflow import WorkflowTokenizer  # noqa: E402
from tore import pruefe_alle_tore  # noqa: E402

DB_DA = katalog.DB_STANDARD.exists()

BEISPIEL = {
    "name": "Lead-Erinnerung",
    "nodes": [
        {"name": "Formular", "type": "n8n-nodes-base.formTrigger", "typeVersion": 2.2},
        {"name": "Warte 24h", "type": "n8n-nodes-base.wait", "typeVersion": 1.1},
        {"name": "Erinnerung senden", "type": "n8n-nodes-base.emailSend", "typeVersion": 2.1},
        {"name": "Slack", "type": "n8n-nodes-base.slack", "typeVersion": 2.2},
    ],
    "connections": {
        "Formular": {"main": [[{"node": "Warte 24h", "type": "main", "index": 0}]]},
        "Warte 24h": {"main": [[{"node": "Erinnerung senden", "type": "main", "index": 0}],
                               [{"node": "Slack", "type": "main", "index": 0}]]},
    },
}


class Kurzschrift(unittest.TestCase):
    def test_rundlauf_erhaelt_typen_und_kanten(self):
        # Fassung 2 (Vorgabe): ohne Namen -- Kanten ueber Knoten-Index vergleichen
        text = ks.serialisiere(BEISPIEL)
        self.assertNotIn("Erinnerung senden", text)
        wf = ks.rendere(text)
        self.assertEqual([n["type"] for n in wf["nodes"]], [n["type"] for n in BEISPIEL["nodes"]])
        self.assertEqual(ks.kanten_index(wf), ks.kanten_index(BEISPIEL))
        self.assertEqual(wf["nodes"][0]["typeVersion"], 2.2)
        self.assertEqual(wf["nodes"][2]["name"], "Send Email")   # Katalog-Anzeigename
        self.assertTrue(pruefe_alle_tore(json.dumps(wf))["alle_bestanden_ohne_import"])
        # Fassung 1: mit Namen, Kanten namensgleich
        wf1 = ks.rendere(ks.serialisiere(BEISPIEL, mit_namen=True))
        self.assertEqual(ks.kanten_menge(wf1), ks.kanten_menge(BEISPIEL))

    def test_fassung_3_ohne_nummern(self):
        text = ks.serialisiere(BEISPIEL)
        self.assertEqual(text.splitlines()[1], "n8n-nodes-base.formTrigger@2.2")
        self.assertIn("n2 >1 n4", text)                    # Kanten behalten absolute Nummern
        wf = ks.rendere(text)
        self.assertEqual(ks.kanten_index(wf), ks.kanten_index(BEISPIEL))
        self.assertEqual(ks.kanten_typen(wf), ks.kanten_typen(BEISPIEL))
        # Fassung 2 (Nummern) und 1 (Namen) rendern auf dasselbe Ergebnis
        self.assertEqual(json.dumps(ks.rendere(ks.serialisiere(BEISPIEL, mit_nummern=True))),
                         json.dumps(wf))
        # ein Typ, der mit "n8" beginnt, ist keine Nummer -- kein Fehlparse
        wf2 = ks.rendere("wf\nn8n-nodes-base.set@3.4\nn8n-nodes-base.code@2\nn1 > n2")
        self.assertEqual([n["type"] for n in wf2["nodes"]], ["n8n-nodes-base.set", "n8n-nodes-base.code"])

    def test_gleiche_typen_bekommen_eindeutige_namen(self):
        wf = ks.rendere("wf\nn1 n8n-nodes-base.set@3.4\nn2 n8n-nodes-base.set@3.4\nn1 > n2")
        self.assertEqual([n["name"] for n in wf["nodes"]], ["Set", "Set 2"])
        self.assertTrue(pruefe_alle_tore(json.dumps(wf))["alle_bestanden_ohne_import"])

    def test_ausgang_index_und_verbindungstyp(self):
        text = ks.serialisiere(BEISPIEL, mit_nummern=True)
        self.assertIn("n2 >1 n4", text)          # zweiter Ausgang
        self.assertIn("n1 > n2", text)           # main, Ausgang 0 -> kurz
        wf = ks.rendere("wf x\nn1 a.b@1 A\nn2 a.c@1 B\nn1 ai_tool> n2")
        self.assertEqual(list(wf["connections"]["A"].keys()), ["ai_tool"])

    def test_rendern_ist_deterministisch(self):
        text = ks.serialisiere(BEISPIEL)
        self.assertEqual(json.dumps(ks.rendere(text)), json.dumps(ks.rendere(text)))

    def test_unlesbare_zeile_wird_abgewiesen(self):
        with self.assertRaises(ks.KurzschriftFehler):
            ks.rendere("wf x\nn1 a.b@1 A\nhier steht Unsinn")
        with self.assertRaises(ks.KurzschriftFehler):
            ks.rendere("wf x\nn1 a.b@1 A\nn1 > n9")   # Kante zu unbekanntem Knoten
        with self.assertRaises(ks.KurzschriftFehler):
            ks.rendere("n1 a.b@1 A")                  # wf-Zeile fehlt

    def test_erfundener_typ_faellt_an_tor_2(self):
        wf = ks.rendere("wf x\nn1 n8n-nodes-base.webhook@2 A\nn2 n8n-nodes-base.sendEmail@1 B\nn1 > n2")
        r = pruefe_alle_tore(json.dumps(wf))
        self.assertTrue(r["tor1_json"] and r["tor3_verbindungen"])
        self.assertFalse(r["tor2_struktur"])


class Instruktionen(unittest.TestCase):
    def setUp(self):
        self.kat = katalog.lade()

    def test_deterministisch_und_nennt_bausteine(self):
        a = instruktionen.erzeuge(BEISPIEL, {"use_cases": ["remind leads"]}, self.kat, "s:1", True)
        b = instruktionen.erzeuge(BEISPIEL, {"use_cases": ["remind leads"]}, self.kat, "s:1", True)
        self.assertEqual(a, b)
        self.assertIn("Send Email", a)
        self.assertIn("Slack", a)
        self.assertIn("Form Trigger", a)

    def test_mutante_bekommt_keinen_use_case(self):
        text = instruktionen.erzeuge(BEISPIEL, {"use_cases": ["remind leads"]}, self.kat, "s:2", False)
        self.assertNotIn("remind leads", text)


class Mutationen(unittest.TestCase):
    def setUp(self):
        self.kat = katalog.lade()
        self.gruppen = mu.tauschgruppen(self.kat)
        self.versionen = {"n8n-nodes-base.emailSend": "2.1", "n8n-nodes-base.slack": "2.2"}
        self.vorlage = {"id": 1, "wf": BEISPIEL, "metadata": {}}

    def test_typentausch_bleibt_gleichartig(self):
        rng = random.Random(3)
        for _ in range(30):
            neu = mu.tausche_typ(BEISPIEL, rng, self.kat, self.gruppen, self.versionen)
            self.assertIsNotNone(neu)
            for alt_n, neu_n in zip(BEISPIEL["nodes"], neu["nodes"]):
                a, b = self.kat[alt_n["type"]], self.kat[neu_n["type"]]
                self.assertEqual((a["kategorie"], a["ausloeser"], a["tool"]),
                                 (b["kategorie"], b["ausloeser"], b["tool"]))
            self.assertTrue(mu.ist_gueltig(neu))

    def test_ohne_namen_wird_umbenennen_ausgelassen(self):
        rng = random.Random(9)
        for _ in range(20):
            m = mu.mutiere(BEISPIEL, rng, self.kat, self.gruppen, self.versionen, mit_namen=False)
            if m is not None:
                self.assertNotIn("umbenenne", m[1])

    def test_blatt_entfernen_laesst_keine_haengende_kante(self):
        neu = mu.entferne_blatt(BEISPIEL, random.Random(1), self.kat)
        namen = {n["name"] for n in neu["nodes"]}
        for _, _, _, ziel in ks.kanten_menge(neu):
            self.assertIn(ziel, namen)
        self.assertTrue(mu.ist_gueltig(neu))

    def test_umbenennen_zieht_kanten_mit(self):
        neu = mu.umbenenne(BEISPIEL, random.Random(5), self.kat)
        self.assertTrue(mu.ist_gueltig(neu))
        self.assertEqual(len(ks.kanten_menge(neu)), len(ks.kanten_menge(BEISPIEL)))

    def test_mutanten_sind_verschieden_gueltig_und_deterministisch(self):
        a = mu.erzeuge_mutanten(self.vorlage, 8, 7, self.kat, self.gruppen, self.versionen)
        b = mu.erzeuge_mutanten(self.vorlage, 8, 7, self.kat, self.gruppen, self.versionen)
        texte = [ks.serialisiere(m["wf"]) for m in a]
        self.assertEqual(texte, [ks.serialisiere(m["wf"]) for m in b])
        self.assertEqual(len(texte), len(set(texte)))
        self.assertNotIn(ks.serialisiere(BEISPIEL), texte)
        for m in a:
            self.assertTrue(mu.ist_gueltig(m["wf"]))


class Split(unittest.TestCase):
    def test_leck_ueber_id_bricht_ab(self):
        t = [{"quelle_id": 1, "kurzschrift": "wf a"}]
        v = [{"quelle_id": 1, "kurzschrift": "wf b"}]
        with self.assertRaises(SystemExit):
            pruefe_split(t, v, [])

    def test_leck_ueber_text_bricht_ab(self):
        t = [{"quelle_id": 1, "kurzschrift": "wf a\nn1 x.y@1 A"}]
        v = [{"quelle_id": 2, "kurzschrift": "wf a\nn1 x.y@1 A"}]
        with self.assertRaises(SystemExit):
            pruefe_split(t, v, [])

    def test_sauberer_split_geht_durch(self):
        pruefe_split([{"quelle_id": 1, "kurzschrift": "a"}],
                     [{"quelle_id": 2, "kurzschrift": "b"}],
                     [{"quelle_id": 3, "kurzschrift": "c"}])


class Tokenizer(unittest.TestCase):
    def test_rundlauf_und_sondertoken(self):
        text = ks.serialisiere(BEISPIEL) * 20
        tok = WorkflowTokenizer.trainiere(text, 400)
        self.assertEqual(tok.groesse, 400)
        ids = tok.kodiere_beispiel("Baue etwas.", ks.serialisiere(BEISPIEL))
        self.assertEqual(ids[0], tok.instr_id)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertIn(tok.wf_id, ids)
        self.assertEqual(tok.dekodiere_antwort(ids), ks.serialisiere(BEISPIEL))
        # Sondertoken duerfen im BPE-Bereich nie vorkommen
        for i in ids[1:-1]:
            if i not in (tok.wf_id,):
                self.assertLess(i, tok.instr_id)


@unittest.skipUnless(DB_DA, "Vorlagen-Datenbank nicht vorhanden")
class EchteVorlagen(unittest.TestCase):
    def test_hundert_vorlagen_rundlauf_exakt(self):
        for v in vorlagen.lade(hoechstens=100):
            text = ks.serialisiere(v["wf"])
            wf = ks.rendere(text)
            self.assertEqual([n["type"] for n in wf["nodes"]], [n["type"] for n in v["wf"]["nodes"]])
            self.assertEqual(ks.kanten_index(wf), ks.kanten_index(v["wf"]))
            self.assertTrue(pruefe_alle_tore(json.dumps(wf))["alle_bestanden_ohne_import"])
            self.assertFalse(any(n["type"] == "n8n-nodes-base.stickyNote" for n in wf["nodes"]))
            wf1 = ks.rendere(ks.serialisiere(v["wf"], mit_namen=True))
            self.assertEqual(ks.kanten_menge(wf1), ks.kanten_menge(v["wf"]))


if __name__ == "__main__":
    unittest.main()
