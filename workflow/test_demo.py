import unittest

from workflow.demo import waehle_beste


class DemoAuswahl(unittest.TestCase):
    def test_nimmt_die_erste_gueltige_probe(self):
        erste = {"gueltig": False, "id": 1}
        gueltige = {"gueltig": True, "id": 2}
        self.assertIs(waehle_beste([erste, gueltige, {"gueltig": True, "id": 3}]), gueltige)

    def test_behaelt_ehrlich_die_erste_ungueltige_probe(self):
        erste = {"gueltig": False, "id": 1}
        self.assertIs(waehle_beste([erste, {"gueltig": False, "id": 2}]), erste)

    def test_leere_liste_ist_ein_programmierfehler(self):
        with self.assertRaises(ValueError):
            waehle_beste([])


if __name__ == "__main__":
    unittest.main()
