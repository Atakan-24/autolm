"""
TESTS FUER DEN GPU-MIETER -- ohne Netz, ohne Konto, ohne einen Cent.

Geprueft wird genau das, was Geld kosten kann, wenn es falsch ist:
das Guthaben-Tor, die Kostenrechnung und die GPU-Auswahl. Der Netzaufruf
selbst wird nicht getestet -- ein Test, der wirklich einen Pod mietet,
waere ein Test, der Geld ausgibt.

    python -m unittest kern.test_runpod_start -v
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import runpod_start as rp  # noqa: E402

GPUS = [
    {"id": "a", "name": "RTX 4070 Ti", "vram": 12, "preis": 0.19},
    {"id": "b", "name": "RTX 4090", "vram": 24, "preis": 0.34},
    {"id": "c", "name": "A100 SXM", "vram": 80, "preis": 1.39},
]


class GpuAuswahl(unittest.TestCase):
    def test_billigste_mit_genug_vram(self):
        self.assertEqual(rp.waehle_gpu(GPUS, 16, None)["name"], "RTX 4090")
        self.assertEqual(rp.waehle_gpu(GPUS, 10, None)["name"], "RTX 4070 Ti")

    def test_wunsch_schlaegt_preis(self):
        self.assertEqual(rp.waehle_gpu(GPUS, 10, "A100")["name"], "A100 SXM")

    def test_unerfuellbarer_wunsch_bricht_ab_statt_zu_raten(self):
        # Eine GPU, die es nicht gibt, darf NICHT stillschweigend durch die
        # billigste ersetzt werden -- sonst mietet man etwas anderes als
        # bestellt und merkt es erst an der Rechnung.
        with self.assertRaises(SystemExit):
            rp.waehle_gpu(GPUS, 10, "H200")
        with self.assertRaises(SystemExit):
            rp.waehle_gpu(GPUS, 999, None)


class Kostenrechnung(unittest.TestCase):
    def test_stunden_aus_schritten(self):
        self.assertAlmostEqual(rp.schaetze_stunden(20000, 0.35), 1.9444, places=3)
        self.assertEqual(rp.schaetze_stunden(0, 0.35), 0.0)

    def test_puffer_ist_groesser_als_eins(self):
        # Ein Puffer <= 1 waere keiner. Der Wert selbst darf sich aendern,
        # die Richtung nicht.
        self.assertGreater(rp.PUFFER, 1.0)


class GuthabenTor(unittest.TestCase):
    """Das eigentliche Sicherheitsnetz: zu wenig Geld -> kein Pod, Exit 2."""

    def _lauf(self, stand: float, argv: list[str]):
        with mock.patch.object(rp, "lade_schluessel", return_value="x"), \
             mock.patch.object(rp, "gpu_liste", return_value=GPUS), \
             mock.patch.object(rp, "guthaben", return_value=stand), \
             mock.patch.object(rp, "frage") as gefragt, \
             mock.patch.object(sys, "argv", ["runpod_start.py"] + argv):
            puffer = io.StringIO()
            code = 0
            try:
                with redirect_stdout(puffer):
                    rp.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
            return code, puffer.getvalue(), gefragt

    def test_leeres_guthaben_bricht_ab_und_mietet_nichts(self):
        code, ausgabe, gefragt = self._lauf(0.0, ["starten", "--wirklich-starten"])
        self.assertEqual(code, 2)
        self.assertIn("ABBRUCH", ausgabe)
        gefragt.assert_not_called()   # keine einzige Mutation ging raus

    def test_knappes_guthaben_reicht_nicht(self):
        # Kosten 20000 * 0.35 s = 1,944 h * 0,34 $ = 0,66 $; mit Puffer 0,99 $.
        # 0,80 $ wuerde OHNE Puffer reichen -- und genau das soll es nicht.
        code, ausgabe, gefragt = self._lauf(0.80, ["starten", "--wirklich-starten"])
        self.assertEqual(code, 2)
        gefragt.assert_not_called()

    def test_trockenlauf_mietet_auch_bei_vollem_guthaben_nichts(self):
        code, ausgabe, gefragt = self._lauf(50.0, ["starten"])
        self.assertEqual(code, 0)
        self.assertIn("TROCKENLAUF", ausgabe)
        gefragt.assert_not_called()

    def test_preis_ist_immer_folgenlos(self):
        code, ausgabe, gefragt = self._lauf(50.0, ["preis"])
        self.assertEqual(code, 0)
        self.assertNotIn("TROCKENLAUF", ausgabe)
        gefragt.assert_not_called()


class SchluesselBleibtGeheim(unittest.TestCase):
    def test_kein_schluessel_in_der_ausgabe(self):
        geheim = "rpa_GEHEIM_NICHT_AUSGEBEN"
        with mock.patch.object(rp, "lade_schluessel", return_value=geheim), \
             mock.patch.object(rp, "gpu_liste", return_value=GPUS), \
             mock.patch.object(rp, "guthaben", return_value=0.0), \
             mock.patch.object(rp, "frage"), \
             mock.patch.object(sys, "argv", ["runpod_start.py", "starten"]):
            puffer = io.StringIO()
            try:
                with redirect_stdout(puffer):
                    rp.main()
            except SystemExit:
                pass
            self.assertNotIn(geheim, puffer.getvalue())
            self.assertNotIn("rpa_", puffer.getvalue())


if __name__ == "__main__":
    unittest.main()
