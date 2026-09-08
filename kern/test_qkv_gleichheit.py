"""
BEWEIST, DASS DER SCHNELLE PFAD DASSELBE RECHNET WIE DER LEHRPFAD.

Warum dieser Test der wichtigste im Repo ist: eine Optimierung, die
nebenbei das Ergebnis aendert, ist kein schnelleres Modell, sondern ein
kaputtes mit Stoppuhr. Und sie faellt NICHT auf -- der Loss sinkt trotzdem,
der Text sieht plausibel aus, nur ist es ein anderes Modell als das
dokumentierte.

Deshalb wird hier nicht die Geschwindigkeit geprueft, sondern die
Gleichheit. Die Geschwindigkeit misst mess_speedup.py.

    python -m pytest kern/test_qkv_gleichheit.py -v
    python kern/test_qkv_gleichheit.py            (ohne pytest)
"""

import torch

from modell import (MehrKoepfe, MehrKoepfeSchnell, MiniGPT, uebertrage_gewichte)

TOLERANZ = 1e-5


def _paar(anzahl=4, dim=64, block=32, seed=0):
    torch.manual_seed(seed)
    langsam = MehrKoepfe(anzahl, dim, block).eval()
    schnell = MehrKoepfeSchnell(anzahl, dim, block).eval()
    uebertrage_gewichte(langsam, schnell)
    return langsam, schnell


def test_gleiche_ausgabe():
    """Kernaussage: gleiche Gewichte, gleiche Eingabe -> gleiche Ausgabe."""
    langsam, schnell = _paar()
    x = torch.randn(3, 32, 64)
    with torch.no_grad():
        a, b = langsam(x), schnell(x)
    abweichung = (a - b).abs().max().item()
    assert abweichung < TOLERANZ, f"Abweichung {abweichung:.2e} ueber {TOLERANZ:.0e}"


def test_kuerzere_sequenz():
    """Auch wenn die Sequenz kuerzer ist als das Kontextfenster.

    Der Lehrpfad schneidet die Maske mit [:T, :T] zu, der schnelle Pfad
    baut sie selbst. Waere eine der beiden Kanten falsch, faellt es NUR
    hier auf -- bei voller Laenge sehen beide gleich aus.
    """
    langsam, schnell = _paar(block=32)
    for T in (1, 2, 7, 31, 32):
        x = torch.randn(2, T, 64)
        with torch.no_grad():
            abweichung = (langsam(x) - schnell(x)).abs().max().item()
        assert abweichung < TOLERANZ, f"T={T}: Abweichung {abweichung:.2e}"


def test_verschiedene_kopfzahlen():
    for anzahl in (1, 2, 4, 8):
        langsam, schnell = _paar(anzahl=anzahl, dim=64, seed=anzahl)
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            abweichung = (langsam(x) - schnell(x)).abs().max().item()
        assert abweichung < TOLERANZ, f"{anzahl} Koepfe: {abweichung:.2e}"


def test_gradienten_gleich():
    """
    Nicht nur die Vorwaerts-, auch die Rueckwaertsrechnung.

    Ein Modell kann vorwaerts identisch sein und rueckwaerts falsche
    Gradienten liefern -- dann trainiert es anders, und das merkt man
    erst nach Stunden GPU-Zeit an einer Kurve, die nicht passt.
    """
    langsam, schnell = _paar()
    x = torch.randn(2, 16, 64)
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    langsam(x1).sum().backward()
    schnell(x2).sum().backward()
    abweichung = (x1.grad - x2.grad).abs().max().item()
    assert abweichung < TOLERANZ, f"Gradient weicht ab: {abweichung:.2e}"


def test_ganzes_modell():
    """Dieselbe Pruefung eine Ebene hoeher, ueber alle Schichten."""
    torch.manual_seed(1)
    a = MiniGPT(128, dim=64, koepfe=4, schichten=3, block=32, schnell=False).eval()
    torch.manual_seed(1)
    b = MiniGPT(128, dim=64, koepfe=4, schichten=3, block=32, schnell=True).eval()
    # Blockweise uebertragen -- die uebrigen Gewichte sind durch den
    # gleichen Seed schon identisch.
    for bl_a, bl_b in zip(a.bloecke, b.bloecke):
        uebertrage_gewichte(bl_a.aufmerksamkeit, bl_b.aufmerksamkeit)
        bl_b.denken.load_state_dict(bl_a.denken.state_dict())
        bl_b.norm1.load_state_dict(bl_a.norm1.state_dict())
        bl_b.norm2.load_state_dict(bl_a.norm2.state_dict())
    b.token_embedding.load_state_dict(a.token_embedding.state_dict())
    b.pos_embedding.load_state_dict(a.pos_embedding.state_dict())
    b.norm.load_state_dict(a.norm.state_dict())
    b.ausgabe.load_state_dict(a.ausgabe.state_dict())

    idx = torch.randint(0, 128, (2, 24))
    with torch.no_grad():
        abweichung = (a(idx)[0] - b(idx)[0]).abs().max().item()
    assert abweichung < TOLERANZ, f"Ganzes Modell weicht ab: {abweichung:.2e}"


def test_ablationsschalter_wirken():
    """
    Gegenprobe zur Ablation: die Schalter muessen das Modell WIRKLICH
    veraendern. Ein Schalter, der nichts tut, wuerde in Stufe 6 als
    "Residuals machen keinen Unterschied" gemessen -- ein Befund, der
    dann Unsinn waere.
    """
    voll = MiniGPT(64, dim=32, koepfe=2, schichten=2, block=16)
    assert voll.pos_embedding is not None
    ohne_pos = MiniGPT(64, dim=32, koepfe=2, schichten=2, block=16, mit_position=False)
    assert ohne_pos.pos_embedding is None
    assert ohne_pos.anzahl_parameter() < voll.anzahl_parameter()

    ohne_norm = MiniGPT(64, dim=32, koepfe=2, schichten=2, block=16, mit_norm=False)
    assert ohne_norm.anzahl_parameter() < voll.anzahl_parameter()

    ohne_res = MiniGPT(64, dim=32, koepfe=2, schichten=2, block=16, mit_residual=False)
    assert ohne_res.bloecke[0].mit_residual is False
    # Gleiche Parameterzahl -- Residuals haben keine Gewichte. Wirken
    # muessen sie trotzdem:
    assert ohne_res.anzahl_parameter() == voll.anzahl_parameter()
    torch.manual_seed(3)
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert not torch.allclose(voll(idx)[0], ohne_res(idx)[0])


if __name__ == "__main__":
    fehler = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK    {name}")
            except AssertionError as e:
                print(f"  FEHLT {name}: {e}")
                fehler += 1
    print(f"\n{'ALLE BESTANDEN' if not fehler else f'{fehler} FEHLGESCHLAGEN'}")
    raise SystemExit(1 if fehler else 0)
