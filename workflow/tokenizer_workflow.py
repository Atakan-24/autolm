"""
TOKENIZER FUER DAS WORKFLOW-MODELL -- eigener BPE, eigene Sondertoken.

Nicht der TinyStories-Tokenizer (daten/tokenizer.pkl): dessen Merges
kennen "once upon a time", nicht "n8n-nodes-base.". Ein Tokenizer, der
`emailSend` in sieben Bytes zerhackt, macht das Auswendiglernen des
Vokabulars unnoetig schwer -- und einer, der jeden Typ als EIN Token
kennt, macht es unmoeglich zu verfehlen (siehe kurzschrift.py). BPE auf
dem eigenen Korpus liegt dazwischen: haeufige Typen werden zu wenigen
Token, seltene bleiben Stueckwerk -- das Modell muss beides koennen.

Drei Sondertoken, die im BPE nie entstehen koennen, weil sie oberhalb des
Merge-Bereichs liegen:

    <|instr|>  Instruktion  <|wf|>  Kurzschrift  <|eos|>

Beim Erzeugen bekommt das Modell alles bis <|wf|> vorgegeben und schreibt
bis <|eos|>. Trainiert wird der Merge-Teil mit kern/bpe_training_schnell,
kodiert mit kern/bpe_schnell -- dieselben Bausteine wie in Stufe 1.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kern"))
from bpe_schnell import SchnellerBPETokenizer  # noqa: E402
from bpe_training_schnell import trainiere_schnell  # noqa: E402

SPEZIAL = ["<|instr|>", "<|wf|>", "<|eos|>"]


class WorkflowTokenizer:
    def __init__(self, verschmelzungen: dict, vokabular: dict, spezial: dict[str, int]):
        self.bpe = SchnellerBPETokenizer()
        self.bpe.verschmelzungen = verschmelzungen
        self.bpe.vokabular = vokabular
        self.bpe.rang = {paar: i for i, paar in enumerate(verschmelzungen)}
        self.spezial = spezial                      # name -> id
        self.spezial_id = {v: k for k, v in spezial.items()}
        self.groesse = max(spezial.values()) + 1     # = vokabular_groesse fuer das Modell

    # ---- Bau ----------------------------------------------------------
    @classmethod
    def trainiere(cls, text: str, vokabular_groesse: int = 4096, zeige: bool = False):
        ziel = vokabular_groesse - len(SPEZIAL)
        verschmelzungen, vokabular = trainiere_schnell(text, ziel, zeige=zeige)
        spezial = {name: ziel + i for i, name in enumerate(SPEZIAL)}
        return cls(verschmelzungen, vokabular, spezial)

    def speichere(self, pfad: Path) -> None:
        with open(pfad, "wb") as f:
            pickle.dump({"verschmelzungen": self.bpe.verschmelzungen,
                         "vokabular": self.bpe.vokabular,
                         "spezial": self.spezial}, f)

    @classmethod
    def lade(cls, pfad: Path):
        with open(pfad, "rb") as f:
            d = pickle.load(f)
        return cls(d["verschmelzungen"], d["vokabular"], d["spezial"])

    # ---- Kodieren -----------------------------------------------------
    @property
    def instr_id(self): return self.spezial["<|instr|>"]
    @property
    def wf_id(self): return self.spezial["<|wf|>"]
    @property
    def eos_id(self): return self.spezial["<|eos|>"]

    def kodiere_text(self, text: str) -> list[int]:
        return self.bpe.kodiere(text)

    def kodiere_beispiel(self, instruktion: str, kurzschrift: str) -> list[int]:
        return ([self.instr_id] + self.bpe.kodiere(instruktion)
                + [self.wf_id] + self.bpe.kodiere(kurzschrift) + [self.eos_id])

    def kodiere_prompt(self, instruktion: str) -> list[int]:
        """Alles bis einschliesslich <|wf|> -- ab hier schreibt das Modell."""
        return [self.instr_id] + self.bpe.kodiere(instruktion) + [self.wf_id]

    def dekodiere(self, ids: list[int]) -> str:
        """Sondertoken werden weggelassen, alles andere byteweise zurueckgebaut."""
        return self.bpe.dekodiere([i for i in ids if i not in self.spezial_id])

    def dekodiere_antwort(self, ids: list[int]) -> str:
        """Nur den Teil zwischen <|wf|> und <|eos|> -- die Kurzschrift."""
        if self.wf_id in ids:
            ids = ids[ids.index(self.wf_id) + 1:]
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        return self.dekodiere(ids)
