"""
ABSTURZSICHERES CHECKPOINTING.

Der Teil, der ueber Erfolg oder Totalverlust entscheidet. Colab-Sitzungen
brechen ohne Vorwarnung ab (Zeitlimit, Verbindungsabbruch, Kontingent
erschoepft). Ohne diese Datei waere ein 3-Stunden-Trainingslauf beim
Abbruch in Minute 179 komplett verloren.

VIER ENTSCHEIDUNGEN, JEDE MIT EINEM ECHTEN FEHLERFALL DAHINTER:

1. ATOMAR SCHREIBEN (temp-Datei + os.replace), NIE IN-PLACE.
   Ein Kill mitten im torch.save() liesse eine halb geschriebene, kaputte
   Datei zurueck -- und genau die waere dann der EINZIGE Checkpoint.

2. ZWEI ROTIERENDE SLOTS statt einem.
   os.replace() ist auf dem LOKALEN Dateisystem atomar. Auf Google Drive
   (FUSE-Schicht, wie Colab es einhaengt) gilt diese Garantie NICHT mehr
   zwingend. Mit zwei Slots ueberlebt mindestens einer jeden denkbaren
   Fehlschlag mitten im Schreiben.

3. OPTIMIZER-STATE, RNG-ZUSTAENDE UND DATENPOSITION gehoeren mit in den
   Checkpoint, nicht nur die Modellgewichte. Ohne die Datenposition faengt
   ein Resume wieder bei Token 0 an -- das Training liefe nicht laenger,
   es liefe im Kreis. AdamW braucht seine Momente, sonst beginnt jede
   Stellschraube nach dem Resume wieder bei "erster Schritt".

4. PRUEFSUMME MITSPEICHERN. Eine Datei, die vollstaendig geschrieben wurde,
   aber Bitfehler beim Uebertragen zu/von Drive hat, faellt sonst erst beim
   Laden auf -- oder gar nicht, und das Training liefe mit stillschweigend
   kaputten Gewichten weiter.

    python kern/test_checkpoint.py    -- Kill-und-Resume-Test
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


class Checkpointer:
    def __init__(self, ordner: Path, praefix: str = "ckpt"):
        self.ordner = Path(ordner)
        self.ordner.mkdir(parents=True, exist_ok=True)
        self.praefix = praefix
        self.meta_datei = self.ordner / f"{praefix}_meta.json"

    def _slot_pfad(self, slot: int) -> Path:
        return self.ordner / f"{self.praefix}_{slot}.pt"

    def speichere(self, modell, optimierer, schritt: int, datenposition: int,
                   verlust_log: list[float] | None = None):
        """
        Schreibt einen neuen Checkpoint in den JEWEILS ANDEREN Slot als
        den zuletzt guten -- der alte bleibt unberuehrt liegen, bis der
        neue nachweislich vollstaendig und gueltig geschrieben ist.
        """
        meta = self._lade_meta()
        naechster_slot = 1 - meta.get("guter_slot", -1) if meta.get("guter_slot", -1) in (0, 1) else 0

        inhalt = {
            "modell": modell.state_dict(),
            "optimierer": optimierer.state_dict(),
            "schritt": schritt,
            "datenposition": datenposition,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "zeit": time.time(),
        }

        ziel = self._slot_pfad(naechster_slot)
        temp = ziel.with_suffix(".tmp")
        torch.save(inhalt, temp)
        pruefsumme = self._pruefsumme(temp)
        temp.replace(ziel)          # atomar auf dem lokalen Dateisystem

        meta["guter_slot"] = naechster_slot
        meta["schritt"] = schritt
        meta["datenposition"] = datenposition
        meta["pruefsumme"] = pruefsumme
        meta["zeit"] = inhalt["zeit"]
        self._speichere_meta(meta)

        if verlust_log is not None:
            with open(self.ordner / f"{self.praefix}_verlust.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({"schritt": schritt, "verlust": verlust_log[-1]}) + "\n")

        return ziel

    def lade_neuesten(self, modell, optimierer):
        """
        Gibt (schritt, datenposition) zurueck, oder (0, 0), wenn es noch
        keinen gueltigen Checkpoint gibt -- dann startet das Training von
        vorn, kein Fehler.
        """
        meta = self._lade_meta()
        slot = meta.get("guter_slot")
        if slot is None:
            return 0, 0

        pfad = self._slot_pfad(slot)
        if not pfad.exists():
            return 0, 0

        # Pruefsumme VOR dem Laden pruefen -- ein Bitfehler soll nicht erst
        # als kryptischer torch.load-Fehler auffallen, sondern klar benannt
        # sein, mit der Moeglichkeit, auf den anderen Slot auszuweichen.
        if self._pruefsumme(pfad) != meta.get("pruefsumme"):
            anderer = self._slot_pfad(1 - slot)
            if anderer.exists():
                print(f"WARNUNG: Checkpoint in Slot {slot} hat falsche Pruefsumme "
                      f"-- weiche auf Slot {1 - slot} aus.")
                pfad = anderer
            else:
                raise RuntimeError(
                    f"Checkpoint {pfad} ist beschaedigt (Pruefsumme stimmt nicht) "
                    f"und es gibt keinen zweiten Slot zum Ausweichen."
                )

        # weights_only=False: seit PyTorch 2.6 ist die Vorgabe True, was das
        # Laden von numpy-RNG-Zustaenden blockiert (Sicherheitsmassnahme
        # gegen Code-Ausfuehrung beim Laden FREMDER Checkpoints). Hier ist
        # das unproblematisch -- der Checkpoint stammt ausschliesslich aus
        # der eigenen speichere()-Methode, nie von aussen. Gefunden durch
        # den Kill-und-Resume-Test: er hat NICHTS mit Kill-Timing zu tun,
        # der Fehler traete bei jedem Resume auf, egal ob nach einem Absturz
        # oder normal beendet.
        inhalt = torch.load(pfad, map_location="cpu", weights_only=False)
        modell.load_state_dict(inhalt["modell"])
        optimierer.load_state_dict(inhalt["optimierer"])
        torch.set_rng_state(inhalt["torch_rng"])
        np.random.set_state(inhalt["numpy_rng"])
        random.setstate(inhalt["python_rng"])

        return inhalt["schritt"], inhalt["datenposition"]

    def _pruefsumme(self, pfad: Path) -> str:
        h = hashlib.sha256()
        with open(pfad, "rb") as f:
            for stueck in iter(lambda: f.read(1 << 20), b""):
                h.update(stueck)
        return h.hexdigest()

    def _lade_meta(self) -> dict:
        if self.meta_datei.exists():
            return json.loads(self.meta_datei.read_text(encoding="utf8"))
        return {}

    def _speichere_meta(self, meta: dict):
        temp = self.meta_datei.with_suffix(".tmp")
        temp.write_text(json.dumps(meta, indent=2), encoding="utf8")
        temp.replace(self.meta_datei)
