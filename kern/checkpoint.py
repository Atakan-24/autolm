"""Checkpoint-Speicherung mit zwei rotierenden Slots.

Modell, Optimierer, CPU-Zufallszustaende und Datenposition werden gemeinsam
gespeichert. Temp-Dateien ersetzen bestehende Dateien; Pruefsummen erlauben
verifizierten Rueckfall. Beschaedigte Metadaten bleiben ein Fehler.
Nur eigene, vertrauenswuerdige Checkpoints laden: torch.load nutzt Pickle."""

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
        meta.setdefault("slot_pruefsummen", {})[str(naechster_slot)] = pruefsumme
        meta["zeit"] = inhalt["zeit"]
        self._speichere_meta(meta)

        if verlust_log:
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
        if not meta:
            return 0, 0
        if type(slot) is not int or slot not in (0, 1):
            raise RuntimeError('Checkpoint-Metadaten enthalten keinen gueltigen Slot')
        checksums = meta.get('slot_pruefsummen', {})
        if not isinstance(checksums, dict):
            raise RuntimeError('Checkpoint-Pruefsummen sind beschaedigt')
        # Legacy metadata verifies only the latest slot. Never deserialize
        # an unchecked fallback just because it exists.
        checksums = {str(slot): meta.get('pruefsumme'), **checksums}
        pfad = None
        for kandidat in (slot, 1 - slot):
            datei = self._slot_pfad(kandidat)
            expected = checksums.get(str(kandidat))
            if datei.exists() and expected and self._pruefsumme(datei) == expected:
                pfad = datei
                if kandidat != slot:
                    print(f'WARNUNG: verwende verifizierten Ersatz-Slot {kandidat}')
                break
        if pfad is None:
            raise RuntimeError('Kein Checkpoint mit gueltiger Pruefsumme verfuegbar')

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
            try:
                meta = json.loads(self.meta_datei.read_text(encoding="utf8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError('Checkpoint-Metadaten sind nicht lesbar') from exc
            if not isinstance(meta, dict) or not meta:
                raise RuntimeError('Checkpoint-Metadaten sind kein gueltiges Objekt')
            return meta
        if any(self._slot_pfad(i).exists() for i in (0, 1)):
            raise RuntimeError('Checkpoint-Dateien vorhanden, aber Metadaten fehlen')
        return {}

    def _speichere_meta(self, meta: dict):
        temp = self.meta_datei.with_suffix(".tmp")
        temp.write_text(json.dumps(meta, indent=2), encoding="utf8")
        temp.replace(self.meta_datei)
