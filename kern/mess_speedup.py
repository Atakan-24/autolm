"""
MISST, WIE VIEL DER FUSIONIERTE QKV-PFAD BRINGT.

Getrennt von test_qkv_gleichheit.py, und das mit Absicht: der Test
beantwortet "rechnet es dasselbe", dieses Skript "ist es schneller".
Zusammen in einer Datei wuerde die erste Frage von der zweiten verdeckt --
und die erste ist die wichtigere.

    python kern/mess_speedup.py
    python kern/mess_speedup.py --gross     (Konfiguration aus Stufe 2)
"""

import sys
import time

import torch

from modell import MiniGPT


def miss(modell, idx, ziele, runden=10, aufwaermen=3):
    """
    Aufwaermen NICHT weglassen: der erste Durchlauf enthaelt einmalige
    Kosten (Speicher anfordern, Kernel waehlen, bei CUDA das Laden). Wer
    ihn mitmisst, misst den Start, nicht den Betrieb -- und bekommt je
    nach Reihenfolge der Messung ein anderes Ergebnis.
    """
    opt = torch.optim.AdamW(modell.parameters(), lr=1e-4)

    def schritt():
        _, verlust = modell(idx, ziele)
        opt.zero_grad(set_to_none=True)
        verlust.backward()
        opt.step()

    for _ in range(aufwaermen):
        schritt()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runden):
        schritt()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runden


def main():
    gross = "--gross" in sys.argv
    if gross:
        dim, koepfe, schichten, block, batch, vok = 384, 6, 6, 256, 8, 8192
    else:
        dim, koepfe, schichten, block, batch, vok = 128, 4, 3, 96, 16, 512

    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Geraet:  {geraet}"
          + (f" ({torch.cuda.get_device_name(0)})" if geraet == "cuda"
             else f" ({torch.get_num_threads()} Threads)"))
    print(f"Aufbau:  dim={dim}, {schichten} Schichten, {koepfe} Koepfe, "
          f"block={block}, batch={batch}")

    torch.manual_seed(0)
    idx = torch.randint(0, vok, (batch, block), device=geraet)
    ziele = torch.randint(0, vok, (batch, block), device=geraet)

    ergebnisse = {}
    for name, schnell in [("Lehrpfad (Schleife)", False), ("fusioniertes QKV", True)]:
        torch.manual_seed(0)
        modell = MiniGPT(vok, dim=dim, koepfe=koepfe, schichten=schichten,
                         block=block, schnell=schnell).to(geraet)
        ergebnisse[name] = miss(modell, idx, ziele)
        print(f"  {name:<22} {ergebnisse[name] * 1000:8.1f} ms/Schritt")

    a, b = ergebnisse["Lehrpfad (Schleife)"], ergebnisse["fusioniertes QKV"]
    faktor = a / b
    print(f"\n  Faktor: {faktor:.2f}x")

    # 340M Token bei diesem Batch, wie in Stufe 2 geplant
    token_je_schritt = batch * block
    schritte = 340_000_000 // token_je_schritt
    print(f"\n  Hochgerechnet auf 340M Token ({schritte:,} Schritte):")
    print(f"    Lehrpfad:         {a * schritte / 3600:7.1f} h")
    print(f"    fusioniertes QKV: {b * schritte / 3600:7.1f} h")

    print(f"""
  EINORDNUNG: auf CPU ist der Gewinn klein, weil dort ohnehin ein Kern
  nach dem anderen rechnet und der Aufruf-Aufwand kaum ins Gewicht faellt.
  Auf GPU ist genau dieser Aufruf-Aufwand der Engpass -- {schichten * koepfe * 3}
  kleine Kernel-Starts je Vorwaertsdurchlauf gegen {schichten}. Der Faktor
  oben ist deshalb die UNTERGRENZE; die GPU-Zahl steht erst nach dem
  ersten Colab-Lauf hier.""".rstrip())


if __name__ == "__main__":
    main()
