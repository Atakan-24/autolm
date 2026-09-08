"""
STUFE 1c -- TINYSTORIES HERUNTERLADEN UND VORTOKENISIEREN.

Was hier passiert, in der Reihenfolge, in der es passiert:

  1. Ein Parquet-Shard von Hugging Face laden (~238 MB, echte TinyStories-
     Geschichten -- kein synthetischer Text mehr).
  2. Jede Geschichte einzeln mit dem SCHNELLEN, parallelen BPE-Encoder aus
     Stufe 1a/1b kodieren (kern/bpe_schnell.py, kern/bpe_parallel.py).
  3. Die Token-IDs als uint16 in eine MEMMAP-Datei anhaengen -- eine Datei
     auf der Platte, die wie ein riesiges Array benutzt wird, ohne je
     komplett in den Arbeitsspeicher zu passen.
  4. Den Parquet-Shard LOESCHEN, sobald er verarbeitet ist.

WARUM SCHRITT 4 (LOESCHEN): dieser Server hat 8,3 GB freie Platte. Vier
Shards gleichzeitig (~950 MB) plus die Memmap-Ausgabe waeren knapp, aber
mehrere Server-Prozesse teilen sich dieselbe Platte. Ein Shard nach dem
anderen zu verarbeiten und sofort zu loeschen haelt den Fussabdruck klein,
unabhaengig davon, wie viele Shards am Ende insgesamt geladen werden.

WARUM UINT16 STATT INT32/INT64: das Vokabular hat 8192 Eintraege, passt
also in 16 Bit (bis 65535). Halbiert die Dateigroesse gegenueber int32 --
bei 340M Token sind das 680 MB statt 1,36 GB.

TRENNZEICHEN ZWISCHEN GESCHICHTEN: Token-ID `EOS_ID` (die letzte freie
Vokabular-Stelle) markiert das Ende jeder Geschichte. Ohne das wuerde das
Modell zwei zufaellig aneinandergehaengte Geschichten wie eine einzige
lesen -- der Satzanfang der naechsten wirkt dann wie eine Fortsetzung.

    python daten/vortokenisiere.py --hoechstens-token 340_000_000
"""

import argparse
import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kern"))
from bpe_schnell import SchnellerBPETokenizer
from bpe_parallel import kodiere_parallel

WURZEL = Path(__file__).parent
ROH = WURZEL / "roh"
TOKENIZER_DATEI = WURZEL / "tokenizer.pkl"
MEMMAP_DATEI = WURZEL / "tinystories_train.bin"
META_DATEI = WURZEL / "tinystories_meta.json"

SHARD_URL = ("https://huggingface.co/api/datasets/roneneldan/TinyStories/"
             "parquet/default/train/{n}.parquet")
ANZAHL_SHARDS = 4  # gemessen: shards 0..3 existieren (HF-API-Abfrage 08.09.2026)


def lade_tokenizer():
    if not TOKENIZER_DATEI.exists():
        sys.exit(f"Tokenizer fehlt: {TOKENIZER_DATEI}. Erst trainieren.")
    with open(TOKENIZER_DATEI, "rb") as f:
        d = pickle.load(f)
    tok = SchnellerBPETokenizer()
    tok.verschmelzungen = d["verschmelzungen"]
    tok.vokabular = d["vokabular"]
    tok.rang = {paar: i for i, paar in enumerate(tok.verschmelzungen)}
    return tok


def lade_meta():
    if META_DATEI.exists():
        return json.loads(META_DATEI.read_text(encoding="utf8"))
    return {"verarbeitete_shards": [], "gesamt_token": 0, "gesamt_geschichten": 0}


def speichere_meta(meta):
    META_DATEI.write_text(json.dumps(meta, indent=2), encoding="utf8")


def lade_shard(n: int) -> Path:
    ziel = ROH / f"shard{n}.parquet"
    if ziel.exists():
        print(f"  Shard {n}: schon vorhanden ({ziel.stat().st_size / 1e6:.0f} MB)")
        return ziel
    ROH.mkdir(parents=True, exist_ok=True)
    url = SHARD_URL.format(n=n)
    print(f"  Shard {n}: lade von {url} ...")
    t0 = time.time()
    urllib.request.urlretrieve(url, ziel)
    print(f"    fertig in {time.time() - t0:.0f}s, {ziel.stat().st_size / 1e6:.0f} MB")
    return ziel


def verarbeite_shard(n: int, tok, eos_id: int, meta: dict) -> int:
    """Gibt die Anzahl neu geschriebener Token zurueck."""
    import pyarrow.parquet as pq
    import numpy as np

    pfad = lade_shard(n)
    tabelle = pq.read_table(pfad, columns=["text"])
    geschichten = [str(x) for x in tabelle.column("text")]
    print(f"  Shard {n}: {len(geschichten):,} Geschichten, "
          f"{sum(len(g) for g in geschichten):,} Zeichen")

    t0 = time.time()
    alle_ids = kodiere_parallel(tok, geschichten, prozesse=4, stapelgroesse=500)
    dt = time.time() - t0

    # Flach mit EOS-Trennzeichen, einmal als numpy-Array -- schneller als
    # Byte fuer Byte in die Memmap zu schreiben.
    flach = []
    for ids in alle_ids:
        flach.extend(ids)
        flach.append(eos_id)
    array = np.array(flach, dtype=np.uint16)

    modus = "r+b" if MEMMAP_DATEI.exists() else "wb"
    with open(MEMMAP_DATEI, modus) as f:
        f.seek(0, 2)  # ans Ende
        f.write(array.tobytes())

    zeichen = sum(len(g) for g in geschichten)
    print(f"    kodiert in {dt:.0f}s ({zeichen / 1e6 / dt:.2f} MB/s), "
          f"{len(array):,} Token geschrieben")

    meta["verarbeitete_shards"].append(n)
    meta["gesamt_token"] += len(array)
    meta["gesamt_geschichten"] += len(geschichten)
    speichere_meta(meta)

    pfad.unlink()
    print(f"    {pfad.name} geloescht (Platz gespart)")
    return len(array)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hoechstens-token", type=int, default=340_000_000,
                    help="Aufhoeren, sobald mindestens so viele Token geschrieben sind "
                         "(Chinchilla-optimal fuer das 17M-Modell aus dem Plan)")
    p.add_argument("--hoechstens-shards", type=int, default=ANZAHL_SHARDS)
    args = p.parse_args()

    tok = lade_tokenizer()
    eos_id = max(tok.vokabular.keys())  # letzte Vokabular-ID als Trennzeichen
    print(f"Tokenizer: {len(tok.vokabular)} Token, EOS-Marke = {eos_id}")

    meta = lade_meta()
    print(f"Bisheriger Stand: {meta['gesamt_token']:,} Token aus "
          f"{len(meta['verarbeitete_shards'])} Shard(s)\n")

    n = 0
    while (meta["gesamt_token"] < args.hoechstens_token
           and n < args.hoechstens_shards
           and n < ANZAHL_SHARDS):
        if n in meta["verarbeitete_shards"]:
            n += 1
            continue
        verarbeite_shard(n, tok, eos_id, meta)
        n += 1
        print(f"  Zwischenstand: {meta['gesamt_token']:,} / "
              f"{args.hoechstens_token:,} Token\n")

    print(f"FERTIG: {meta['gesamt_token']:,} Token, "
          f"{meta['gesamt_geschichten']:,} Geschichten, "
          f"{len(meta['verarbeitete_shards'])} Shard(s)")
    print(f"Memmap: {MEMMAP_DATEI} ({MEMMAP_DATEI.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
