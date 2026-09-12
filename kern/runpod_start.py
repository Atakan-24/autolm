"""
GPU MIETEN, TRAINIEREN, WIEDER FREIGEBEN -- der fehlende Schritt vor gpu_bootstrap.sh.

`kern/gpu_bootstrap.sh` beschreibt, was AUF einer GPU-Maschine passiert.
Dieses Skript besorgt die Maschine: es fragt RunPods API nach Preisen,
rechnet die Kosten fuer den geplanten Lauf aus, mietet den Rechner, startet
das Bootstrap-Skript und gibt ihn danach wieder frei.

    python kern/runpod_start.py preis                     # was wuerde es kosten
    python kern/runpod_start.py starten                   # Trockenlauf, startet NICHTS
    python kern/runpod_start.py starten --wirklich-starten # mietet wirklich (kostet Geld)
    python kern/runpod_start.py zustand                   # laufende Pods + Guthaben
    python kern/runpod_start.py stoppen --pod <id>        # beenden und freigeben

DREI DINGE, DIE HIER BEWUSST SO GEBAUT SIND:

1. **Der Trockenlauf ist die Vorgabe.** `starten` ohne `--wirklich-starten`
   rechnet, prueft und zeigt den fertigen Aufruf -- und mietet nichts.
   Dieselbe Bauart wie `scripts/voicemail-setup.mjs` und die uebrigen
   ausgabewirksamen Skripte in diesem Setup: wer Geld ausgibt, tut das mit
   einem zweiten, bewussten Handgriff.

2. **Das Guthaben ist ein hartes Tor, kein Hinweis.** Reicht das Guthaben
   nicht fuer den geschaetzten Lauf plus Puffer, bricht das Skript ab, BEVOR
   ein Pod entsteht. Grund: ein Pod, dem mitten im Lauf das Geld ausgeht,
   wird von RunPod beendet -- die Checkpoints liegen dann auf einer Maschine,
   die es nicht mehr gibt. Ein zu frueher Abbruch kostet nichts, ein zu
   spaeter kostet den ganzen Lauf.

3. **Die Mietmaschine bekommt keine Zugangsdaten.** Kein SSH-Schluessel
   dieses Servers, kein GitHub-Token, kein API-Schluessel wird
   weitergereicht -- sie zieht nur das oeffentliche Repo. Steht schon so im
   Kopf von `gpu_bootstrap.sh` und gilt hier genauso: die Checkpoints holt
   man sich von aussen ab, statt der Fremdmaschine Schluessel zu geben.

Der Schluessel kommt aus `.env.local` (gitignored) oder aus der Umgebung.
Er wird nie ausgegeben -- auch nicht gekuerzt, auch nicht im Fehlerfall.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WURZEL = Path(__file__).parent.parent
ENV_DATEI = WURZEL / ".env.local"
API = "https://api.runpod.io/graphql"

# Das Bild, das auf der Mietmaschine laeuft. RunPods eigenes PyTorch-Bild --
# bringt CUDA-torch mit, weshalb gpu_bootstrap.sh Schritt 2 dort nichts
# nachinstallieren muss (und genau das auch prueft, statt es anzunehmen).
BILD = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Sicherheitspuffer auf die Kostenschaetzung. Eine Schaetzung, die exakt
# aufgeht, ist keine Schaetzung -- und bei RunPod endet "Guthaben leer"
# nicht mit einer Warnung, sondern mit einem beendeten Pod.
PUFFER = 1.5


def lade_schluessel() -> str:
    schluessel = os.environ.get("RUNPOD_API_KEY")
    if not schluessel and ENV_DATEI.exists():
        for zeile in ENV_DATEI.read_text(encoding="utf8").splitlines():
            if zeile.startswith("RUNPOD_API_KEY="):
                schluessel = zeile.split("=", 1)[1].strip()
                break
    if not schluessel:
        raise SystemExit(
            f"RUNPOD_API_KEY fehlt (weder in der Umgebung noch in {ENV_DATEI}).\n"
            "Anlegen: https://www.runpod.io/console/user/settings -> API Keys"
        )
    return schluessel


def frage(schluessel: str, query: str, variables: dict | None = None) -> dict:
    daten = json.dumps({"query": query, "variables": variables or {}}).encode()
    anfrage = urllib.request.Request(
        API, data=daten,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {schluessel}",
                 # OHNE eigenen User-Agent antwortet RunPod mit HTTP 403 --
                 # der Vorgabewert `Python-urllib/3.x` wird vom vorgelagerten
                 # Schutz abgewiesen. Gemessen am 12.09.2026: identischer
                 # Schluessel, identischer Body, curl 200 / urllib 403.
                 # Der Fehler sieht aus wie ein ungueltiger Schluessel und ist
                 # keiner -- deshalb steht der Grund hier und nicht im Commit.
                 "User-Agent": "autolm/1.0 (+https://github.com/Atakan-24/autolm)"},
    )
    try:
        with urllib.request.urlopen(anfrage, timeout=60) as antwort:
            ergebnis = json.loads(antwort.read())
    except urllib.error.HTTPError as e:
        # Der Schluessel steht im Header, nicht im Text -- trotzdem nur die
        # Statuszeile ausgeben, nie den ganzen Request.
        raise SystemExit(f"RunPod antwortet mit HTTP {e.code}. Schluessel gueltig?")
    if ergebnis.get("errors"):
        raise SystemExit("RunPod-Fehler: " +
                         "; ".join(f.get("message", "?") for f in ergebnis["errors"]))
    return ergebnis.get("data") or {}


def guthaben(schluessel: str) -> float:
    d = frage(schluessel, "query { myself { clientBalance } }")
    return float((d.get("myself") or {}).get("clientBalance") or 0.0)


def gpu_liste(schluessel: str) -> list[dict]:
    d = frage(schluessel, """
        query { gpuTypes {
            id displayName memoryInGb communityCloud secureCloud
            lowestPrice(input:{gpuCount:1}) { uninterruptablePrice }
        } }""")
    aus = []
    for g in d.get("gpuTypes", []):
        preis = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        if preis:
            aus.append({"id": g["id"], "name": g["displayName"],
                        "vram": g.get("memoryInGb"), "preis": float(preis)})
    return sorted(aus, key=lambda x: x["preis"])


def waehle_gpu(gpus: list[dict], mindest_vram: int, wunsch: str | None) -> dict:
    if wunsch:
        treffer = [g for g in gpus if wunsch.lower() in g["name"].lower() or g["id"] == wunsch]
        if not treffer:
            raise SystemExit(f"Keine GPU passt auf {wunsch!r}. Verfuegbar:\n" +
                             "\n".join(f"  {g['name']} ({g['preis']:.3f} $/h)" for g in gpus[:12]))
        return treffer[0]
    passend = [g for g in gpus if (g["vram"] or 0) >= mindest_vram]
    if not passend:
        raise SystemExit(f"Keine GPU mit >= {mindest_vram} GB VRAM verfuegbar.")
    return passend[0]


def schaetze_stunden(schritte: int, s_pro_schritt: float) -> float:
    return schritte * s_pro_schritt / 3600.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("befehl", choices=["preis", "starten", "zustand", "stoppen"])
    p.add_argument("--schritte", type=int, default=20000,
                   help="geplante Trainingsschritte (Vorgabe: 20.000 fuer die 17M-Konfiguration)")
    p.add_argument("--s-pro-schritt", type=float, default=0.35,
                   help="GESCHAETZTE Sekunden je Schritt auf der GPU. Vorgabe 0,35 s -- "
                        "grob 10x schneller als die auf diesem Server GEMESSENEN 3,5 s/Schritt "
                        "(4 CPU-Kerne). Eine Schaetzung, keine Messung: nach dem ersten "
                        "echten Lauf durch den gemessenen Wert ersetzen.")
    p.add_argument("--mindest-vram", type=int, default=16)
    p.add_argument("--gpu", default=None, help="Name oder ID statt 'billigste passende'")
    p.add_argument("--pod", default=None, help="Pod-ID fuer 'stoppen'")
    p.add_argument("--wirklich-starten", action="store_true",
                   help="ohne diesen Schalter wird NICHTS gemietet")
    args = p.parse_args()

    schluessel = lade_schluessel()

    if args.befehl == "zustand":
        d = frage(schluessel, """
            query { myself { clientBalance currentSpendPerHr
                pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }""")
        m = d.get("myself") or {}
        print(f"Guthaben:            {float(m.get('clientBalance') or 0):.2f} $")
        print(f"laufende Kosten:     {float(m.get('currentSpendPerHr') or 0):.3f} $/h")
        pods = m.get("pods") or []
        print(f"Pods:                {len(pods)}")
        for pod in pods:
            gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
            print(f"  {pod['id']}  {pod.get('desiredStatus')}  {gpu}  "
                  f"{float(pod.get('costPerHr') or 0):.3f} $/h  {pod.get('name')}")
        if not pods:
            print("  (keiner -- es laeuft nichts, es kostet nichts)")
        return

    if args.befehl == "stoppen":
        if not args.pod:
            raise SystemExit("--pod <id> fehlt (IDs zeigt `zustand`)")
        frage(schluessel, "mutation($id: String!) { podTerminate(input: {podId: $id}) }",
              {"id": args.pod})
        print(f"Pod {args.pod} beendet und freigegeben.")
        print("Gegenprobe: python kern/runpod_start.py zustand")
        return

    gpus = gpu_liste(schluessel)
    gpu = waehle_gpu(gpus, args.mindest_vram, args.gpu)
    stunden = schaetze_stunden(args.schritte, args.s_pro_schritt)
    kosten = stunden * gpu["preis"]
    stand = guthaben(schluessel)

    print(f"GPU:            {gpu['name']} ({gpu['vram']} GB VRAM)")
    print(f"Preis:          {gpu['preis']:.3f} $/h")
    print(f"geplant:        {args.schritte:,} Schritte a {args.s_pro_schritt} s "
          f"(GESCHAETZT) = {stunden:.1f} h")
    print(f"Kosten:         {kosten:.2f} $   (mit Puffer x{PUFFER}: {kosten * PUFFER:.2f} $)")
    print(f"Guthaben:       {stand:.2f} $")
    print("\ndrei billigste Alternativen:")
    for g in gpus[:3]:
        print(f"  {g['name']:<22} {g['vram']:>3} GB  {g['preis']:.3f} $/h")

    if args.befehl == "preis":
        return

    if stand < kosten * PUFFER:
        print(f"\nABBRUCH: Guthaben {stand:.2f} $ deckt {kosten * PUFFER:.2f} $ nicht.")
        print("Es wurde nichts gemietet und nichts berechnet.")
        print("Aufladen: https://www.runpod.io/console/user/billing -> Add Credits")
        sys.exit(2)

    startbefehl = ("bash -lc 'curl -fsSL "
                   "https://raw.githubusercontent.com/Atakan-24/autolm/master/kern/gpu_bootstrap.sh "
                   "-o /workspace/gpu_bootstrap.sh && bash /workspace/gpu_bootstrap.sh'")

    if not args.wirklich_starten:
        print("\nTROCKENLAUF -- es wurde NICHTS gemietet.")
        print("Der Pod waere:")
        print(f"  Bild:     {BILD}")
        print(f"  GPU:      {gpu['name']} x1")
        print(f"  Start:    {startbefehl}")
        print("\nWirklich starten:  python kern/runpod_start.py starten --wirklich-starten")
        return

    d = frage(schluessel, """
        mutation($gpu: String!, $bild: String!, $befehl: String!) {
          podFindAndDeployOnDemand(input: {
            cloudType: ALL, gpuCount: 1, gpuTypeId: $gpu, imageName: $bild,
            name: "autolm-stufe2", volumeInGb: 40, containerDiskInGb: 20,
            dockerArgs: $befehl, ports: "22/tcp", supportPublicIp: true
          }) { id costPerHr machine { gpuDisplayName } }
        }""", {"gpu": gpu["id"], "bild": BILD, "befehl": startbefehl})
    pod = d.get("podFindAndDeployOnDemand") or {}
    if not pod.get("id"):
        raise SystemExit("Kein Pod entstanden -- RunPod hat nichts zurueckgegeben.")
    print(f"\nPod laeuft: {pod['id']}  ({float(pod.get('costPerHr') or 0):.3f} $/h)")
    print("Ab jetzt laeuft die Uhr. Beenden:")
    print(f"  python kern/runpod_start.py stoppen --pod {pod['id']}")


if __name__ == "__main__":
    main()
