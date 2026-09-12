#!/bin/bash
# STUFE 2 -- START AUF EINER FRISCH GEMIETETEN CLOUD-GPU.
#
# Ersetzt das Colab-Notebook durch etwas, das ohne Browser, ohne Google-Konto
# und ohne einen einzigen Klick laeuft: ein Skript, das auf einer leeren
# Linux-GPU-Maschine alles selbst herstellt und das Training startet.
#
# WARUM ES DAS GIBT: der Colab-Weg braucht zwingend einen Menschen, der sich
# in seinem eigenen Browser bei Google anmeldet -- das ist die eine Stelle,
# die kein Agent uebernehmen kann. Auf einer gemieteten Maschine gibt es
# diese Huerde nicht: SSH rein, Skript starten, fertig.
#
# CHECKPOINTS: bleiben absichtlich LOKAL auf der Maschine (Standard
# ~/autolm-checkpoints). Die Maschine bekommt bewusst KEINE Zugangsdaten
# fuer irgendeinen anderen Server -- wer die Checkpoints braucht, HOLT sie
# per scp von aussen ab. Eine gemietete Fremdmaschine mit eigenen
# Schluesseln auszustatten waere genau die Art Zugang, die man nicht
# verteilt, nur weil es bequem ist.
#
#   bash gpu_bootstrap.sh                 # volle 17M-Konfiguration
#   bash gpu_bootstrap.sh --test          # winziger Lauf, prueft nur die Kette
#
set -euo pipefail

REPO="${REPO:-https://github.com/Atakan-24/autolm.git}"
ARBEIT="${ARBEIT:-$HOME/autolm}"
CKPT="${CKPT:-$HOME/autolm-checkpoints}"
TOKEN_BUDGET="${TOKEN_BUDGET:-340000000}"
TESTLAUF=0
[ "${1:-}" = "--test" ] && TESTLAUF=1

echo "=== 0. Maschine ansehen (erst messen, dann rechnen) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "FEHLER: keine GPU sichtbar (nvidia-smi fehlt oder schlaegt fehl)." >&2
    echo "Auf dieser Maschine hat ein Trainingslauf keinen Sinn -- abbrechen." >&2
    exit 1
}
echo "CPU-Kerne: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GB   Platte frei: $(df -BG --output=avail "$HOME" | tail -1)"

echo
echo "=== 1. Repo holen oder aktualisieren (idempotent) ==="
# Gleiche Lehre wie im Colab-Notebook (Commit 3514e00): ein blindes
# `git clone` scheitert beim zweiten Lauf lautlos und man arbeitet
# unbemerkt auf einem alten Stand weiter.
if [ -d "$ARBEIT/.git" ]; then
    git -C "$ARBEIT" pull --ff-only
else
    git clone "$REPO" "$ARBEIT"
fi
cd "$ARBEIT"
echo "Stand: $(git rev-parse --short HEAD) -- $(git log -1 --format=%s)"

echo
echo "=== 2. Abhaengigkeiten ==="
python3 -m pip install --quiet --upgrade pip
# torch NICHT blind installieren: die meisten GPU-Images bringen eine
# passende, CUDA-gebaute Version schon mit. Ein `pip install torch`
# darueber kann sie durch eine CPU-Version ersetzen -- dann laeuft das
# Training auf der teuren GPU-Maschine auf der CPU, und das faellt erst
# an der Laufzeit auf.
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch vorhanden, CUDA sichtbar: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
else
    echo "torch fehlt oder sieht keine GPU -- installiere CUDA-Build ..."
    python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA weiterhin nicht sichtbar'; print(torch.__version__, torch.cuda.get_device_name(0))"
fi
python3 -m pip install --quiet numpy pyarrow

echo
echo "=== 3. Daten vortokenisieren (falls noch nicht da) ==="
if [ -f daten/tinystories_train.bin ] && [ -f daten/tinystories_meta.json ]; then
    echo "schon vorhanden: $(du -h daten/tinystories_train.bin | cut -f1)"
else
    python3 daten/vortokenisiere.py --hoechstens-token "$TOKEN_BUDGET"
fi
ls -lh daten/tinystories_train.bin daten/tinystories_meta.json

echo
echo "=== 4. Training ==="
mkdir -p "$CKPT"
if [ "$TESTLAUF" = "1" ]; then
    echo "(TESTLAUF: winzige Konfiguration, nur um die Kette zu beweisen)"
    python3 kern/trainiere.py \
        --daten daten/tinystories_train.bin \
        --meta daten/tinystories_meta.json \
        --checkpoint-ordner "$CKPT" \
        --schritte 20
else
    # nohup + Logdatei: die SSH-Verbindung darf abbrechen, ohne das
    # Training mitzureissen. Fortschritt steht danach in trainingslauf.log,
    # Checkpoints in $CKPT -- beides von aussen per scp abholbar.
    nohup python3 kern/trainiere.py \
        --daten daten/tinystories_train.bin \
        --meta daten/tinystories_meta.json \
        --checkpoint-ordner "$CKPT" \
        --gross \
        --token-budget "$TOKEN_BUDGET" \
        > "$HOME/trainingslauf.log" 2>&1 &
    echo "gestartet, PID $!"
    echo "Fortschritt:  tail -f $HOME/trainingslauf.log"
    echo "Checkpoints:  $CKPT"
    sleep 20
    echo "--- erste Ausgabe ---"
    tail -n 15 "$HOME/trainingslauf.log" || true
fi
