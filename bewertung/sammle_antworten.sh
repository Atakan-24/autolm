#!/bin/bash
# Ruft ein Modell (gratis, ueber Hermes/OpenRouter auf kiserver) fuer jede
# Instruktion in instruktionen.jsonl auf und sammelt die rohen Antworten
# als JSONL im Format, das bewertung/vergleich.py erwartet.
#
# Verallgemeinerung von sammle_nemotron_antworten.sh: das Modell kommt als
# Argument, damit derselbe Lauf -- IDENTISCHER Systemprompt, IDENTISCHE
# Instruktionen -- gegen mehrere Modelle wiederholt werden kann. Das ist die
# Fairness-Bedingung aus dem Plan (Stufe 3): ein Vergleich zaehlt nur, wenn
# alle Modelle exakt dieselbe Aufgabe bekommen.
#
# Laeuft AUF kiserver (dort ist Hermes installiert):
#   bash ~/sammle_antworten.sh "<openrouter-modell>" "<label>" > antworten.jsonl
set -euo pipefail

MODELL="${1:?Modell-ID fehlt, z.B. google/gemma-4-31b-it:free}"
LABEL="${2:-$MODELL}"

SYS=$(cat ~/systemprompt.txt)
HERMES="/home/ubuntu/.hermes/hermes-agent/venv/bin/hermes"

while IFS= read -r zeile; do
    id=$(echo "$zeile" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    text=$(echo "$zeile" | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])")

    voll="$SYS

Beschreibung: $text"

    # timeout: ein haengender Aufruf darf nicht den ganzen Lauf blockieren.
    # Bei Fehler steht ein Marker in der Antwort -- vergleich.py wertet das
    # dann sauber als "Tor 1 nicht bestanden", nicht als fehlende Zeile.
    antwort=$(timeout 180 sudo -u hermes env HOME=/home/ubuntu HERMES_HOME=/home/ubuntu/.hermes \
        "$HERMES" -m "$MODELL" --provider openrouter \
        --cli -z "$voll" -t "" 2>/dev/null || echo "FEHLER_BEIM_AUFRUF")

    python3 -c "
import json, sys
print(json.dumps({
    'modell': sys.argv[4],
    'instruktion': sys.argv[1],
    'antwort_roh': sys.argv[2],
    'id': sys.argv[3],
}, ensure_ascii=False))
" "$text" "$antwort" "$id" "$LABEL"

    >&2 echo "fertig: $id"
done < ~/instruktionen.jsonl
