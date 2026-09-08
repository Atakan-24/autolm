#!/bin/bash
# Ruft Nemotron-3 (gratis, ueber Hermes/OpenRouter auf kiserver) fuer jede
# Instruktion in instruktionen.jsonl auf und sammelt die rohen Antworten
# als JSONL im Format, das bewertung/vergleich.py erwartet.
#
# Laeuft AUF kiserver (dort ist Hermes installiert). Von der Windows-Seite
# per: ssh kiserver 'bash ~/sammle_nemotron_antworten.sh' > antworten.jsonl
set -euo pipefail

SYS=$(cat ~/systemprompt.txt)
HERMES="/home/ubuntu/.hermes/hermes-agent/venv/bin/hermes"

while IFS= read -r zeile; do
    id=$(echo "$zeile" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    text=$(echo "$zeile" | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])")

    voll="$SYS

Beschreibung: $text"

    antwort=$(sudo -u hermes env HOME=/home/ubuntu HERMES_HOME=/home/ubuntu/.hermes \
        "$HERMES" -m nvidia/nemotron-3-ultra-550b-a55b:free --provider openrouter \
        --cli -z "$voll" -t "" 2>/dev/null || echo "FEHLER_BEIM_AUFRUF")

    python3 -c "
import json, sys
print(json.dumps({
    'modell': 'nemotron-3-ultra-550b (gratis)',
    'instruktion': sys.argv[1],
    'antwort_roh': sys.argv[2],
    'id': sys.argv[3],
}, ensure_ascii=False))
" "$text" "$antwort" "$id"

    >&2 echo "fertig: $id"
done < ~/instruktionen.jsonl
