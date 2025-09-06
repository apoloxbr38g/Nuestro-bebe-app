#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PORT="${PORT:-5173}"

echo "🚀 Sirviendo frontend/ en http://127.0.0.1:${PORT}"
python3 -m http.server "${PORT}" -d frontend >/dev/null 2>&1 &
PID=$!

# abrir en el navegador
sleep 1
xdg-open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true

# detener con Enter
echo "🛑 Presiona Enter para detener el servidor..."
read -r _
kill $PID
echo "✅ Frontend detenido"
