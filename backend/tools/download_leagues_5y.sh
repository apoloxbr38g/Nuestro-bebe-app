#!/usr/bin/env bash
set -euo pipefail

# === Ligas a incluir (puedes añadir/quitar) ===
LEAGUES=(SP1 SP2 E0 E1 D1 D2 I1 I2 F1 F2 N1 P1 SC0 B1 T1 J1 G1 UKR)

# === Temporadas (últimos 5 ciclos) ===
SEASONS=(2526 2425 2324 2223 2122)

BASE_URL="https://www.football-data.co.uk/mmz4281"

mkdir -p data/raw

for L in "${LEAGUES[@]}"; do
  mkdir -p "data/raw/${L}"
  for S in "${SEASONS[@]}"; do
    OUT="data/raw/${L}/${S}.csv"
    URL="${BASE_URL}/${S}/${L}.csv"
    echo ">>> ${L} ${S}"
    # Descarga silenciosa; si falla, continuamos con la siguiente
    if ! wget -q -O "${OUT}.part" "${URL}"; then
      echo "    - No disponible (404/timeout): ${URL}"
      rm -f "${OUT}.part" || true
      continue
    fi
    # Si está vacío o muy pequeño, descartamos
    if [ ! -s "${OUT}.part" ] || [ "$(wc -c < "${OUT}.part")" -lt 200 ]; then
      echo "    - Vacío o incompleto, lo omito."
      rm -f "${OUT}.part"
      continue
    fi
    mv "${OUT}.part" "${OUT}"
    echo "    + Guardado: ${OUT}"
  done
done

echo
echo "Resumen de archivos existentes:"
find data/raw -type f -name "*.csv" | sort
