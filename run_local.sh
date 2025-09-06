#!/usr/bin/env bash
set -e

# Ir a la carpeta del proyecto (donde está este script)
cd "$(dirname "$0")"

# Activar venv o crearlo si no existe
if [ ! -d "backend/.venv" ]; then
  python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate

# Instalar deps si hace falta
pip install -r backend/requirements.txt

# Ejecutar la app desde la raíz, apuntando al módulo completo
exec uvicorn backend.app:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
