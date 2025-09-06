#!/usr/bin/env bash
set -e

# Ir a la raíz del proyecto (carpeta actual)
cd "$(dirname "$0")"

# Activar entorno virtual
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo "⚠️ No existe venv, creándolo..."
  python3 -m venv venv
  source venv/bin/activate
fi

# Instalar dependencias
if [ -f "backend/requirements.txt" ]; then
  pip install -r backend/requirements.txt
else
  echo "⚠️ No se encontró requirements.txt, instalando paquetes básicos..."
  pip install fastapi uvicorn[standard] httpx pydantic pandas numpy scikit-learn joblib requests xgboost
fi

# Lanzar API en http://127.0.0.1:8000
exec uvicorn backend.app:app --host 0.0.0.0 --port "${PORT:-8000}" --reload

