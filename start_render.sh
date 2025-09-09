#!/bin/bash
# Script de inicio para Render

# Exportar el path del proyecto
export PYTHONPATH=$PWD

# Arrancar uvicorn en el puerto que Render asigna
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
