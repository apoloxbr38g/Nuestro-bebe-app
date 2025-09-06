# backend/ingest/ingest_apifootball.py
# -*- coding: utf-8 -*-
"""
Ingestor de datos desde apifootball (apiv3.apifootball.com) a CSV crudos
por liga interna (CN1, UA1, etc.) para luego unificarlos con merge_ingested.py.

Salida: backend/data/raw/<PAIS>/<CODE>_2024-2025.csv  (ej: backend/data/raw/CN/CN1_2024-2025.csv)
"""

import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

# -------- Config básica --------
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
API_BASE = "https://apiv3.apifootball.com/"
API_KEY = os.getenv("APIFOOTBALL_KEY", "").strip()

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Ligas a ingerir desde apifootball:
# - UA1 (Ucrania), TR1 (Turquía), JP1 (Japón), GR1 (Grecia), CN1 (China)
#   -> ¡Si quieres añadir más, sólo agrega aquí!
TARGET_LEAGUES: Dict[str, Dict[str, str]] = {
    "UA1": {"folder": "UA", "league_id": "325", "name": "Premier League"},          # Ukraine
    "TR1": {"folder": "TR", "league_id": "322", "name": "Süper Lig"},               # Turkey
    "JP1": {"folder": "JP", "league_id": "209", "name": "J1 League"},               # Japan
    "GR1": {"folder": "GR", "league_id": "178", "name": "Super League 1"},          # Greece
    "CN1": {"folder": "CN", "league_id": "118", "name": "Chinese Super League"},    # China PR
}

# Temporadas a cubrir (usa años calendario para el rango de fechas)
SEASONS: List[int] = [2024, 2025]

# Respeta el rate limit: apifootball suele permitir ~10 req/min
SLEEP_SECONDS_BETWEEN_CALLS = 1.2


def _api_get(params: Dict) -> List[Dict]:
    """Llama apifootball (accion get_events) y retorna la lista de partidos o []."""
    if not API_KEY:
        print("[ERROR] Falta APIFOOTBALL_KEY en backend/.env")
        return []

    params = dict(params)
    params["APIkey"] = API_KEY

    try:
        r = requests.get(API_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        # En apifootball, los errores vienen como dict con 'error' o similares
        if isinstance(data, dict) and "error" in data:
            print(f"[API ERR] {params} -> {data}")
            return []
        if isinstance(data, dict) and ("rateLimit" in data or "message" in data):
            print(f"[API WARN] {params} -> {data}")
            return []
        if not isinstance(data, list):
            # A veces responde algo raro; normalizamos a lista vacía
            print(f"[API WARN] Respuesta no-list {type(data)}: {data}")
            return []
        return data
    except Exception as e:
        print(f"[API EXC] {params} -> {e}")
        return []


def _year_range(y: int) -> (str, str):
    """Devuelve from/to YYYY-MM-DD para un año calendario."""
    return f"{y}-01-01", f"{y}-12-31"


def _norm_int(x):
    try:
        if x in (None, "", "null", "NULL"):
            return None
        return int(x)
    except Exception:
        return None


def normalize_event(evt: Dict, internal_code: str) -> Dict:
    """
    Convierte un partido de apifootball a nuestro esquema plano mínimo.
    Campos comunes que usaremos en la app:
      - League, Date, HomeTeam, AwayTeam, FTHG, FTAG (más opcionales)
    """
    # Campos típicos en apifootball get_events:
    # match_date: '2025-09-06'
    # match_time: '13:30'
    # match_hometeam_name, match_awayteam_name
    # match_hometeam_score, match_awayteam_score (strings)
    # Nota: No siempre incluyen corners/tarjetas; dejamos None si no hay.

    d = str(evt.get("match_date") or "").strip()
    t = str(evt.get("match_time") or "").strip()
    date_str = d  # guardamos sólo la fecha; el merger ya normaliza

    home = (evt.get("match_hometeam_name") or "").strip()
    away = (evt.get("match_awayteam_name") or "").strip()
    hg = _norm_int(evt.get("match_hometeam_score"))
    ag = _norm_int(evt.get("match_awayteam_score"))

    # opcionales (si aparecen en otros endpoints/planes; si no, None):
    hc = None  # corners home
    ac = None  # corners away
    hy = None  # yellows home
    ay = None  # yellows away
    hr = None  # reds home
    ar = None  # reds away

    return {
        "League": internal_code,   # <- clave: así la izquierda reconocerá la liga como CN1/UA1/etc.
        "Date": date_str,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": hg,
        "FTAG": ag,
        # extendidos opcionales:
        "HC": hc, "AC": ac,
        "HY": hy, "AY": ay,
        "HR": hr, "AR": ar,
    }


def ingest_league_year(internal_code: str, league_id: str, year: int) -> pd.DataFrame:
    """
    Descarga todos los partidos del año (from 1/1 to 12/31) para una liga y normaliza a DataFrame.
    """
    from_str, to_str = _year_range(year)
    params = {
        "action": "get_events",
        "league_id": league_id,
        "from": from_str,
        "to": to_str,
        # opcional: "timezone": "UTC"  # si lo quieres
    }
    print(f"  → Descargando {league_id} {year}…")
    rows = _api_get(params)
    time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

    if not rows:
        print(f"    · 0 partidos")
        return pd.DataFrame(columns=["League","Date","HomeTeam","AwayTeam","FTHG","FTAG","HC","AC","HY","AY","HR","AR"])

    recs = [normalize_event(evt, internal_code) for evt in rows]
    df = pd.DataFrame.from_records(recs)
    print(f"    · {len(df)} partidos")
    return df


def ingest_league(internal_code: str, league_id: str, seasons: List[int]) -> pd.DataFrame:
    """
    Descarga y concatena varios años para una liga concreta.
    """
    frames = []
    for y in seasons:
        frames.append(ingest_league_year(internal_code, league_id, y))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def main():
    if not API_KEY:
        print("[ERROR] No hay APIFOOTBALL_KEY en backend/.env — no puedo seguir.")
        return

    # Asegura carpetas país
    for code, meta in TARGET_LEAGUES.items():
        (RAW_DIR / meta["folder"]).mkdir(parents=True, exist_ok=True)

    # Descarga todas las ligas definidas
    for code, meta in TARGET_LEAGUES.items():
        folder = meta["folder"]
        league_id = meta["league_id"]
        print(f"[INGEST] {code} ({meta['name']}) league_id={league_id} seasons={SEASONS}")
        df = ingest_league(code, league_id, SEASONS)

        out = RAW_DIR / folder / f"{code}_{SEASONS[0]}-{SEASONS[-1]}.csv"
        if len(df):
            # Ordena por fecha si se puede
            try:
                _d = pd.to_datetime(df["Date"], errors="coerce")
                df = df.assign(_d=_d).sort_values("_d").drop(columns=["_d"])
            except Exception:
                pass
            df.to_csv(out, index=False)
            print(f"[OK] Guardado {out} con {len(df)} partidos")
        else:
            # Aun sin partidos, crea CSV con cabecera para evitar errores aguas abajo
            df = pd.DataFrame(columns=["League","Date","HomeTeam","AwayTeam","FTHG","FTAG","HC","AC","HY","AY","HR","AR"])
            df.to_csv(out, index=False)
            print(f"[OK] Guardado {out} con 0 partidos")


if __name__ == "__main__":
    main()
