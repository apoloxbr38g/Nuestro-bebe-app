# backend/utils/data_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Columnas típicas de football-data.co.uk
# (pueden variar entre ligas/años, por eso usamos .get con fallback)
BASE_COLS = [
    "Div", "Date",
    "HomeTeam", "AwayTeam",
    "FTHG", "FTAG",  # goles final (full-time)
]

def _read_single_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path)  # fallback

    # Asegurar columnas base
    cols_present = [c for c in BASE_COLS if c in df.columns]
    if len(cols_present) < 5:
        # muchos archivos antiguos cambian nombres; intentamos alias comunes
        alias = {
            "HomeTeam": ["Home", "HomeTeamName"],
            "AwayTeam": ["Away", "AwayTeamName"],
            "FTHG": ["FTH", "HG"],
            "FTAG": ["FTA", "AG"],
        }
        for target, candidates in alias.items():
            if target not in df.columns:
                for c in candidates:
                    if c in df.columns:
                        df[target] = df[c]
                        break

    # Selección y limpieza básica
    keep = [c for c in ["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG"] if c in df.columns]
    df = df[keep].copy()

    # Parseo de fecha (football-data suele ser DD/MM/YY o DD/MM/YYYY)
    # dayfirst=True para interpretar 03/09/25 como 3 septiembre 2025
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Tipos numéricos en goles
    for g in ("FTHG","FTAG"):
        if g in df.columns:
            df[g] = pd.to_numeric(df[g], errors="coerce")

    # Quitar filas sin fecha/equipos/goles
    for c in ("Date","HomeTeam","AwayTeam","FTHG","FTAG"):
        if c not in df.columns:
            # si faltan columnas críticas, descartamos este CSV
            return None
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])

    # Normalizar nombres (opcional, simple)
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

    # Orden cronológico
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def load_league_history(league_code: str, seasons: Iterable[str]) -> pd.DataFrame:
    """Concatena DataFrames de múltiples temporadas para una liga."""
    frames: List[pd.DataFrame] = []
    league_dir = DATA_RAW / league_code
    for s in seasons:
        csv_path = league_dir / f"{s}.csv"
        df = _read_single_csv(csv_path)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No encontré datos válidos para {league_code} en temporadas {list(seasons)}"
        )
    out = pd.concat(frames, ignore_index=True)
    # sanity check final
    out = out.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out
