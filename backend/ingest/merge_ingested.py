# backend/ingest/merge_ingested.py
import pandas as pd
from pathlib import Path
from backend.data_loader import MERGED

# Archivos a sumar (ajusta según lo que ingestas)
CANDIDATES = [
    Path("backend/data/raw/UA/UA1_2022-2025.csv"),
    # Path("backend/data/raw/TR/TR1_2022-2025.csv"),
    # Path("backend/data/raw/JP/JP1_2022-2025.csv"),
    # Path("backend/data/raw/GR/GR1_2022-2025.csv"),
    # Path("backend/data/raw/CN/CN1_2022-2025.csv"),
]

def load_safe(p: Path):
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    # normaliza columnas mínimas
    for c in ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"]:
        if c not in df.columns: df[c] = None
    return df[["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"] + [c for c in df.columns if c not in ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"]]]

if __name__ == "__main__":
    base = pd.read_csv(MERGED) if MERGED.exists() else pd.DataFrame(columns=["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"])
    adds = [load_safe(p) for p in CANDIDATES]
    extra = pd.concat(adds, ignore_index=True)
    merged = pd.concat([base, extra], ignore_index=True)

    # saneo: quita duplicados exactos de partido (fecha+equipos+goles)
    merged = merged.drop_duplicates(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"], keep="last")

    # orden por fecha
    try:
        merged["_d"] = pd.to_datetime(merged["Date"], errors="coerce", dayfirst=True)
        merged = merged.sort_values("_d").drop(columns=["_d"])
    except: pass

    MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED, index=False)
    print(f"MERGED actualizado → {MERGED} · filas: {len(merged)}")
