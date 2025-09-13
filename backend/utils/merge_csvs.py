# backend/utils/merge_csvs.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

def find_data_dirs():
    here = Path(__file__).resolve()
    # Candidatos: <repo>/data/raw  y  <repo>/backend/data/raw
    candidates = [
        here.parents[2] / "data" / "raw",      # <repo>/data/raw
        here.parents[1] / "data" / "raw",      # <repo>/backend/data/raw
    ]
    existing = [p for p in candidates if p.is_dir()]
    if not existing:
        return None, None
    raw_dir = existing[0]
    data_dir = raw_dir.parent
    return raw_dir, data_dir

def read_csv_loose(p: Path) -> pd.DataFrame:
    # Lector tolerante: autodetecta separador, ignora errores de encoding
    try:
        df = pd.read_csv(p, sep=None, engine="python", low_memory=False, encoding_errors="ignore")
    except Exception:
        # Fallback a coma
        df = pd.read_csv(p, low_memory=False, encoding_errors="ignore")
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Homogeneiza nombres conocidos a un set estándar
    rename_map = {}
    for c in list(df.columns):
        lc = str(c).strip()
        lc_norm = lc.replace(" ", "").replace("-", "").replace(".", "").lower()

        # Mapeos comunes
        if lc_norm in {"hometeam","home","teamhome"}: rename_map[c] = "HomeTeam"
        elif lc_norm in {"awayteam","away","teamaway"}: rename_map[c] = "AwayTeam"
        elif lc_norm in {"fthg","fulltimehomegoals","homegoals"}: rename_map[c] = "FTHG"
        elif lc_norm in {"ftag","fulltimeawaygoals","awaygoals"}: rename_map[c] = "FTAG"
        elif lc_norm in {"hc","homecorners"}: rename_map[c] = "HC"
        elif lc_norm in {"ac","awaycorners"}: rename_map[c] = "AC"
        elif lc_norm in {"hy","homeyellow","homeyellows","homeyellowcards"}: rename_map[c] = "HY"
        elif lc_norm in {"ay","awayyellow","awayyellows","awayyellowcards"}: rename_map[c] = "AY"
        elif lc_norm in {"hr","homered","homeredcards"}: rename_map[c] = "HR"
        elif lc_norm in {"ar","awayred","awayredcards"}: rename_map[c] = "AR"
        elif lc_norm in {"date"}: rename_map[c] = "Date"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Asegura presencia de columnas clave (si faltan, créalas vacías)
    for col in ["HomeTeam","AwayTeam","FTHG","FTAG","HC","AC","HY","AY","HR","AR","Date"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Tipos básicos
    for col in ["FTHG","FTAG","HC","AC","HY","AY","HR","AR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def main():
    raw_dir, data_dir = find_data_dirs()
    if raw_dir is None or data_dir is None:
        print("No encontré ninguna carpeta data/raw en el proyecto.")
        print("Asegúrate de tener CSVs en ./data/raw o ./backend/data/raw")
        sys.exit(1)

    csvs = sorted(raw_dir.glob("**/*.csv"))
    if not csvs:
        print(f"No encontré CSVs en {raw_dir}. Corre primero backend/tools/download_leagues_5y.sh")
        sys.exit(1)

    print(f"📂 Raw dir: {raw_dir}")
    print(f"🧾 CSVs encontrados: {len(csvs)}")

    frames = []
    for p in csvs:
        try:
            df = read_csv_loose(p)
            df = normalize_cols(df)

            # Liga desde la carpeta superior inmediata (…/raw/<LEAGUE>/xxxx.csv)
            try:
                league = p.parent.name
            except Exception:
                league = "UNK"
            df["League"] = league

            # Fecha como string (no romper si trae formatos distintos)
            if "Date" in df.columns:
                df["Date"] = df["Date"].astype(str)

            frames.append(df)
        except Exception as e:
            print(f"⚠️  Error leyendo {p}: {e}")

    if not frames:
        print("❌ No pude leer ningún CSV válido.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    # Columnas de interés primero (si existen)
    preferred = ["League","Date","HomeTeam","AwayTeam","FTHG","FTAG","HC","AC","HY","AY","HR","AR"]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols]

    out_path = data_dir / "merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"✅ Guardado: {out_path}")
    print(f"   Filas: {len(merged):,} | Columnas: {len(merged.columns)}")

if __name__ == "__main__":
    main()
