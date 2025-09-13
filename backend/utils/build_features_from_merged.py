# backend/utils/build_features_from_merged.py
import os
import pandas as pd
from pathlib import Path

IN_PATH  = os.environ.get("DATA_PATH", "data/merged.csv")
OUT_PATH = "data/merged_features.csv"

def main():
    p = Path(IN_PATH)
    if not p.exists():
        raise FileNotFoundError(f"No encontré dataset en: {p.resolve()}")

    print(f"📂 Leyendo: {p.resolve()}")
    df = pd.read_csv(p, low_memory=False)

    # ---- Helpers de columnas (tolerantes a alias) ----
    cols = {c.lower(): c for c in df.columns}
    def pick(*names, default=None):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return default

    col_home = pick("HomeTeam","Home","HTeam")
    col_away = pick("AwayTeam","Away","ATeam")
    col_fthg = pick("FTHG","HG","HomeGoals","GoalsH")
    col_ftag = pick("FTAG","AG","AwayGoals","GoalsA")
    col_date = pick("Date","MatchDate","Fecha")

    need = [col_home, col_away, col_fthg, col_ftag]
    if any(x is None for x in need):
        raise ValueError(f"Faltan columnas mínimas HomeTeam/AwayTeam/FTHG/FTAG. Tengo: {list(df.columns)[:30]} ...")

    # ---- Tipos: fuerza goles a numéricos y parsea fecha si existe ----
    df[col_fthg] = pd.to_numeric(df[col_fthg], errors="coerce")
    df[col_ftag] = pd.to_numeric(df[col_ftag], errors="coerce")
    if col_date and col_date in df.columns:
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True)
        df = df.sort_values(by=[col_date]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # ---- BTTS indicador (para medias rodantes) ----
    btts = ((df[col_fthg] > 0) & (df[col_ftag] > 0)).astype(float)

    # ---- Helper rolling con transform (alineado) ----
    def rolling_mean_last5_by(group_col, series):
        # Serie ya alineada al índice original:
        s = series.copy()
        return (
            s.groupby(df[group_col])
             .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

    # H_*: rendimiento como local (GF=FTHG, GA=FTAG)
    df["H_GF5"]    = rolling_mean_last5_by(col_home, df[col_fthg])
    df["H_GA5"]    = rolling_mean_last5_by(col_home, df[col_ftag])
    df["H_BTTS5"]  = rolling_mean_last5_by(col_home, btts)

    # A_*: rendimiento como visitante (GF=FTAG, GA=FTHG)
    df["A_GF5"]    = rolling_mean_last5_by(col_away, df[col_ftag])
    df["A_GA5"]    = rolling_mean_last5_by(col_away, df[col_fthg])
    df["A_BTTS5"]  = rolling_mean_last5_by(col_away, btts)

    # Relleno prudente
    for c in ["H_GF5","H_GA5","H_BTTS5","A_GF5","A_GA5","A_BTTS5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    out = Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Features guardadas en: {out.resolve()}")
    print(f"   Filas: {len(df):,} | Columnas: {len(df.columns)}")
    print("   Nuevas cols: H_GF5, H_GA5, H_BTTS5, A_GF5, A_GA5, A_BTTS5")

if __name__ == "__main__":
    main()
