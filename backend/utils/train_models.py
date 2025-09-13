# backend/utils/train_models.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = os.environ.get("DATA_PATH", "data/merged_features.csv")
OUT_DIR   = Path("backend/models")

def pickcol(cols, *cands):
    icol = {c.lower(): c for c in cols}
    for c in cands:
        if c and c.lower() in icol:
            return icol[c.lower()]
    return None

def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No encontré dataset en: {p}")
    print(f"📂 Cargando dataset: {p.resolve()}")
    # low_memory=False para evitar DtypeWarning molestos
    df = pd.read_csv(p, low_memory=False)
    # Normaliza tipos comunes
    # fechas (si hay)
    col_date = pickcol(df.columns, "Date", "MatchDate", "Fecha")
    if col_date:
        # dayfirst=True porque tu merge suele venir dd/mm/yyyy
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True)
        df = df.sort_values(col_date).reset_index(drop=True)
    # goles a numéricos
    col_fthg = pickcol(df.columns, "FTHG","HG","HomeGoals","GoalsH")
    col_ftag = pickcol(df.columns, "FTAG","AG","AwayGoals","GoalsA")
    for c in (col_fthg, col_ftag):
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_features(df: pd.DataFrame):
    # Preferimos tus 6 features nuevas; si no, intenta con Elo si existiera
    cand = ["H_GF5","H_GA5","H_BTTS5","A_GF5","A_GA5","A_BTTS5"]
    feats = [c for c in cand if c in df.columns]
    if not feats:
        # fallback (por si agregas Elo luego)
        for c in ["Elo_H","Elo_A"]:
            if c in df.columns:
                feats.append(c)
    if len(feats) < 2:
        raise ValueError("No encontré columnas de features suficientes. Esperaba al menos H_*/A_* (las nuevas) o Elo_H/Elo_A.")
    print(f"🧠 Usando {len(feats)} columnas de features: {feats}")
    return feats

def build_targets(df: pd.DataFrame):
    col_fthg = pickcol(df.columns, "FTHG","HG","HomeGoals","GoalsH")
    col_ftag = pickcol(df.columns, "FTAG","AG","AwayGoals","GoalsA")
    if not col_fthg or not col_ftag:
        raise ValueError("Faltan columnas de goles FTHG/FTAG para crear objetivos.")

    H = df[col_fthg]
    A = df[col_ftag]

    # 1X2: away=0, draw=1, home=2
    y_1x2 = pd.Series(np.where(H > A, 2, np.where(H == A, 1, 0)), index=df.index, name="y_1x2")

    # OU2.5: 1 si total > 2.5
    y_ou  = pd.Series(((H + A) > 2.5).astype(int), index=df.index, name="y_ou")

    # BTTS: 1 si ambos > 0
    y_btts = pd.Series(((H > 0) & (A > 0)).astype(int), index=df.index, name="y_btts")

    return y_1x2, y_ou, y_btts

def split_xy(df, feats, y_series):
    # máscara de filas válidas (features y target no nulos/NaN)
    mask = df[feats].apply(pd.to_numeric, errors="coerce").notna().all(axis=1) & y_series.notna()
    work = df.loc[mask, feats].astype(float)
    y    = y_series.loc[mask].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(work, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    return work, y, Xtr, Xte, ytr, yte

def clf_xgb_multi():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        predictor="cpu_predictor",
        n_jobs=4,
        random_state=42,
    )

def clf_xgb_bin():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        predictor="cpu_predictor",
        n_jobs=4,
        random_state=42,
    )

def train_1x2(df, feats, y_1x2):
    work, y, Xtr, Xte, ytr, yte = split_xy(df, feats, y_1x2)
    model = clf_xgb_multi()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    yhat  = proba.argmax(axis=1)
    acc   = accuracy_score(yte, yhat)
    ll    = log_loss(yte, proba, labels=[0,1,2])
    print(f"✅ 1X2 -> acc={acc:.4f} | logloss={ll:.4f} | n={len(y)}")
    return {"model": model, "features": feats}

def train_ou(df, feats, y_ou):
    work, y, Xtr, Xte, ytr, yte = split_xy(df, feats, y_ou)
    model = clf_xgb_bin()
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:,1]
    try:
        auc = roc_auc_score(yte, p)
        print(f"✅ OU2.5 -> auc={auc:.4f} | n={len(y)}")
    except Exception:
        ll = log_loss(yte, np.vstack([1-p, p]).T)
        print(f"✅ OU2.5 -> logloss={ll:.4f} | n={len(y)}")
    return {"model": model, "features": feats}

def train_btts(df, feats, y_btts):
    work, y, Xtr, Xte, ytr, yte = split_xy(df, feats, y_btts)
    model = clf_xgb_bin()
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:,1]
    try:
        auc = roc_auc_score(yte, p)
        print(f"✅ BTTS -> auc={auc:.4f} | n={len(y)}")
    except Exception:
        ll = log_loss(yte, np.vstack([1-p, p]).T)
        print(f"✅ BTTS -> logloss={ll:.4f} | n={len(y)}")
    return {"model": model, "features": feats}

def main():
    df = load_df(DATA_PATH)
    feats = get_features(df)
    y1, you, yb = build_targets(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    b1x2  = train_1x2(df, feats, y1)
    bou   = train_ou(df, feats, you)
    bbtts = train_btts(df, feats, yb)

    joblib.dump(b1x2,  OUT_DIR / "model_1x2.pkl")
    joblib.dump(bou,    OUT_DIR / "model_ou25.pkl")
    joblib.dump(bbtts,  OUT_DIR / "model_btts.pkl")
    print(f"💾 Modelos guardados en: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
