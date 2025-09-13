# backend/app.py
from __future__ import annotations

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import time
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import httpx
from datetime import date, timedelta, datetime

# --- Parche para modelos XGBoost serializados antiguos ---
def _patch_legacy_xgb_bundle(bundle):
    """
    Evita errores tipo:
      AttributeError: 'XGBClassifier' object has no attribute 'use_label_encoder'
    y diferencias de predictor entre versiones.
    """
    if bundle is None:
        return None
    # bundle puede ser un dict {"model": xgb, "features":[...]} o el modelo directo
    mdl = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    try:
        if not hasattr(mdl, "use_label_encoder"):
            setattr(mdl, "use_label_encoder", False)
        if not hasattr(mdl, "predictor"):
            setattr(mdl, "predictor", None)
    except Exception as e:
        print("[WARN] No pude parchear modelo XGB:", repr(e))
    return bundle

# ==== Carga de .env (claves API) ====
load_dotenv()  # APISPORTS_KEY, APIFOOTBALL_API_KEY en backend/.env

APISPORTS_KEY    = os.getenv("APISPORTS_KEY")
APIFOOTBALL_KEY  = os.getenv("APIFOOTBALL_API_KEY")

APISPORTS_BASE   = os.getenv("APISPORTS_BASE", "https://v3.football.api-sports.io")
APIFOOTBALL_BASE = os.getenv("APIFOOTBALL_BASE", "https://apiv3.apifootball.com/")

# ==== Importar proveedor ApiSports (nuestro cliente HTTPX) ====
from backend.providers import apisports  # asegúrate que exista backend/providers/apisports.py

# ==== Modelos y utilidades del proyecto ====
from backend.models.baseline import PoissonModel
from backend.data_loader import refresh_dataset, MERGED
from backend.train import MODEL_PATH
from backend.train_ou import MODEL_OU_PATH
from backend.train_btts import MODEL_BTTS_PATH
from backend.models.features import build_training_table
from backend.utils.data_loader import load_league_history

# ---------- App & CORS ----------
app = FastAPI(
    title="Nuestro Bebé App",
    version="0.8.1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta si quieres restringir
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración de ligas soportadas y temporadas ---
DEFAULT_SEASONS: List[str] = ["2526", "2425", "2324", "2223", "2122"]

SUPPORTED_LEAGUES = {
    "SP1": "LaLiga (Primera)",
    "SP2": "LaLiga (Segunda)",
    "E0":  "Premier League",
    "E1":  "Championship",
    "D1":  "Bundesliga",
    "D2":  "2. Bundesliga",
    "I1":  "Serie A",
    "I2":  "Serie B",
    "F1":  "Ligue 1",
    "F2":  "Ligue 2",
    "N1":  "Eredivisie",
    "P1":  "Primeira Liga",
    "SC0": "Scottish Premiership",
    "B1":  "Belgian First Division A",
    # 💜 NUEVAS
    "T1":  "Superliga Turquía",
    "J1":  "J1 League (Japón)",
    "G1":  "Superliga Grecia",
    "UKR": "Liga Premier Ucrania",
}

# ---------- Cache global ----------
df_global: Optional[pd.DataFrame] = None
Xy_global: Optional[pd.DataFrame] = None
current_csv_path: Optional[Path] = None

# ---------- Estado modelos ----------
model = PoissonModel()
clf_bundle = None       # XGB 1X2
clf_ou_bundle = None    # XGB Over/Under 2.5
clf_btts_bundle = None  # XGB BTTS

# ---------- Paths datos ----------
LOCAL_SAMPLE = Path(__file__).parent / "data" / "sample_matches.csv"
DATA_PATH = MERGED if MERGED.exists() else LOCAL_SAMPLE

# ---------- Frontend ----------
FRONT_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONT_DIR), html=True), name="frontend")
app.mount("/assets", StaticFiles(directory=str(FRONT_DIR / "assets")), name="assets")
app.mount("/app/assets", StaticFiles(directory=str(FRONT_DIR / "assets")), name="assets_app")  # alias útil

# ---------- Utilidades internas ----------
def _set_data(df: pd.DataFrame, csv_path: Path | None = None, build_features: bool = True):
    """Carga dataset en memoria, re-entrena Poisson y opcionalmente construye features."""
    global df_global, Xy_global, current_csv_path, model
    df_global = df.copy()
    current_csv_path = csv_path
    model.fit(df_global)
    if build_features:
        try:
            Xy_global = build_training_table(df_global, window=10)
        except Exception as e:
            Xy_global = None
            print("[WARN] build_training_table falló:", e)

def _ensure_data_loaded():
    global df_global
    if df_global is None:
        df = pd.read_csv(DATA_PATH)
        _set_data(df, DATA_PATH, build_features=False)

def _ensure_features():
    """Construye Xy_global si aún no existe (perezoso)."""
    global Xy_global, df_global
    if Xy_global is None and df_global is not None:
        Xy_global = build_training_table(df_global, window=10)

def _get_Xy():
    _ensure_data_loaded()
    _ensure_features()
    return Xy_global if Xy_global is not None else build_training_table(df_global, window=10)

# ---------- Etiquetas de features ----------
FEATURE_LABELS = {
    "Elo_H": "Elo Local",
    "Elo_A": "Elo Visitante",
    "H_r_GF": "F recientes (Local)",
    "H_r_GA": "GA recientes (Local)",
    "H_r_GD": "GD recientes (Local)",
    "H_r_W":  "Racha victorias (Local)",
    "H_r_D":  "Racha empates (Local)",
    "H_r_L":  "Racha derrotas (Local)",
    "H_r_HomeRate":"Fortaleza en casa",
    "A_r_GF": "GF recientes (Visita)",
    "A_r_GA": "GA recientes (Visita)",
    "A_r_GD": "GD recientes (Visita)",
    "A_r_W":  "Racha victorias (Visita)",
    "A_r_D":  "Racha empates (Visita)",
    "A_r_L":  "Racha derrotas (Visita)",
    "A_r_HomeRate":"Rend. visitante",
}

def top_drivers(row: pd.DataFrame, feature_list: list[str], k: int = 3):
    colz = [c for c in feature_list if c in row.columns]
    if not colz: return []
    vals = row[colz].iloc[0]
    med = np.median(vals.values)
    mad = np.median(np.abs(vals.values - med)) or 1.0
    score = (vals - med) / mad
    idx = np.argsort(-np.abs(score.values))[:k]
    out = []
    for i in idx:
        feat = colz[i]; s = score.values[i]
        pretty = FEATURE_LABELS.get(feat, feat)
        arrow = "↑" if s > 0 else "↓"
        out.append(f"{pretty} {arrow}")
    return out

# ---------- Helpers simples ----------
def _safe_ts(dt_str: str) -> int:
    """Convierte 'YYYY-MM-DD HH:MM' a timestamp para ordenar; 0 si falla."""
    try:
        return int(datetime.fromisoformat(dt_str.replace(" ", "T")).timestamp())
    except Exception:
        return 0

def _to_int(s):
    """Convierte '2' → 2; '', None → None; tolera strings raros."""
    try:
        return int(s) if s is not None and str(s).strip() != "" else None
    except Exception:
        return None

# ---------- Response Model ----------
class PredictResponse(BaseModel):
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    exp_goals_home: float
    exp_goals_away: float
    exp_goals_total: float
    ou_over25: float
    ou_under25: float
    btts_yes: float
    btts_no: float
    top_scorelines: list
    src_1x2: str
    src_ou25: str
    src_btts: str
    # extras
    exp_corners_home: float | None = None
    exp_corners_away: float | None = None
    exp_corners_total: float | None = None
    corners_over95: float | None = None
    corners_under95: float | None = None
    exp_yellows_home: float | None = None
    exp_yellows_away: float | None = None
    exp_yellows_total: float | None = None
    yellows_over45: float | None = None
    yellows_under45: float | None = None
    exp_reds_home: float | None = None
    exp_reds_away: float | None = None
    exp_reds_total: float | None = None
    p_goals_0: float | None = None
    p_goals_1: float | None = None
    p_goals_2: float | None = None
    p_goals_3plus: float | None = None

# === Selecciones: payload de predicción ===
class NationalPredictIn(BaseModel):
    home_id: int
    away_id: int
    neutral: bool = False
    lookback: int = 10  # últimos N partidos para estimar tasas

# ---------- Startup ----------
@app.on_event("startup")
def _load_and_train():
    global clf_bundle, clf_ou_bundle, clf_btts_bundle
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        df = pd.read_csv(LOCAL_SAMPLE)
    _set_data(df, DATA_PATH, build_features=False)

    # Carga modelos ML si existen
    clf_bundle      = joblib.load(MODEL_PATH)      if MODEL_PATH.exists()      else None
    clf_ou_bundle   = joblib.load(MODEL_OU_PATH)   if MODEL_OU_PATH.exists()   else None
    clf_btts_bundle = joblib.load(MODEL_BTTS_PATH) if MODEL_BTTS_PATH.exists() else None

    # 🔧 Parchea modelos XGB legacy para evitar AttributeError
    clf_bundle      = _patch_legacy_xgb_bundle(clf_bundle)
    clf_ou_bundle   = _patch_legacy_xgb_bundle(clf_ou_bundle)
    clf_btts_bundle = _patch_legacy_xgb_bundle(clf_btts_bundle)

    # Precarga features (warmup)
    _ensure_data_loaded()
    _ensure_features()
    print("💜 Warmup completado: features precargadas.")

# ---------- Entrenamientos ----------
@app.post("/train")
def train_endpoint():
    from backend.train import train as _train
    info = _train(str(DATA_PATH))
    global clf_bundle
    clf_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    return {"status": "ok", **info}

@app.post("/train_ou")
def train_ou_endpoint():
    from backend.train_ou import train_ou as _train_ou
    info = _train_ou(str(DATA_PATH))
    global clf_ou_bundle
    clf_ou_bundle = joblib.load(MODEL_OU_PATH) if MODEL_OU_PATH.exists() else None
    return {"status": "ok", **info}

@app.post("/train_btts")
def train_btts_endpoint():
    from backend.train_btts import train_btts as _train_btts
    info = _train_btts(str(DATA_PATH))
    global clf_btts_bundle
    clf_btts_bundle = joblib.load(MODEL_BTTS_PATH) if MODEL_BTTS_PATH.exists() else None
    return {"status": "ok", **info}

# ---------- Recargas CSV ----------
@app.post("/reload")
def reload_data(build_features: bool = False):
    path = refresh_dataset(leagues=("SP1",), start_years=(2023, 2024))
    df = pd.read_csv(path)
    _set_data(df, Path(path), build_features=build_features)
    return {"status": "ok", "rows": int(len(df)), "file": str(path), "built": build_features}

@app.post("/reload_multi")
def reload_multi(
    leagues: List[str] = Body(..., embed=True),
    start_years: Optional[List[int]] = Body(None, embed=True),
    last_n: Optional[int] = Body(4, embed=True),
    build_features: bool = Body(False, embed=True),
):
    kwargs = {}
    if start_years:
        kwargs["start_years"] = tuple(start_years)
        kwargs["last_n_years"] = None
    else:
        kwargs["start_years"] = tuple()
        kwargs["last_n_years"] = int(last_n or 4)

    path = refresh_dataset(leagues=tuple(leagues), **kwargs)
    df = pd.read_csv(path)
    _set_data(df, Path(path), build_features=build_features)
    return {
        "status": "ok",
        "rows": int(len(df)),
        "file": str(path),
        "leagues": leagues,
        "years": (start_years if start_years else f"last_{last_n}"),
        "built": build_features
    }

# ---------- Proveedor: ApiSports (clubes) ----------
@app.get("/providers/ping")
async def providers_ping():
    """Comprueba si las llaves de los proveedores están cargadas."""
    return {
        "apisports_key_loaded": bool(APISPORTS_KEY),
        "apifootball_key_loaded": bool(APIFOOTBALL_KEY),
    }

@app.get("/leagues/by-country")
async def leagues_by_country(
    country: str = Query(..., description="Ej: Turkey, Ukraine, Japan, Greece, China"),
):
    leagues = await apisports.leagues_by_country(country)
    return {"country": country, "leagues": leagues}

@app.get("/fixtures/by-league")
async def fixtures_by_league(
    league_id: str = Query(..., description="ID de liga ApiSports (p.ej. 39)"),
    season: int | None = Query(None),
    date_: str | None = Query(None, alias="date"),
    next_: int | None = Query(None, alias="next"),
    days: int | None = Query(30),
):
    """Partidos por liga usando date/season o atajo next. No combinar 'next' con 'days/date'."""
    if next_ is not None and (date_ or days not in (None, 30) or season is not None):
        raise HTTPException(status_code=400, detail="Usa 'next' solo, o usa 'date/season/days' sin 'next'.")
    fixtures = await apisports.fixtures_by_league(league_id, season=season, date=date_, next=next_, days=days or 30)
    return {"count": len(fixtures), "fixtures": fixtures}

@app.get("/fixtures/global_next5")
async def fixtures_global_next5(tz: str = "America/Santiago", days: int = 14):
    """
    Próximos 5 partidos globales desde hoy hasta hoy+days.
    Incluye selecciones, amistosos y Sub-20 (no filtramos por liga).
    """
    if not APIFOOTBALL_KEY:
        raise HTTPException(status_code=500, detail="Falta APIFOOTBALL_API_KEY")

    start = date.today()
    end   = start + timedelta(days=max(1, min(days, 30)))  # seguridad

    params = {
        "APIkey": APIFOOTBALL_KEY,
        "action": "get_events",
        "from": start.isoformat(),
        "to":   end.isoformat(),
        "timezone": tz,
    }

    async with httpx.AsyncClient(timeout=40.0) as cx:
        r = await cx.get(APIFOOTBALL_BASE, params=params)
        r.raise_for_status()
        data = r.json()

    if not isinstance(data, list):
        return {"count": 0, "fixtures": []}

    items = []
    for it in data:
        items.append({
            "datetime": f"{it.get('match_date','')} {it.get('match_time','')}".strip(),
            "league":   it.get("league_name") or "",
            "home":     it.get("match_hometeam_name") or "",
            "away":     it.get("match_awayteam_name") or "",
            "status":   it.get("match_status") or "",
        })

    items.sort(key=lambda x: x["datetime"])
    return {"count": min(5, len(items)), "fixtures": items[:5]}

# --- Top Scorers (ApiSports) ---
CODE_TO_ID = {
    "E0": 39,    # Premier League
    "SP1": 140,  # LaLiga
    "I1": 135,   # Serie A
    "D1": 78,    # Bundesliga
    "F1": 61,    # Ligue 1
    # agrega los que uses...
}

@app.get("/topscorers")
async def topscorers(
    league: Optional[str] = Query(None, description="Código tipo E0, SP1…"),
    league_id: Optional[int] = Query(None, description="ID ApiSports (p.ej. 39 Premier)"),
    season: Optional[int] = Query(None, description="Año de inicio de la temporada, ej 2024"),
):
    if not APISPORTS_KEY:
        raise HTTPException(status_code=500, detail="Falta APISPORTS_KEY")

    # Resolver ID de liga si solo mandaron código
    if league_id is None:
        if not league:
            raise HTTPException(status_code=422, detail="Manda league_id numérico o league (código).")
        code = league.upper()
        if code not in CODE_TO_ID:
            raise HTTPException(status_code=400, detail=f"Código de liga no soportado: {code}")
        league_id = CODE_TO_ID[code]

    if season is None:
        season = date.today().year

    url = f"{APISPORTS_BASE}/players/topscorers"
    headers = {"x-apisports-key": APISPORTS_KEY}

    async def _fetch(season_year: int):
        params = {"league": league_id, "season": season_year}
        async with httpx.AsyncClient(timeout=40.0) as cx:
            r = await cx.get(url, headers=headers, params=params)
            r.raise_for_status()
            return r.json()

    # Primer intento con la temporada pedida
    payload = await _fetch(season)
    resp = payload.get("response") or []

    # Fallback: probar con la temporada anterior si está vacío
    if not resp and season > 2000:
        payload = await _fetch(season - 1)
        resp = payload.get("response") or []

    # Normaliza para el frontend
    items = []
    for it in resp:
        player = it.get("player") or {}
        stats  = (it.get("statistics") or [{}])[0]
        team   = (stats.get("team") or {})
        goals  = (stats.get("goals") or {}).get("total")
        items.append({
            "player_name": player.get("name") or f"{player.get('firstname','')} {player.get('lastname','')}".strip(),
            "photo": player.get("photo"),
            "team_name": team.get("name"),
            "goals": goals or 0,
            "player": player,
            "statistics": it.get("statistics"),
            "team": team,
        })

    return {"count": len(items), "players": items}

# ---------- Últimos resultados (APIFOOTBALL) ----------
@app.get("/recent_live")
async def recent_live(days: int = 8, limit: int = 10):
    """
    Últimos resultados FINALIZADOS (días hacia atrás).
    Devuelve: Date, League, HomeTeam, AwayTeam, FTHG, FTAG
    """
    if not APIFOOTBALL_KEY:
        raise HTTPException(status_code=500, detail="Falta APIFOOTBALL_API_KEY")

    start = date.today() - timedelta(days=max(1, min(days, 30)))
    end   = date.today()

    params = {
        "APIkey": APIFOOTBALL_KEY,
        "action": "get_events",
        "from": start.isoformat(),
        "to":   end.isoformat(),
        "timezone": "UTC",
    }

    try:
        async with httpx.AsyncClient(timeout=40.0) as cx:
            r = await cx.get(APIFOOTBALL_BASE, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proveedor falló: {e}")

    if not isinstance(data, list):
        return {"matches": []}

    FINISHED_KEYS = {
        "finished", "ft", "ended", "match finished",
        "after extra time", "aet", "after penalties", "ap"
    }
    EXCLUDE_KEYS = {"postponed", "canceled", "cancelled", "abandoned", "walkover"}

    items = []
    for it in data:
        dt = f"{it.get('match_date','')} {it.get('match_time','')}".strip()

        # goles FT (compatibilidad)
        home_ft = it.get("match_hometeam_ft_score")
        away_ft = it.get("match_awayteam_ft_score")
        if home_ft is None and away_ft is None:
            home_ft = it.get("match_hometeam_score")
            away_ft = it.get("match_awayteam_score")

        # estado normalizado
        raw_status = (it.get("match_status") or "").strip()
        status_norm = raw_status.lower()

        # criterio de finalizado
        is_finished = (
            status_norm in FINISHED_KEYS or
            (_to_int(home_ft) is not None and _to_int(away_ft) is not None and status_norm not in EXCLUDE_KEYS)
        )
        if not is_finished:
            continue

        items.append({
            "Date":     dt.split(" ")[0] if dt else "",
            "League":   it.get("league_name") or "",
            "HomeTeam": it.get("match_hometeam_name") or "",
            "AwayTeam": it.get("match_awayteam_name") or "",
            "FTHG":     _to_int(home_ft),
            "FTAG":     _to_int(away_ft),
            "ts":       _safe_ts(dt),
        })

    # Orden y límite
    items.sort(key=lambda x: x.get("ts", 0), reverse=True)
    items = items[:max(1, min(limit, 50))]
    for it in items:
        it.pop("ts", None)

    return {"matches": items}

# ---------- Proveedor teams por liga (ApiSports) ----------
@app.get("/providers/teams")
async def providers_teams(
    league_id: str = Query(..., description="ID de liga en ApiSports"),
    season: int | None = Query(None, description="Temporada opcional"),
):
    teams = await apisports.teams_by_league(league_id, season=season)
    return {"count": len(teams), "teams": teams}

# ---------- Selecciones (nacionales) ----------
@app.get("/teams/national")
async def teams_national(q: Optional[str] = Query(None, description="Filtro: 'Chile', 'Argentina', 'Bra'…")):
    """
    Devuelve selecciones nacionales.
    - Si mandas ?q=, usa búsqueda inteligente (search y country).
    - Si no mandas q, intenta un fallback amplio con 'a'.
    - Si la API falla, devuelve 200 con teams=[], y el error en 'error' para debug.
    """
    try:
        if q and q.strip():
            teams = await apisports.teams_national_search_smart(q.strip())
        else:
            teams = await apisports.teams_national_search("a")
        return {"count": len(teams), "teams": teams}
    except Exception as e:
        print("[/teams/national] ERROR:", repr(e))
        return JSONResponse(status_code=200, content={"count": 0, "teams": [], "error": str(e)})

@app.post("/predict/national")
async def predict_national(inp: NationalPredictIn):
    """
    Predicción Local vs Visita entre selecciones:
    - Usa partidos recientes reales (ApiSports) para estimar λ por Poisson
    - Devuelve probabilidades Home/Draw/Away y marcadores más probables
    """
    if not APISPORTS_KEY:
        raise HTTPException(status_code=500, detail="Falta APISPORTS_KEY en backend/.env")

    lam_h, lam_a = await _poisson_rates_from_recent(
        inp.home_id, inp.away_id, n=inp.lookback, neutral=inp.neutral
    )

    max_g = 6
    def pois(lam, k): return math.exp(-lam) * (lam**k) / math.factorial(k)
    pmf_h = np.array([pois(lam_h, k) for k in range(max_g+1)])
    pmf_a = np.array([pois(lam_a, k) for k in range(max_g+1)])
    mat = np.outer(pmf_h, pmf_a)

    prob_home = float(np.tril(mat, -1).sum())
    prob_draw = float(np.trace(mat))
    prob_away = float(np.triu(mat, +1).sum())

    # top marcadores
    flat = [((h,a), float(mat[h,a])) for h in range(max_g+1) for a in range(max_g+1)]
    flat.sort(key=lambda x: x[1], reverse=True)
    top_scores = [{"score": f"{h}-{a}", "p": round(p, 4)} for (h,a), p in flat[:5]]

    return {
        "lambda": {"home": round(lam_h, 3), "away": round(lam_a, 3)},
        "probabilities": {
            "home": round(prob_home, 4),
            "draw": round(prob_draw, 4),
            "away": round(prob_away, 4),
        },
        "top_scores": top_scores,
        "meta": {"neutral": inp.neutral, "lookback": inp.lookback},
    }

# ---------- Utils varias ----------
def _col_ok(df, c):
    return (c in df.columns) and (df[c].notna().sum() > 0)

def _recent_mean_home(df, team, col, n=10):
    if not _col_ok(df, col): return None
    d = df.loc[df["HomeTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _recent_mean_away(df, team, col, n=10):
    if not _col_ok(df, col): return None
    d = df.loc[df["AwayTeam"] == team, col].dropna().tail(n)
    return float(d.mean()) if len(d) else None

def _sum_opt(a, b):
    if a is None and b is None: return None
    return (a or 0.0) + (b or 0.0)

def _poisson_cdf(k, lam):
    if lam is None: return None
    k = int(k)
    s = 0.0
    for i in range(0, k+1):
        s += math.exp(-lam) * (lam**i) / math.factorial(i)
    return s

def _poisson_pmf(k, lam):
    if lam is None: return None
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def _clip01(x):
    if x is None: return None
    return max(0.0, min(1.0, float(x)))

# === Selecciones: utilidades Poisson a partir de partidos recientes ===
async def _fixtures_recent(team_id: int, n: int = 10):
    return await apisports.fixtures_national(team_id, last=n)

def _stats_from_fixtures(recent: list[dict], team_id: int):
    """Devuelve promedios GF/GA y desgloses home/away a partir de fixtures ApiSports."""
    gf, ga, home_gf, away_gf = [], [], [], []
    for fx in recent:
        fixture = fx.get("fixture") or {}
        teams   = fx.get("teams") or {}
        goals   = fx.get("goals") or {}
        th = (teams.get("home") or {}).get("id")
        ta = (teams.get("away") or {}).get("id")
        gh = goals.get("home")
        ga_ = goals.get("away")
        if gh is None or ga_ is None:
            continue
        if th == team_id:
            gf.append(gh); ga.append(ga_)
            home_gf.append(gh)
        elif ta == team_id:
            gf.append(ga_); ga.append(gh)
            away_gf.append(ga_)
    import statistics as st
    def avg(xs): return st.mean(xs) if xs else 1.1  # evita 0 total
    return {
        "gf": avg(gf), "ga": avg(ga),
        "gf_home": avg(home_gf), "gf_away": avg(away_gf),
    }

async def _poisson_rates_from_recent(home_id: int, away_id: int, n: int = 10, neutral: bool = False):
    """Estimación rápida de λ_home/λ_away desde promedios recientes + ajuste de localía."""
    h_recent, a_recent = await _fixtures_recent(home_id, n), await _fixtures_recent(away_id, n)
    hs, as_ = _stats_from_fixtures(h_recent, home_id), _stats_from_fixtures(a_recent, away_id)

    lam_home = (hs["gf"] + as_["ga"]) / 2.0
    lam_away = (as_["gf"] + hs["ga"]) / 2.0

    if not neutral:
        lam_home *= 1.10   # pequeño boost de localía
        lam_away *= 0.90

    # límites suaves para estabilidad numérica
    lam_home = max(0.2, min(3.5, lam_home))
    lam_away = max(0.2, min(3.5, lam_away))
    return lam_home, lam_away

# ---------- Endpoints de datos (UI) ----------
class _PredictPayload(BaseModel):
    pass

@app.get("/recent")
def recent(limit: int = 10):
    _ensure_data_loaded()
    df = df_global
    if df is None or len(df) == 0:
        return {"matches": []}

    def _first_col(df, primary, alts):
        for c in (primary, *alts):
            if c in df.columns:
                return c
        return None

    colL = _first_col(df, "League", ["Div"])
    colH = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    colA = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    colHG = _first_col(df, "FTHG", ["HG", "HomeGoals", "Home_Goals"])
    colAG = _first_col(df, "FTAG", ["AG", "AwayGoals", "Away_Goals"])
    colD  = _first_col(df, "Date", ["MatchDate", "DateStr", "Fecha"])

    if not (colL and colH and colA):
        return {"matches": []}

    dfx = df[[c for c in [colD, colL, colH, colA, colHG, colAG] if c in df.columns]].copy()
    if colD in dfx.columns:
        dfx["_d"] = pd.to_datetime(dfx[colD], errors="coerce", dayfirst=True)
    else:
        dfx["_d"] = pd.NaT

    dfx = dfx.sort_values("_d", ascending=False).head(int(limit))

    rows = []
    for _, r in dfx.iterrows():
        rows.append({
            "Date": (r["_d"].strftime("%Y-%m-%d") if pd.notnull(r["_d"]) else str(r.get(colD, ""))),
            "League": str(r.get(colL, "")),
            "HomeTeam": str(r.get(colH, "")),
            "AwayTeam": str(r.get(colA, "")),
            "FTHG": (None if colHG not in dfx.columns or pd.isna(r.get(colHG)) else int(r.get(colHG))),
            "FTAG": (None if colAG not in dfx.columns or pd.isna(r.get(colAG)) else int(r.get(colAG))),
        })
    return {"matches": rows}

@app.get("/leagues")
def leagues():
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"leagues": []}

    def _first_col(df, primary, alts):
        for c in (primary, *alts):
            if c in df.columns:
                return c
        return None

    colL = _first_col(df, "League", ["Div"])
    if not colL:
        return {"leagues": []}
    lgs = sorted([str(x) for x in df[colL].dropna().unique()])
    return {"leagues": lgs}

@app.get("/teams")
def teams(league: Optional[str] = Query(None, description="Código liga: ej E0, SP1, I1")):
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"league": league, "teams": []}

    def _first_col(df, primary, alts):
        for c in (primary, *alts):
            if c in df.columns:
                return c
        return None

    colL = _first_col(df, "League", ["Div"])
    if league and colL:
        df = df[df[colL] == league]
    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col:
        return {"league": league, "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"league": league, "teams": all_teams}

@app.get("/health")
def health():
    _ensure_data_loaded()
    df = df_global
    if df is None:
        return {"status": "empty", "teams": []}

    def _first_col(df, primary, alts):
        for c in (primary, *alts):
            if c in df.columns:
                return c
        return None

    home_col = _first_col(df, "HomeTeam", ["Home", "Home_Team"])
    away_col = _first_col(df, "AwayTeam", ["Away", "Away_Team"])
    if not home_col or not away_col:
        return {"status": "ok", "teams": []}
    homes = set(df[home_col].dropna().astype(str).unique())
    aways = set(df[away_col].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    return {"status": "ok", "teams": all_teams}

# ---------- Predicción ----------
@app.get("/predict", response_model=PredictResponse)
def predict(
    home: str = Query(..., description="Equipo local"),
    away: str = Query(..., description="Equipo visitante"),
):
    start_total = time.time()
    _ensure_data_loaded()

    # Poisson base
    t0 = time.time()
    res = model.predict(home, away)
    res["exp_goals_total"] = round(res["exp_goals_home"] + res["exp_goals_away"], 3)
    res["src_1x2"] = "poisson"
    res["src_ou25"] = "xg"
    res["src_btts"] = "poisson"
    print(f"[PERF] Poisson tomó {time.time() - t0:.3f} s")

    # 1X2 (ML)
    if clf_bundle:
        t1 = time.time()
        Xy = _get_Xy()

        def prof(Xy, team, is_home):
            cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
            mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
            return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c: 0.0 for c in cols})

        hp, ap = prof(Xy, home, True), prof(Xy, away, False)
        Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
        Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
        if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
        if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

        row = pd.DataFrame([{
            "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
            **{k: float(v) for k, v in hp.items()},
            **{k: float(v) for k, v in ap.items()},
        }])
        feats = clf_bundle["features"]
        row = row.reindex(columns=feats).fillna(row.median(numeric_only=True))
        probs = clf_bundle["model"].predict_proba(row)[0]  # [p_away, p_draw, p_home]
        res["p_away"], res["p_draw"], res["p_home"] = map(lambda x: round(float(x), 4), probs)
        res["src_1x2"] = "ml"
        print(f"[PERF] ML 1X2 tomó {time.time() - t1:.3f} s")

    # Over/Under 2.5 (ML con fallback seguro)
    if clf_ou_bundle:
        try:
            t2 = time.time()
            Xy = _get_Xy()

            def _prof_ou(Xy, team, is_home):
                cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
                mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
                if mask.any():
                    return Xy.loc[mask, cols].tail(5).median()
                return pd.Series({c: 0.0 for c in cols})

            hp, ap = _prof_ou(Xy, home, True), _prof_ou(Xy, away, False)
            Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
            Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
            if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
            if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

            row_ou = pd.DataFrame([{
                "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
                **{k: float(v) for k, v in hp.items()},
                **{k: float(v) for k, v in ap.items()},
            }])
            feats_ou = clf_ou_bundle["features"]
            row_ou = row_ou.reindex(columns=feats_ou).fillna(row_ou.median(numeric_only=True))

            p_over = float(clf_ou_bundle["model"].predict_proba(row_ou)[0, 1])

            res["ou_over25"] = round(p_over, 4)
            res["ou_under25"] = round(1.0 - p_over, 4)
            res["src_ou25"] = "ml"
            print(f"[PERF] ML OU tomó {time.time() - t2:.3f} s")
        except Exception as e:
            xt = res.get("exp_goals_total")
            if xt is None:
                eh = float(res.get("exp_goals_home", 0.0))
                ea = float(res.get("exp_goals_away", 0.0))
                xt = eh + ea
            p_over = 1.0 / (1.0 + math.exp(-1.1 * (xt - 2.5)))
            res["ou_over25"] = round(p_over, 4)
            res["ou_under25"] = round(1.0 - p_over, 4)
            res["src_ou25"] = "xg"
            print("[WARN] OU ML fallback por error:", repr(e))
    else:
        xt = res.get("exp_goals_total")
        if xt is None:
            eh = float(res.get("exp_goals_home", 0.0))
            ea = float(res.get("exp_goals_away", 0.0))
            xt = eh + ea
        p_over = 1.0 / (1.0 + math.exp(-1.1 * (xt - 2.5)))
        res["ou_over25"] = round(p_over, 4)
        res["ou_under25"] = round(1.0 - p_over, 4)
        res["src_ou25"] = "xg"

    # BTTS (ML con fallback seguro)
    if clf_btts_bundle:
        try:
            t3 = time.time()
            Xy = _get_Xy()

            def _prof_bt(Xy, team, is_home):
                cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
                mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
                if mask.any():
                    return Xy.loc[mask, cols].tail(5).median()
                return pd.Series({c: 0.0 for c in cols})

            hp, ap = _prof_bt(Xy, home, True), _prof_bt(Xy, away, False)
            Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
            Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
            if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
            if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

            row_bt = pd.DataFrame([{
                "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
                **{k: float(v) for k, v in hp.items()},
                **{k: float(v) for k, v in ap.items()},
            }])
            feats_bt = clf_btts_bundle["features"]
            row_bt = row_bt.reindex(columns=feats_bt).fillna(row_bt.median(numeric_only=True))

            p_yes = float(clf_btts_bundle["model"].predict_proba(row_bt)[0, 1])

            res["btts_yes"] = round(p_yes, 4)
            res["btts_no"]  = round(1.0 - p_yes, 4)
            res["src_btts"] = "ml"
            print(f"[PERF] ML BTTS tomó {time.time() - t3:.3f} s")
        except Exception as e:
            lam_h = float(res.get("exp_goals_home", 0.0))
            lam_a = float(res.get("exp_goals_away", 0.0))
            p_yes = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
            res["btts_yes"] = round(p_yes, 4)
            res["btts_no"]  = round(1.0 - p_yes, 4)
            res["src_btts"] = "poisson"
            print("[WARN] BTTS ML fallback por error:", repr(e))
    else:
        lam_h = float(res.get("exp_goals_home", 0.0))
        lam_a = float(res.get("exp_goals_away", 0.0))
        p_yes = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        res["btts_yes"] = round(p_yes, 4)
        res["btts_no"]  = round(1.0 - p_yes, 4)
        res["src_btts"] = "poisson"

    # ======== PÍLDORAS extra: córners, amarillas, rojas ========
    df = df_global  # usa tu dataframe global
    eh = float(res.get("exp_goals_home", 0.0))
    ea = float(res.get("exp_goals_away", 0.0))
    et = float(res.get("exp_goals_total", eh + ea)) or (eh + ea)
    share_h = 0.5 if et <= 1e-9 else max(0.15, min(0.85, eh / et))  # reparte al local

    # --- Córners ---
    exp_ch = exp_ca = exp_ct = over95 = under95 = None
    if df is not None and _col_ok(df, "HC") and _col_ok(df, "AC"):
        exp_ch = _recent_mean_home(df, home, "HC", n=10)
        exp_ca = _recent_mean_away(df, away, "AC", n=10)
        exp_ct = _sum_opt(exp_ch, exp_ca)
    else:
        exp_ct = max(6.0, min(15.0, 4.2 + 2.6 * et))
        exp_ch = exp_ct * share_h
        exp_ca = exp_ct * (1.0 - share_h)
    cdf9 = _poisson_cdf(9, exp_ct) if exp_ct is not None else None
    if cdf9 is not None:
        over95, under95 = max(0.0, min(1.0, 1.0 - cdf9)), max(0.0, min(1.0, cdf9))
    res["exp_corners_home"]  = round(exp_ch, 2) if exp_ch is not None else None
    res["exp_corners_away"]  = round(exp_ca, 2) if exp_ca is not None else None
    res["exp_corners_total"] = round(exp_ct, 2) if exp_ct is not None else None
    res["corners_over95"]    = round(over95, 4) if over95 is not None else None
    res["corners_under95"]   = round(under95, 4) if under95 is not None else None

    # --- Amarillas (aprox lineal con xG si no hay columnas YC) ---
    exp_yh = exp_ya = None
    if df is not None and _col_ok(df, "HY") and _col_ok(df, "AY"):
        exp_yh = _recent_mean_home(df, home, "HY", n=10)
        exp_ya = _recent_mean_away(df, away, "AY", n=10)
    else:
        base_y = max(2.5, min(7.0, 3.0 + 0.7 * et))
        exp_yh = base_y * (0.55 * share_h + 0.45 * (1 - share_h))
        exp_ya = base_y - exp_yh
    res["exp_yellows_home"] = round(exp_yh, 2) if exp_yh is not None else None
    res["exp_yellows_away"] = round(exp_ya, 2) if exp_ya is not None else None

    # --- Rojas (muy raras; pequeña fracción de amarillas) ---
    exp_rh = exp_ra = None
    if df is not None and _col_ok(df, "HR") and _col_ok(df, "AR"):
        exp_rh = _recent_mean_home(df, home, "HR", n=15)
        exp_ra = _recent_mean_away(df, away, "AR", n=15)
    else:
        ratio_r = 0.08  # ~8% de las amarillas derivan en roja (aprox)
        exp_rh = ratio_r * (exp_yh or 0.0)
        exp_ra = ratio_r * (exp_ya or 0.0)
    res["exp_reds_home"] = round(exp_rh, 3) if exp_rh is not None else None
    res["exp_reds_away"] = round(exp_ra, 3) if exp_ra is not None else None

    # Distribución de goles (0/1/2/3+)
    lam = res.get("exp_goals_total", None)
    if lam is not None:
        p0 = _poisson_pmf(0, lam)
        p1 = _poisson_pmf(1, lam)
        p2 = _poisson_pmf(2, lam)
        if None not in (p0, p1, p2):
            res["p_goals_0"]     = round(p0, 4)
            res["p_goals_1"]     = round(p1, 4)
            res["p_goals_2"]     = round(p2, 4)
            res["p_goals_3plus"] = round(max(0.0, min(1.0, 1.0 - (p0 + p1 + p2))), 4)

    total_elapsed = time.time() - start_total
    print(f"[PERF] Predicción COMPLETA {home} vs {away} tomó {total_elapsed:.3f} s")

    return PredictResponse(**res)

# ---------- Explicaciones ----------
@app.get("/explain")
def explain(
    home: str = Query(..., description="Equipo local"),
    away: str = Query(..., description="Equipo visitante"),
):
    _ensure_data_loaded()
    Xy = _get_Xy()

    def team_prof(Xy, team, is_home):
        cols = [c for c in Xy.columns if c.startswith("H_" if is_home else "A_")]
        mask = (Xy["HomeTeam"] == team) if is_home else (Xy["AwayTeam"] == team)
        return Xy.loc[mask, cols].tail(5).median() if len(Xy.loc[mask, cols]) else pd.Series({c:0.0 for c in cols})

    hp, ap = team_prof(Xy, home, True), team_prof(Xy, away, False)
    Elo_H = Xy.loc[Xy["HomeTeam"] == home, "Elo_H"].tail(1)
    Elo_A = Xy.loc[Xy["AwayTeam"] == away, "Elo_A"].tail(1)
    if len(Elo_H) == 0: Elo_H = pd.Series([Xy["Elo_H"].median()])
    if len(Elo_A) == 0: Elo_A = pd.Series([Xy["Elo_A"].median()])

    base_row = pd.DataFrame([{
        "Elo_H": float(Elo_H.values[-1]), "Elo_A": float(Elo_A.values[-1]),
        **{k: float(v) for k, v in hp.items()},
        **{k: float(v) for k, v in ap.items()},
    }])

    expl = {"home": home, "away": away, "reasons": {}}

    if clf_bundle:
        feats = clf_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["1x2"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["1x2"] = ["Poisson por xG de ambos"]

    if clf_ou_bundle:
        feats = clf_ou_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["ou25"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["ou25"] = ["xG total vs 2.5"]

    if clf_btts_bundle:
        feats = clf_btts_bundle["features"]
        row = base_row.reindex(columns=feats).fillna(base_row.median(numeric_only=True))
        expl["reasons"]["btts"] = top_drivers(row, feats, 3)
    else:
        expl["reasons"]["btts"] = ["Prob.(H>0)·Prob.(A>0) por Poisson"]

    return expl

# ---------- Rutas de comodidad ----------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")

@app.get("/app/index.html", include_in_schema=False)
def app_index_file():
    return FileResponse(FRONT_DIR / "index.html")

@app.get("/status", response_class=HTMLResponse)
def status_page():
    csv = MERGED if MERGED.exists() else DATA_PATH
    df = pd.read_csv(csv)
    homes = set(df["HomeTeam"].dropna().astype(str).unique())
    aways = set(df["AwayTeam"].dropna().astype(str).unique())
    all_teams = sorted(list(homes.union(aways)))
    t0 = all_teams[0] if all_teams else "TeamA"
    t1 = all_teams[1] if len(all_teams) > 1 else "TeamB"
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Status</title></head>
      <body style="font-family:system-ui;background:#0f1115;color:#e9e9f1">
        <div style="max-width:900px;margin:40px auto">
          <h1>✅ API viva</h1>
          <p>Equipos cargados (todas las ligas): <b>{len(all_teams)}</b></p>
          <p>Prueba rápida: <code>/predict?home={t0}&amp;away={t1}</code></p>
          <p>Docs: <a href="/docs">/docs</a></p>
          <p>App: <a href="/app/">/app/</a></p>
        </div>
      </body>
    </html>
    """
    return html

# ---------- Estado del bot / métricas ----------
@app.get("/bot_status")
def bot_status():
    return {
        "data_rows": (0 if df_global is None else int(len(df_global))),
        "features_ready": Xy_global is not None,
        "models": {
            "poisson": True,
            "ml_1x2": clf_bundle is not None,
            "ml_ou25": clf_ou_bundle is not None,
            "ml_btts": clf_btts_bundle is not None,
        },
        "current_csv": str(current_csv_path) if current_csv_path else None,
    }

@app.get("/importances")
def importances():
    if clf_bundle and "model" in clf_bundle:
        try:
            mdl = clf_bundle["model"]
            feats = clf_bundle["features"]
            if hasattr(mdl, "feature_importances_"):
                vals = mdl.feature_importances_
                top = sorted(zip(feats, vals), key=lambda x: -x[1])[:10]
                return {"top_features": [{"name": f, "importance": float(v)} for f, v in top]}
        except Exception as e:
            return {"error": str(e)}
    return {"top_features": []}

@app.get("/metrics")
def metrics():
    _ensure_data_loaded()
    if df_global is None:
        return {"status": "empty"}

    def _first_col(df, primary, alts):
        for c in (primary, *alts):
            if c in df.columns:
                return c
        return None

    n_matches = len(df_global)
    leagues = sorted(df_global[_first_col(df_global, "League", ["Div"])].unique()) if _first_col(df_global, "League", ["Div"]) else []
    teams = set(df_global[_first_col(df_global, "HomeTeam", ["Home", "Home_Team"])].dropna()) | set(df_global[_first_col(df_global, "AwayTeam", ["Away", "Away_Team"])].dropna())

    return {
        "matches": int(n_matches),
        "leagues": leagues,
        "n_teams": len(teams),
    }

# ---------- Mini monitor ----------
@app.get("/monitor", response_class=HTMLResponse)
def monitor():
    html = """
    <!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Monitor — Sexy Sports Predictor</title>
      <link rel="preconnect" href="https://cdn.jsdelivr.net">
      <style>
        :root{
          --bg:#0b0d12; --panel:#11151d; --muted:#9aa3b2; --text:#e6eaf2; --brand:#8b5cf6; --brand2:#06b6d4;
          --ring: 0 0 0 2px hsl(255 85% 65% / .35);
        }
        *{box-sizing:border-box}
        body{margin:0;background:radial-gradient(1200px 800px at 80% -10%,#13203a 0%,transparent 60%),
                         radial-gradient(1000px 600px at -10% 90%,#1e123a 0%,transparent 60%),var(--bg);
             color:var(--text);font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial}
        .container{max-width:1100px;margin:0 auto;padding:22px}
        h1{margin:6px 0 18px;font-size:26px}
        .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
        .col-4{grid-column:span 4} .col-6{grid-column:span 6} .col-8{grid-column:span 8} .col-12{grid-column:span 12}
        @media (max-width:900px){.col-4,.col-6,.col-8{grid-column:span 12}}
        .card{background:color-mix(in oklab,var(--panel) 86%,black 14%);border:1px solid #1c2230;border-radius:18px;padding:18px;box-shadow:0 8px 30px #0008}
        .muted{color:var(--muted)}
        .pill{display:inline-flex;gap:8px;align-items:center;background:#0e1320;border:1px solid #20283a;padding:7px 10px;border-radius:999px;margin:4px 6px 0 0}
        .ok{color:#22c55e} .warn{color:#f59e0b}
        button{cursor:pointer;padding:10px 14px;border-radius:12px;border:1px solid #20283a;background:#0f1420;color:var(--text)}
        button.btn{background:linear-gradient(135deg,var(--brand),var(--brand2));color:white;border:none;font-weight:700}
        code{background:#0f1420;border:1px solid #20283a;border-radius:8px;padding:2px 6px}
        ul{margin:8px 0 0 18px}
        footer{margin:26px 0 8px;text-align:center;color:#7f8798;font-size:13px}
      </style>
    </head>
    <body>
      <div class="container">
        <h1>📊 Monitor — <span class="muted">Sexy Sports Predictor</span></h1>

        <div class="grid">
          <div class="col-8">
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
                <h3 style="margin:0">Estado del bot</h3>
                <div class="muted" id="csvPath">—</div>
              </div>
              <div id="botState" style="margin-top:8px"></div>
              <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
                <button class="btn" id="btnWarmup">⚡ Precalentar features</button>
                <button id="btnRefresh">🔄 Refrescar</button>
                <a href="/docs" target="_blank"><button>📚 API Docs</button></a>
                <a href="/app/" target="_blank"><button>🖥️ Abrir App</button></a>
              </div>
            </div>
          </div>

          <div class="col-4">
            <div class="card">
              <h3 style="margin:0 0 8px">Métricas</h3>
              <div class="muted" id="metricsSummary">Cargando…</div>
              <ul id="leaguesList"></ul>
            </div>
          </div>

          <div class="col-12">
            <div class="card">
              <h3 style="margin:0 0 8px">Top features (1X2)</h3>
              <canvas id="featChart" height="120"></canvas>
              <div class="muted" id="featNote" style="margin-top:8px"></div>
            </div>
          </div>
        </div>

        <footer>Hecho con 💜 por nosotros</footer>
      </div>

      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
      <script>
        const $ = s => document.querySelector(s);
        const api = (p, opt) => fetch(p, opt).then(r => { if(!r.ok) throw new Error(r.status); return r.json(); });

        let chart;

        async function loadAll(){
          const bs = await api('/bot_status');
          $('#csvPath').textContent = bs.current_csv || '—';
          const s = [];
          s.push(`<span class="pill">filas: <b>${bs.data_rows}</b></span>`);
          s.push(`<span class="pill">features: <b class="${bs.features_ready ? 'ok':'warn'}">${bs.features_ready ? 'listas':'no listas'}</b></span>`);
          s.push(`<span class="pill">Poisson: <b class="ok">${bs.models.poisson ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML 1X2: <b class="${bs.models.ml_1x2 ? 'ok':'warn'}">${bs.models.ml_1x2 ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML O/U: <b class="${bs.models.ml_ou25 ? 'ok':'warn'}">${bs.models.ml_ou25 ? 'OK':'—'}</b></span>`);
          s.push(`<span class="pill">ML BTTS: <b class="${bs.models.ml_btts ? 'ok':'warn'}">${bs.models.ml_btts ? 'OK':'—'}</b></span>`);
          $('#botState').innerHTML = s.join(' ');

          const m = await api('/metrics');
          if(m.status === 'empty'){
            $('#metricsSummary').textContent = 'Sin datos cargados.';
            $('#leaguesList').innerHTML = '';
          }else{
            $('#metricsSummary').textContent = `Partidos: ${m.matches} · Equipos: ${m.n_teams}`;
            const ul = $('#leaguesList'); ul.innerHTML = '';
            (m.leagues || []).slice(0,12).forEach(code=>{
              const li = document.createElement('li');
              li.textContent = code;
              ul.appendChild(li);
            });
          }

          const imp = await api('/importances');
          const items = (imp.top_features || []);
          $('#featNote').textContent = items.length ? '' : 'Sin importancias disponibles (¿modelo ML 1X2 entrenado?).';

          const labels = items.map(x=>x.name);
          const data = items.map(x=>x.importance);

          const ctx = document.getElementById('featChart');
          const cfg = {
            type: 'bar',
            data: { labels, datasets:[{ label:'Importancia', data }] },
            options: {
              responsive:true,
              scales:{ y:{ beginAtZero:true } },
              plugins:{ legend:{ display:false } }
            }
          };
          if(chart){ chart.data = cfg.data; chart.update(); }
          else { chart = new Chart(ctx, cfg); }
        }

        $('#btnRefresh').addEventListener('click', ()=>loadAll());
        $('#btnWarmup').addEventListener('click', async ()=>{
          try{
            const r = await api('/warmup', {method:'POST'});
            alert('Warmup: ' + (r.features_ready ? 'features listas' : 'features no listas'));
            loadAll();
          }catch(e){ alert('No se pudo precalentar'); }
        });

        loadAll();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ---------- Warmup ----------
@app.post("/warmup")
def warmup():
    _ensure_data_loaded()
    _ensure_features()
    return {"status": "ok", "features_ready": Xy_global is not None}

# =================== ENTRENAMIENTO MULTI-TEMPORADA (nuevo) ===================

class TrainHistoryRequest(BaseModel):
    league: str
    seasons: Optional[List[str]] = None  # si no mandan, usamos DEFAULT_SEASONS

@app.post("/train_history")
def train_with_history(body: TrainHistoryRequest):
    """Re-entrena Poisson (y features) con 5 temporadas (o las que mandes)."""
    league = body.league.upper()
    if league not in SUPPORTED_LEAGUES:
        raise HTTPException(status_code=400, detail=f"Liga no soportada: {league}")

    seasons = body.seasons or DEFAULT_SEASONS

    try:
        df = load_league_history(league, seasons)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Reutiliza tu pipeline existente:
    global df_global, Xy_global, current_csv_path, model
    df_global = df.copy()
    current_csv_path = None  # multi-temporada, no hay un único CSV
    model.fit(df_global)

    try:
        Xy_global = build_training_table(df_global, window=10)
    except Exception as e:
        Xy_global = None
        print("[WARN] build_training_table falló:", e)

    return {
        "message": "Modelo re-entrenado (multi-temporada)",
        "league": league,
        "league_name": SUPPORTED_LEAGUES[league],
        "seasons": seasons,
        "rows": int(df_global.shape[0]),
        "from": (str(df_global["Date"].min()) if "Date" in df_global else None),
        "to": (str(df_global["Date"].max()) if "Date" in df_global else None),
    }

@app.get("/data/status")
def data_status():
    """Estado simple de los datos cargados para depurar."""
    if df_global is None:
        return {"loaded": False}
    out = {"loaded": True, "rows": int(df_global.shape[0])}
    if "Date" in df_global:
        out["from"] = str(df_global["Date"].min())
        out["to"] = str(df_global["Date"].max())
    return out

@app.get("/leagues/supported")
def leagues_supported():
    """Lista códigos soportados + si ya hay CSVs en data/raw/<code>."""
    base = Path(__file__).parent / "data" / "raw"
    out = []
    for code, name in SUPPORTED_LEAGUES.items():
        liga_dir = base / code
        has_data = liga_dir.exists() and any(liga_dir.glob("*.csv"))
        seasons = sorted([p.stem for p in liga_dir.glob("*.csv")]) if liga_dir.exists() else []
        out.append({
            "code": code,
            "name": name,
            "has_data": has_data,
            "seasons": seasons
        })
    return {"leagues": out}
