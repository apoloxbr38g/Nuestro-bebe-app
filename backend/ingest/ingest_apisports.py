# backend/ingest/ingest_apisports.py
import asyncio, os, time, calendar
from datetime import date
from pathlib import Path
import pandas as pd
import aiohttp

# ── .env opcional ─────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(ENV_PATH)
    except Exception:
        pass

API_KEY = os.getenv("APISPORTS_KEY", "").strip()
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

if not API_KEY:
    raise SystemExit("APISPORTS_KEY no está definido (backend/.env o var de entorno).")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _row(item: dict):
    f = item.get("fixture", {}) or {}
    l = item.get("league", {}) or {}
    t = item.get("teams", {})  or {}
    g = item.get("goals", {})  or {}
    return {
        "Date": (f.get("date") or "")[:10],
        "Div":  "",
        "League": l.get("name"),
        "Season": l.get("season"),
        "HomeTeam": (t.get("home") or {}).get("name"),
        "AwayTeam": (t.get("away") or {}).get("name"),
        "FTHG": g.get("home"),
        "FTAG": g.get("away"),
        "Status": (f.get("status") or {}).get("short"),
    }

async def _get(session: aiohttp.ClientSession, path: str, params: dict, pause: float = 1.0, max_retries: int = 4):
    url = BASE + path
    attempt = 0
    wait = pause
    while True:
        async with session.get(url, headers=HEADERS, params=params) as r:
            if r.status == 429:
                txt = await r.text()
                attempt += 1
                if attempt > max_retries:
                    return {"errors": {"status": 429, "message": txt}, "response": [], "paging": {"total": 1}}
                time.sleep(wait)
                wait = min(wait * 1.6, 6.0)
                continue
            if r.status >= 400:
                txt = await r.text()
                return {"errors": {"status": r.status, "message": txt}, "response": [], "paging": {"total": 1}}
            return await r.json()

async def _seasons_for_league(session: aiohttp.ClientSession, league_id: int) -> list[int]:
    data = await _get(session, "/leagues", {"id": league_id})
    seasons = []
    for it in data.get("response", []):
        for s in it.get("seasons", []):
            y = s.get("year")
            if isinstance(y, int):
                seasons.append(y)
    return sorted(set(seasons))

async def _fetch_by_season(session: aiohttp.ClientSession, league_id: int, season: int, pause: float = 1.0):
    rows = []
    page = 1
    while True:
        data = await _get(session, "/fixtures", {"league": league_id, "season": season, "page": page}, pause=pause)
        rows.extend([_row(x) for x in data.get("response", [])])
        paging = data.get("paging") or {}
        cur = paging.get("current") or page
        tot = paging.get("total") or 1
        if cur >= tot:
            break
        page += 1
        time.sleep(pause)
    return rows

async def _fetch_by_months(session: aiohttp.ClientSession, league_id: int, season: int, pause: float = 1.0):
    """
    Fallback: recorre enero..diciembre con from/to + season (algunos tenants lo exigen).
    """
    rows = []
    for m in range(1, 13):
        last_day = calendar.monthrange(season, m)[1]
        params = {
            "league": league_id,
            "season": season,
            "from": f"{season}-{m:02d}-01",
            "to":   f"{season}-{m:02d}-{last_day:02d}",
            "timezone": "UTC",
        }
        data = await _get(session, "/fixtures", params, pause=pause)
        rows.extend([_row(x) for x in data.get("response", [])])
        time.sleep(pause)
    return rows

async def ingest_league_seasons(league_id: int, seasons: list[int], internal_code: str, out_dir: Path, pause: float = 1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    first = min(seasons) if seasons else "auto"
    last  = max(seasons) if seasons else "auto"
    out_csv = out_dir / f"{internal_code}_{first}-{last}.csv"

    async with aiohttp.ClientSession() as session:
        if not seasons:
            seasons = await _seasons_for_league(session, league_id)
        print(f"[INGEST] Liga {league_id} seasons: {seasons}")

        all_rows = []
        for s in seasons:
            print(f"  → Temporada {s} (paginado season)…")
            rows = await _fetch_by_season(session, league_id, s, pause=pause)
            print(f"    · {len(rows)} partidos via season")
            if len(rows) == 0:
                print(f"    · 0 filas; fallback por meses (from/to + season)…")
                rows = await _fetch_by_months(session, league_id, s, pause=pause)
                print(f"    · {len(rows)} partidos via meses")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["Div"] = internal_code
        base = ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"]
        df = df[base + [c for c in df.columns if c not in base]]
    df.to_csv(out_csv, index=False)
    print(f"[OK] Guardado {out_csv} con {len(df)} partidos")
    return out_csv

# ── Config: ligas objetivo ────────────────────────────────────────────────────
TARGET_LEAGUES = [
    {"league_id": 333, "code": "UA1", "folder": "UA", "seasons": [2024, 2025]},  # Ucrania
    {"league_id": 203, "code": "TR1", "folder": "TR", "seasons": [2024, 2025]},  # Turquía
    {"league_id": 98,  "code": "JP1", "folder": "JP", "seasons": [2024, 2025]},  # Japón
    {"league_id": 197, "code": "GR1", "folder": "GR", "seasons": [2024, 2025]},  # Grecia
    {"league_id": 169, "code": "CN1", "folder": "CN", "seasons": [2024, 2025]},  # China
]

async def main():
    for it in TARGET_LEAGUES:
        try:
            await ingest_league_seasons(
                league_id=int(it["league_id"]),
                seasons=list(it.get("seasons") or []),
                internal_code=str(it["code"]),
                out_dir=Path(f"backend/data/raw/{it.get('folder') or it['code'][:2]}"),
                pause=1.1,  # sube este valor si ves muchos 429
            )
        except Exception as e:
            print(f"[WARN] Falló {it['code']} ({it['league_id']}): {e}")

if __name__ == "__main__":
    asyncio.run(main())
