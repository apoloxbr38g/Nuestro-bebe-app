# backend/providers/apisports.py
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
import httpx

# Lee tu key del .env
API_KEY = os.getenv("APISPORTS_KEY")  # asegúrate de tenerla en backend/.env
BASE    = os.getenv("APISPORTS_BASE", "https://v3.football.api-sports.io")

HEADERS = {
    # API-SPORTS acepta este header (nuevo); si tu key es de RapidAPI, cambia por x-rapidapi-key + host
    "x-apisports-key": API_KEY or "",
}

async def _get(path: str, params: dict) -> dict:
    """GET genérico a API-SPORTS; devuelve el JSON completo."""
    url = f"{BASE}{path}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        return r.json()

def _response_list(js: dict) -> list:
    """Extrae la lista estandarizada `response`."""
    if not isinstance(js, dict):
        return []
    return js.get("response", []) or []

# ---------------- Endpoints de conveniencia ----------------

async def leagues_by_country(country: str) -> list[dict]:
    """
    Devuelve ligas por país (nombre/ID/temporadas).
    """
    js = await _get("/leagues", {"country": country})
    resp = _response_list(js)
    out = []
    for item in resp:
        lg = item.get("league") or {}
        seasons = item.get("seasons") or []
        last_season = None
        if seasons:
            # toma la temporada más reciente por year
            last_season = max((s.get("year") for s in seasons if "year" in s), default=None)
        out.append({
            "id": str(lg.get("id")) if lg.get("id") is not None else None,
            "name": lg.get("name"),
            "type": lg.get("type"),
            "country": (item.get("country") or {}).get("name"),
            "season": last_season,
            "raw": item,
        })
    return out

async def teams_by_league(league_id: str, season: int | None = None) -> list[dict]:
    """
    Lista de equipos de una liga (opcionalmente temporada); si no se da temporada,
    intenta usar el año actual.
    """
    if season is None:
        season = datetime.now(timezone.utc).year
    js = await _get("/teams", {"league": league_id, "season": season})
    resp = _response_list(js)
    out = []
    for item in resp:
        team = item.get("team") or {}
        out.append({
            "id": team.get("id"),
            "name": team.get("name"),
            "code": team.get("code"),
            "country": (item.get("venue") or {}).get("country"),
        })
    return out

async def fixtures_by_league(
    league_id: str,
    season: int | None = None,
    date: str | None = None,   # 'YYYY-MM-DD'
    next: int | None = None,   # próximos N
    days: int = 30,            # ventana si no hay date/next
) -> list[dict]:
    """
    Futuros/diarios por liga. Preferencias:
    - si pasas `date`, usa ese día
    - si pasas `next`, usa próximos N
    - si nada de lo anterior, usa rango [hoy, hoy+N días]
    """
    params = {"league": league_id, "timezone": "UTC"}
    if season is not None:
        params["season"] = season

    if date:
        params["date"] = date
    elif next is not None:
        params["next"] = int(next)
    else:
        today = datetime.now(timezone.utc).date()
        to   = today + timedelta(days=max(1, int(days)))
        params["from"] = today.isoformat()
        params["to"]   = to.isoformat()

    js = await _get("/fixtures", params)
    resp = _response_list(js)

    out = []
    for f in resp:
        fixture = f.get("fixture") or {}
        league  = f.get("league") or {}
        teams   = f.get("teams") or {}
        goals   = f.get("goals") or {}

        out.append({
            "id": fixture.get("id"),
            "league": {
                "id": league.get("id"),
                "name": league.get("name"),
                "season": league.get("season"),
            },
            "date_utc": fixture.get("date"),  # ISO UTC
            "status": (fixture.get("status") or {}).get("short"),
            "home": {
                "id": (teams.get("home") or {}).get("id"),
                "name": (teams.get("home") or {}).get("name"),
            },
            "away": {
                "id": (teams.get("away") or {}).get("id"),
                "name": (teams.get("away") or {}).get("name"),
            },
            "goals": {
                "home": goals.get("home"),
                "away": goals.get("away"),
            }
        })
    return out
