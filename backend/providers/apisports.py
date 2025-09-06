# backend/ingest/ingest_apisports.py
import asyncio, os
from pathlib import Path
import pandas as pd

# usamos el mismo cliente que ya funciona en tu backend
from backend.providers import apisports
from dotenv import load_dotenv

load_dotenv()  # toma APISPORTS_KEY de backend/.env

def _row_from_fixture(item):
    f = item.get("fixture", {}) or {}
    l = item.get("league", {}) or {}
    t = item.get("teams", {})  or {}
    g = item.get("goals", {})  or {}

    return {
        "Date": (f.get("date") or "")[:10],
        "Div":  str(l.get("id") or ""),         # luego lo mapeamos a código interno (ej. UA1)
        "League": l.get("name"),
        "Season": l.get("season"),
        "HomeTeam": (t.get("home") or {}).get("name"),
        "AwayTeam": (t.get("away") or {}).get("name"),
        "FTHG": g.get("home"),
        "FTAG": g.get("away"),
        "Status": (f.get("status") or {}).get("short"),
    }

async def _fetch_all_for_season(league_id: int, season: int):
    page = 1
    out  = []
    while True:
        data = await apisports._get("/fixtures", {"league": league_id, "season": season, "page": page})
        # Log de errores informativos (la API devuelve 200 con "errors" a veces)
        if data.get("errors"):
            print(f"[API ERR] season {season} page {page} →", data["errors"])
        resp = data.get("response", [])
        out.extend([_row_from_fixture(x) for x in resp])

        # paginación
        paging = data.get("paging", {}) or {}
        total_pages = int(paging.get("total", 1) or 1)
        if page >= total_pages:
            break
        page += 1
    return out

async def ingest_league(league_id: int, seasons: list[int], out_csv: Path, code_internal: str):
    rows = []
    for y in seasons:
        print(f"Descargando league={league_id} season={y} ...")
        part = await _fetch_all_for_season(league_id, y)
        rows.extend(part)

    df = pd.DataFrame(rows)
    # mapea Div numérico → código interno (ej. "UA1")
    if not df.empty:
        df["Div"] = code_internal
        # columnas básicas ordenadas
        base_cols = ["Date","Div","HomeTeam","AwayTeam","FTHG","FTAG"]
        extras = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + extras]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Guardado {out_csv} con {len(df)} partidos")
    return out_csv

if __name__ == "__main__":
    # ⚠️ Edita aquí lo que quieras ingestar:
    TARGETS = [
        # (league_id, seasons, output_csv_path, internal_code)
        (333, [2022, 2023, 2024, 2025], Path("backend/data/raw/UA/UA1_2022-2025.csv"), "UA1"),  # Ucrania
        # (203, [2022, 2023, 2024, 2025], Path("backend/data/raw/TR/TR1_2022-2025.csv"), "TR1"),  # Turquía
        # (98,  [2022, 2023, 2024, 2025], Path("backend/data/raw/JP/JP1_2022-2025.csv"), "JP1"),  # Japón
        # (197, [2022, 2023, 2024, 2025], Path("backend/data/raw/GR/GR1_2022-2025.csv"), "GR1"),  # Grecia
        # (169, [2022, 2023, 2024, 2025], Path("backend/data/raw/CN/CN1_2022-2025.csv"), "CN1"),  # China
    ]

    async def main():
        outs = []
        for league_id, years, out_csv, code in TARGETS:
            f = await ingest_league(league_id, years, out_csv, code)
            outs.append(str(f))
        print("CSV generados:", outs)

    asyncio.run(main())
# dentro de backend/providers/apisports.py
import datetime as _dt

async def fixtures_by_league(league_id: str, season: int | None = None, date: str | None = None, next: int | None = None, days: int = 30):
    # 1) intento con next
    if next:
        data = await _get("/fixtures", {"league": league_id, "next": int(next), "timezone": "UTC"})
        fx = _normalize_fixtures_list(data)
        if fx:
            return fx

    # 2) rango por temporada actual
    if not season:
        data_league = await _get("/leagues", {"id": league_id})
        season = None
        for it in data_league.get("response", []):
            for s in it.get("seasons", []):
                if s.get("current"): season = int(s.get("year")); break

    today = _dt.date.today()
    if days and season:
        data = await _get("/fixtures", {
            "league": league_id,
            "season": int(season),
            "from": today.isoformat(),
            "to":   (today + _dt.timedelta(days=int(days))).isoformat(),
            "timezone": "UTC",
        })
        fx = _normalize_fixtures_list(data)
        if fx:
            return fx

    # 3) hoy
    if not date:
        date = today.isoformat()
    data = await _get("/fixtures", {"league": league_id, "season": season, "date": date, "timezone": "UTC"})
    return _normalize_fixtures_list(data)

