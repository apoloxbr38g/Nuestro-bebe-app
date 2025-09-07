# test_apisports.py
import os, asyncio, httpx
from dotenv import load_dotenv

load_dotenv("backend/.env")
load_dotenv()

API_KEY  = os.getenv("APISPORTS_KEY")
BASE_URL = os.getenv("APISPORTS_BASE", "https://v3.football.api-sports.io")
LEAGUE_ID = 203  # Süper Lig

if not API_KEY:
    print("🔑 Key usada: ❌ no encontrada (APISPORTS_KEY)"); raise SystemExit(1)
print("🔑 Key usada:", API_KEY[:6] + "…")

HEADERS = {"x-apisports-key": API_KEY}

async def choose_season(cx: httpx.AsyncClient) -> list[int]:
    """Devuelve una lista ordenada de temporadas a intentar: [current, luego descendente]."""
    r = await cx.get("/leagues", params={"id": LEAGUE_ID})
    r.raise_for_status()
    resp = r.json().get("response", [])
    if not resp:
        print("⚠️ No se encontró la liga 203"); return []
    seasons = resp[0].get("seasons", [])
    # busca current=true primero
    current = [s["year"] for s in seasons if s.get("current") and isinstance(s.get("year"), int)]
    others  = sorted({s["year"] for s in seasons if isinstance(s.get("year"), int)}, reverse=True)
    ordered = []
    if current:
        # evita duplicado si current ya está en others
        ordered = current + [y for y in others if y not in current]
    else:
        ordered = others
    print("📅 Temporadas (orden de intento):", ordered[:6])
    return ordered

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=30) as cx:
        # sanity: ligas por país
        r1 = await cx.get("/leagues", params={"country": "Turkey"}); r1.raise_for_status()
        print(f"🇹🇷 Ligas en Turquía → {len(r1.json().get('response', []))}")

        # temporadas candidatas
        candidates = await choose_season(cx)
        if not candidates:
            return

        season_used = None
        teams_count = 0
        fixtures_count = 0

        for season in candidates[:5]:  # prueba hasta 5
            # equipos
            r2 = await cx.get("/teams", params={"league": LEAGUE_ID, "season": season})
            r2.raise_for_status()
            teams = r2.json().get("response", [])
            tc = len(teams)
            print(f"👕 Equipos Süper Lig {season} → {tc}")
            if tc > 0:
                season_used = season
                teams_count = tc
                break

        if not season_used:
            print("⚠️ No hubo equipos en las temporadas probadas (intentemos igualmente fixtures con la primera).")
            season_used = candidates[0]

        # fixtures: next y si no, rango
        r3 = await cx.get("/fixtures", params={"league": LEAGUE_ID, "season": season_used, "next": 10, "timezone": "UTC"})
        r3.raise_for_status()
        fixtures = r3.json().get("response", [])
        fixtures_count = len(fixtures)
        print(f"📅 Próximos partidos (season={season_used}, next=10) → {fixtures_count}")

        if fixtures_count == 0:
            from datetime import date, timedelta
            start = date.today()
            end   = start + timedelta(days=21)
            r4 = await cx.get("/fixtures", params={
                "league": LEAGUE_ID, "season": season_used,
                "from": start.isoformat(), "to": end.isoformat(), "timezone": "UTC"
            })
            r4.raise_for_status()
            fixtures2 = r4.json().get("response", [])
            print(f"📅 Próximos partidos (rango {start}→{end}, season={season_used}) → {len(fixtures2)}")
            for f in fixtures2[:3]:
                dt = f.get("fixture",{}).get("date","")
                h  = f.get("teams",{}).get("home",{}).get("name","—")
                a  = f.get("teams",{}).get("away",{}).get("name","—")
                st = f.get("fixture",{}).get("status",{}).get("short","")
                print("  •", dt, "-", h, "vs", a, f"({st})")

if __name__ == "__main__":
    asyncio.run(main())
