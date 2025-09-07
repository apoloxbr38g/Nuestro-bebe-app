import os
from dotenv import load_dotenv

# forzamos a cargar backend/.env siempre
ok = load_dotenv("backend/.env")
print("load_dotenv devolvió:", ok)

print("APIFOOTBALL_API_KEY =", os.getenv("APIFOOTBALL_API_KEY"))
print("APISPORTS_KEY =", os.getenv("APISPORTS_KEY"))
