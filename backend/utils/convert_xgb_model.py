import joblib
import xgboost as xgb
import sys
from pathlib import Path

def convert_model(old_path, new_path=None):
    old_path = Path(old_path)
    if not old_path.exists():
        print(f"❌ No encontré el archivo {old_path}")
        return

    # Nombre nuevo si no lo pasaste
    if new_path is None:
        new_path = old_path.with_suffix(".json")

    print(f"📂 Cargando modelo antiguo: {old_path}")
    model = joblib.load(old_path)

    # Aseguramos que sea Booster
    if isinstance(model, xgb.XGBModel):  
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        print("❌ El archivo no parece un modelo XGBoost")
        return

    print(f"💾 Guardando en formato nuevo: {new_path}")
    booster.save_model(str(new_path))
    print("✅ Conversión completada.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convert_xgb_model.py modelo.pkl [modelo.json]")
    else:
        old_file = sys.argv[1]
        new_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_model(old_file, new_file)
