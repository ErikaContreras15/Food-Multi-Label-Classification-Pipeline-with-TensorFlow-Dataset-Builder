# lista_corregido.py
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

print("="*80)
print("TODAS LAS VERSIONES DEL MODELO (INCLUIDAS LAS HISTÓRICAS)")
print("="*80)

# Buscar TODAS las versiones sin filtrar por stage
models = client.search_registered_models()
for model in models:
    print(f"\nModelo: '{model.name}'")
    versions = client.search_model_versions(f"name='{model.name}'")
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    
    if not versions_sorted:
        print("  ❌ No hay versiones")
    else:
        print(f"  ✅ Total de versiones: {len(versions_sorted)}")
        for v in versions_sorted:
            print(f"    → Versión {v.version:2d} | Run ID: {v.run_id[:12]} | Stage: {v.current_stage:10s} | Creada: {v.creation_timestamp}")
print("="*80)