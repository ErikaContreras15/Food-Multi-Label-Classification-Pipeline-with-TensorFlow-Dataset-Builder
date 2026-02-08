# lista_corregido.py
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

print("="*80)
print("DIAGNÓSTICO COMPLETO DE MLFLOW - TODOS LOS MODELOS Y VERSIONES")
print("="*80)

# Listar TODOS los modelos registrados
models = client.search_registered_models()
if not models:
    print("\n❌ NO HAY MODELOS REGISTRADOS EN MLFLOW")
    print("="*80)
    exit()

print(f"\n✅ Total de modelos registrados: {len(models)}\n")

for model in models:
    print(f"{'='*80}")
    print(f"MODELO: '{model.name}'")
    print(f"{'='*80}")
    
    # Obtener TODAS las versiones (no solo las 'latest')
    versions = client.search_model_versions(f"name='{model.name}'")
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    
    if not versions_sorted:
        print("  ❌ No hay versiones para este modelo")
    else:
        print(f"  ✅ Total de versiones: {len(versions_sorted)}\n")
        for v in versions_sorted:
            print(f"  Versión {v.version:2d} | Run ID: {v.run_id[:12]} | Stage: {v.current_stage:10s} | Creada: {v.creation_timestamp}")

print("\n" + "="*80)
print("NOTA: Si ves múltiples modelos con nombres similares,")
print("      ejecuta 'limpiar_modelos.py' para eliminar duplicados.")
print("="*80)