import os
import uuid
import numpy as np
from threading import Thread, Lock
import mlflow  

from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf

from retrain_service import IncrementalRetrainer
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  


# ========= CONFIG =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))              
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))      

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model.keras")
IMG_SIZE = (224, 224)

CLASSES = ["lacteos", "arroz", "frutas/verduras"]

THRESH = 0.50
NONE_IF_MAX_BELOW = 0.45

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12MB

# ========= HELPERS =========
def normalize_lang(lang: str) -> str:
    lang = (lang or "es").lower().strip()
    return "en" if lang.startswith("en") else "es"

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in ALLOWED_EXTS

def load_img_for_model(path: str):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)

# ========= FLASK APP =========
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Crear carpeta uploads (y detectar si "uploads" es archivo)
if os.path.exists(UPLOAD_DIR) and not os.path.isdir(UPLOAD_DIR):
    raise RuntimeError(f"❌ '{UPLOAD_DIR}' existe pero NO es carpeta. Borra ese archivo y crea una carpeta.")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ========= I18N =========
I18N = {
    "es": {
        "brand": "Food Detector",
        "nav_home": "Inicio",
        "nav_inspo": "Inspiración",
        "nav_tutorials": "Tutoriales",
        "nav_tools": "Herramientas",
        "nav_pricing": "Precios",
        "btn_feedback": "💬 Comentarios",
        "btn_login": "Iniciar sesión",

        "hero_title": "Generador de Descripción de Imagen",
        "hero_sub": "Sube una imagen (o varias). Obtén predicciones por clase (lácteos, arroz, frutas/verduras) y un veredicto final.",

        "tab_upload": "Subir imagen",
        "tab_info": "Información",
        "tab_models": "Modelos MLflow",  # ✅ Nueva pestaña

        "lang_label": "Idioma del resultado",
        "lang_es": "Español",
        "lang_en": "English",

        "dz_title": "Suba una foto o arrastre y suelte",
        "dz_hint": "PNG, JPG o WEBP • hasta 12MB",
        "dz_pick": "Seleccionar imagen",

        "mode_label": "Modo de carga",
        "mode_single": "Solo 1",
        "mode_multi": "Varias",

        "btn_analyze": "✨ Analizar imagen",
        "formats_hint": "Formatos: JPG, PNG, WEBP",

        "preview_title": "Vista previa de la imagen",
        "preview_empty": "Su imagen se mostrará aquí",

        "results_title": "Resultados",
        "results_empty": "Tu descripción / resultados aparecerán aquí.",

        "err_no_file": "Selecciona una imagen (o varias) para analizar.",
        "err_bad_format": "Formato no permitido (usa JPG/PNG/WEBP).",

        "badge_ok": "✅ La imagen pertenece a la categoría de comida",
        "badge_uncertain": "🤔 Resultado incierto",
        "badge_none": "🚫 La imagen no pertenece a la categoría de comida",

        "seen_text": "En la imagen se observa:",
        "final_none": "No encontrado / No pertenece",
        "final_uncertain": "Incierto (más probable: {best})",

        "info_title": "¿Cómo funciona?",
        "info_p1": "El modelo predice probabilidades para: lacteos, arroz, frutas/verduras.",
        "info_p2": "Si ninguna clase supera el umbral, se marca como No encontrado / No pertenece.",
        "info_p3": "Umbral: {thresh} • none_if_max_below: {none_if_max_below}",
    },
    "en": {
        "brand": "Food Detector",
        "nav_home": "Home",
        "nav_inspo": "Inspiration",
        "nav_tutorials": "Tutorials",
        "nav_tools": "Tools",
        "nav_pricing": "Pricing",
        "btn_feedback": "💬 Feedback",
        "btn_login": "Sign in",

        "hero_title": "Image Description Generator",
        "hero_sub": "Upload one (or multiple) images. Get class probabilities (dairy, rice, fruits/vegetables) and a final decision.",

        "tab_upload": "Upload image",
        "tab_info": "Info",
        "tab_models": "MLflow Models",  # ✅ Nueva pestaña

        "lang_label": "Result language",
        "lang_es": "Español",
        "lang_en": "English",

        "dz_title": "Upload a photo or drag & drop",
        "dz_hint": "PNG, JPG or WEBP • up to 12MB",
        "dz_pick": "Choose image",

        "mode_label": "Upload mode",
        "mode_single": "Single",
        "mode_multi": "Multiple",

        "btn_analyze": "✨ Analyze image",
        "formats_hint": "Formats: JPG, PNG, WEBP",

        "preview_title": "Image preview",
        "preview_empty": "Your image will appear here",

        "results_title": "Results",
        "results_empty": "Your description / results will appear here.",

        "err_no_file": "Select an image (or multiple images) to analyze.",
        "err_bad_format": "Format not allowed (use JPG/PNG/WEBP).",

        "badge_ok": "✅ The image belongs to the food category",
        "badge_uncertain": "🤔 Uncertain result",
        "badge_none": "🚫 The image does not belong to the food category",

        "seen_text": "In the image we see:",
        "final_none": "Not found / Not food",
        "final_uncertain": "Uncertain (most likely: {best})",

        "info_title": "How does it work?",
        "info_p1": "The model predicts probabilities for: lacteos, arroz, frutas/verduras.",
        "info_p2": "If no class passes the threshold, it is marked as Not found / Not food.",
        "info_p3": "Threshold: {thresh} • none_if_max_below: {none_if_max_below}",
    },
}

def tr(lang: str, key: str, **kwargs) -> str:
    lang = normalize_lang(lang)
    text = I18N.get(lang, I18N["es"]).get(key, key)
    try:
        return text.format(**kwargs)
    except Exception:
        return text

# ========= LOAD MODEL =========
print("✅ Cargando modelo desde:", MODEL_PATH)
print("✅ Existe modelo?:", os.path.exists(MODEL_PATH))
model_lock = Lock()
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado OK")

def predict_one(image_path: str):
    x = load_img_for_model(image_path)

    with model_lock:
        probs = model.predict(x, verbose=0)[0]  # (3,)

    probs_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
    labels = [c for c, p in zip(CLASSES, probs) if p >= THRESH]

    if not labels and float(np.max(probs)) < NONE_IF_MAX_BELOW:
        status = "none"
        final = "none"
    elif not labels:
        status = "uncertain"
        final = "uncertain"
    else:
        status = "ok"
        final = ", ".join(labels)

    return probs_dict, final, status

def reload_model():
    global model
    with model_lock:
        print("🔄 Recargando modelo desde:", MODEL_PATH)
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo recargado")

# ========= RETRAINER =========
retrainer = IncrementalRetrainer(
    model_path=MODEL_PATH,
    corrections_dir=os.path.join(PROJECT_ROOT, "corrections"),
    project_root=PROJECT_ROOT
)

# ✅ RESETEAR ESTADO AL INICIAR
try:
    retrainer._write_status(
        "idle",
        f"Esperando correcciones... ({retrainer.get_pending_count()}/{retrainer.MIN_CORRECTIONS})"
    )
except Exception as e:
    print("⚠️ No se pudo resetear retrain_status al iniciar:", e)

# ========= RUTAS =========
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None
    lang = "es"

    if request.method == "POST":
        lang = normalize_lang(request.form.get("lang", "es"))
        mode = request.form.get("mode", "single")  # single | multi

        if mode == "multi":
            files = request.files.getlist("images")
        else:
            f = request.files.get("file")
            files = [f] if f and f.filename else []

        files = [f for f in files if f is not None and f.filename]

        if not files:
            error = tr(lang, "err_no_file")
            return render_template(
                "index.html",
                results=results,
                error=error,
                classes=CLASSES,
                thresh=THRESH,
                none_if_max_below=NONE_IF_MAX_BELOW,
                lang=lang,
                ui=I18N[lang],
            )

        for f in files:
            if not allowed_file(f.filename):
                results.append({
                    "ok": False,
                    "filename": f.filename,
                    "error": tr(lang, "err_bad_format"),
                })
                continue

            safe = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{safe}"
            save_path = os.path.join(UPLOAD_DIR, unique)
            f.save(save_path)

            probs_dict, final_key, status = predict_one(save_path)

            if status == "ok":
                badge_text = tr(lang, "badge_ok")
                final_txt = final_key
            elif status == "uncertain":
                badge_text = tr(lang, "badge_uncertain")
                best = max(probs_dict, key=probs_dict.get)
                final_txt = tr(lang, "final_uncertain", best=best)
            else:
                badge_text = tr(lang, "badge_none")
                final_txt = tr(lang, "final_none")

            results.append({
                "ok": True,
                "filename": safe,
                "image_url": f"/static/uploads/{unique}",
                "probs": probs_dict,
                "final": final_txt,
                "status": status,
                "badge_text": badge_text,
                "seen_text": tr(lang, "seen_text"),
            })

    return render_template(
        "index.html",
        results=results,
        error=error,
        classes=CLASSES,
        thresh=THRESH,
        none_if_max_below=NONE_IF_MAX_BELOW,
        lang=lang,
        ui=I18N[normalize_lang(lang)],
    )

@app.route("/correct", methods=["POST"])
def correct_labels():
    """
    Guarda una corrección del usuario para reentrenamiento.
    Espera JSON: { image_url: "/static/uploads/xxx.jpg", correct_labels: ["lacteos", ...] }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_url = data.get("image_url")
        corrected_labels = data.get("correct_labels", [])

        if not image_url:
            return jsonify({"error": "URL de imagen requerida"}), 400

        # Validar formato y convertir a path real
        prefix = "/static/"
        if not str(image_url).startswith(prefix):
            return jsonify({"error": f"Formato de image_url inválido: {image_url}"}), 400

        # Ej: "/static/uploads/abc.png" -> "uploads/abc.png"
        relative = str(image_url)[len(prefix):]
        image_path = os.path.normpath(os.path.join(BASE_DIR, "static", relative))

        # Seguridad: evitar path traversal fuera de /static
        static_root = os.path.normpath(os.path.join(BASE_DIR, "static"))
        if not image_path.startswith(static_root):
            return jsonify({"error": "Ruta inválida (path traversal)"}), 400

        if not os.path.exists(image_path):
            return jsonify({"error": f"Imagen no encontrada: {image_path}"}), 404

        pending_count = retrainer.add_correction(image_path, corrected_labels)

        return jsonify({
            "status": "success",
            "message": "✅ Corrección guardada exitosamente",
            "pending_corrections": pending_count,
            "pending_count": retrainer.get_pending_count(),
            "ready_for_retrain": retrainer.should_retrain(),
            "threshold": retrainer.MIN_CORRECTIONS
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/trigger_retrain", methods=["POST"])
def trigger_retrain():
    """
    Inicia reentrenamiento en background para no bloquear la UI.
    """
    try:
        if not retrainer.should_retrain():
            pending = retrainer.get_pending_count()
            return jsonify({
                "error": f"Necesitas al menos {retrainer.MIN_CORRECTIONS} correcciones para reentrenar",
                "pending_corrections": pending,
                "pending_count": pending,
                "ready_for_retrain": False,
                "threshold": retrainer.MIN_CORRECTIONS
            }), 400

        def run_retraining():
            
            try:
                
                result = retrainer.incremental_finetune(epochs=3)
                print(f"✅ Reentrenamiento completado: {result}")

                # RECARGAR MODELO NUEVO EN FLASK
                reload_model()

            except Exception as e:
                print(f"❌ Error en reentrenamiento: {e}")
                # ✅ Guardar error usando el método del retrainer 
                try:
                    retrainer._write_status("error", f"❌ ERROR: {str(e)}")
                except Exception:
                    pass

        Thread(target=run_retraining, daemon=True).start()

        return jsonify({
            "status": "started",
            "message": "🔄 Reentrenamiento iniciado en background",
            "check_status_at": "/retrain_status"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/retrain_status", methods=["GET"])
def retrain_status():
    try:
        status = retrainer.get_retraining_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "pending_corrections": retrainer.get_pending_count(),
        "ready_for_retrain": retrainer.should_retrain(),
        "threshold": retrainer.MIN_CORRECTIONS
    })

# RUTA PARA VER MODELOS REGISTRADOS EN MLFLOW 
@app.route("/models")
def list_models():
    try:
        
        mlflow_db = os.path.join(PROJECT_ROOT, "mlflow.db")
        mlflow.set_tracking_uri("sqlite:///" + mlflow_db.replace("\\", "/"))

        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        
        model_list = []
        for model in models:
            versions = client.get_latest_versions(model.name)
            for version in versions:
                model_list.append({
                    "name": model.name,
                    "version": version.version,
                    "status": version.status,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp
                })
        
        return render_template('models.html', models=model_list)
    except Exception as e:
        return jsonify({"error": f"No se pudieron cargar modelos de MLflow: {str(e)}"}), 500

# ✅ REDIRECCIÓN A MLFLOW UI EXTERNA (conveniencia)
@app.route("/mlflow-ui")
def mlflow_redirect():
    return redirect("http://localhost:5000", code=302)  # MLflow UI corre en puerto 5000 por defecto


# ========= RUN =========
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("✅ SERVIDOR FLASK INICIADO")
    print("=" * 60)
    print("📊 Retrainer inicializado")
    print(f"   Correcciones pendientes: {retrainer.get_pending_count()}")
    print(f"   Umbral para reentrenar: {retrainer.MIN_CORRECTIONS}")
    print(f"   Listo para reentrenar: {retrainer.should_retrain()}")
    print("=" * 60)
    print("🌐 Tu app: http://localhost:5001")
    print("📊 MLflow UI: http://localhost:5000 (ejecuta: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
    print("=" * 60 + "\n")

    
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
