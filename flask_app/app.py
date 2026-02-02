import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf

# ========= CONFIG =========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "model.keras")
IMG_SIZE = (224, 224)

CLASSES = ["lacteos", "arroz", "frutas/verduras"]

THRESH = 0.50
NONE_IF_MAX_BELOW = 0.45

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Crear carpeta uploads (y detectar si "uploads" es archivo)
if os.path.exists(UPLOAD_DIR) and not os.path.isdir(UPLOAD_DIR):
    raise RuntimeError(f"❌ '{UPLOAD_DIR}' existe pero NO es carpeta. Borra ese archivo y crea una carpeta.")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB

print("✅ Cargando modelo desde:", MODEL_PATH)
print("✅ Existe modelo?:", os.path.exists(MODEL_PATH))
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado OK")


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

def normalize_lang(lang: str) -> str:
    lang = (lang or "es").lower().strip()
    return "en" if lang.startswith("en") else "es"

def tr(lang: str, key: str, **kwargs) -> str:
    lang = normalize_lang(lang)
    text = I18N[lang].get(key, key)
    try:
        return text.format(**kwargs)
    except Exception:
        return text


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTS


def load_img_for_model(path: str):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)


def predict_one(image_path: str):
    x = load_img_for_model(image_path)
    probs = model.predict(x, verbose=0)[0]  # (3,)

    probs_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
    labels = [c for c, p in zip(CLASSES, probs) if p >= THRESH]

    if not labels and float(np.max(probs)) < NONE_IF_MAX_BELOW:
        final = "none"
        status = "none"
    elif not labels:
        final = "uncertain"
        status = "uncertain"
    else:
        final = ", ".join(labels)
        status = "ok"

    return probs_dict, final, status


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    # idioma por defecto
    lang = "es"

    if request.method == "POST":
        lang = normalize_lang(request.form.get("lang", "es"))
        mode = request.form.get("mode", "single")  # single | multi

        files = []
        if mode == "multi":
            files = request.files.getlist("images")
        else:
            f = request.files.get("file")
            files = [f] if f and f.filename else []

        files = [f for f in files if f is not None and f.filename != ""]

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

            # textos por idioma para badge/final
            if status == "ok":
                badge_text = tr(lang, "badge_ok")
                final_txt = final_key  # aquí ya es "lacteos, arroz..."
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


if __name__ == "__main__":
    app.run(debug=True)
