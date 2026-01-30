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
        final = "No encontrado / No pertenece"
        status = "none"
    elif not labels:
        best = CLASSES[int(np.argmax(probs))]
        final = f"Incierto (más probable: {best})"
        status = "uncertain"
    else:
        final = ", ".join(labels)
        status = "ok"

    return probs_dict, final, status


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    if request.method == "POST":
        mode = request.form.get("mode", "one")  # one | many

        files = []
        if mode == "many":
            files = request.files.getlist("images")
        else:
            f = request.files.get("image")
            files = [f] if f else []

        # limpiar lista de None
        files = [f for f in files if f is not None and f.filename != ""]

        if not files:
            error = "Selecciona una imagen (o varias) para analizar."
            return render_template("index.html", results=results, error=error, classes=CLASSES)

        for f in files:
            if not allowed_file(f.filename):
                results.append({
                    "ok": False,
                    "filename": f.filename,
                    "error": "Formato no permitido (usa JPG/PNG/WEBP)."
                })
                continue

            safe = secure_filename(f.filename)
            unique = f"{uuid.uuid4().hex}_{safe}"
            save_path = os.path.join(UPLOAD_DIR, unique)
            f.save(save_path)

            probs_dict, final, status = predict_one(save_path)

            results.append({
                "ok": True,
                "filename": safe,
                "image_url": f"/static/uploads/{unique}",
                "probs": probs_dict,
                "final": final,
                "status": status
            })

    return render_template(
        "index.html",
        results=results,
        error=error,
        classes=CLASSES,
        thresh=THRESH,
        none_if_max_below=NONE_IF_MAX_BELOW
    )


if __name__ == "__main__":
    app.run(debug=True)
