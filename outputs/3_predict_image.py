import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "outputs/model.keras"
IMG_SIZE = (224, 224)

# ✅ Debe ser igual que en labels.csv y 2_train_multilabel.py
CLASSES = ["leche", "arroz", "fruta"]

# Umbrales
THRESH = 0.50            # umbral por clase
NONE_IF_MAX_BELOW = 0.45 # si ninguna pasa THRESH y max<esto => "NINGUNO"


def load_img(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)


def main():
    if len(sys.argv) < 2:
        print("Uso: python outputs/3_predict_image.py ruta/imagen.jpg")
        return

    image_path = sys.argv[1]

    model = tf.keras.models.load_model(MODEL_PATH)

    x = load_img(image_path)
    probs = model.predict(x)[0]

    print("\n📌 Imagen:", image_path)
    print("\nProbabilidades:")
    for c, p in zip(CLASSES, probs):
        print(f" - {c}: {p*100:.2f}%")

    labels = [c for c, p in zip(CLASSES, probs) if p >= THRESH]

    if not labels and float(np.max(probs)) < NONE_IF_MAX_BELOW:
        print("\n✅ Resultado FINAL: NINGUNO (no hay producto)")
    elif not labels:
        best = CLASSES[int(np.argmax(probs))]
        print(f"\n⚠️ Resultado FINAL: INCIERTO (más probable: {best})")
    else:
        print("\n✅ Resultado FINAL:", ", ".join(labels))


if __name__ == "__main__":
    main()
