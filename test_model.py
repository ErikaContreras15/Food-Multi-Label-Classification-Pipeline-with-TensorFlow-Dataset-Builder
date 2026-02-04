import tensorflow as tf
import numpy as np

MODEL_PATH = "outputs/model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["lacteos", "arroz", "frutas/verduras"]

THRESH = 0.50
NONE_IF_MAX_BELOW = 0.45

def load_img(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)

model = tf.keras.models.load_model(MODEL_PATH)


img_path = r"C:\Users\HP OMEN\Documents\GitHub\Food-Multi-Label-Classification-Pipeline-with-TensorFlow-Dataset-Builder\flask_app\static\uploads\2f296af610d84f5ba88b8d9039256672_pelotas2.png"

x = load_img(img_path)
probs = model.predict(x, verbose=0)[0]

probs_dict = {c: float(p) for c, p in zip(CLASSES, probs)}
mx = float(np.max(probs))
labels = [c for c, p in probs_dict.items() if p >= THRESH]

print("Pred:", probs_dict)
print("Max:", max(probs_dict, key=probs_dict.get), "=>", mx)

if not labels and mx < NONE_IF_MAX_BELOW:
    print("✅ FINAL:", "none (no pertenece)")
elif not labels:
    print("✅ FINAL:", "uncertain (inseguro)")
else:
    print("✅ FINAL:", labels)
