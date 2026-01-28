import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 20  # puedes dejar 10, pero 20 + EarlyStopping va bien

LABELS_CSV = "dataset/labels.csv"
IMAGES_DIR = "dataset/images"
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "model.keras")

# ✅ Debe coincidir con labels.csv
CLASSES = ["leche", "arroz", "fruta"]

# ✅ Peso extra para ejemplos "NINGUNO" (no comida)
NEG_WEIGHT = 4.0   # prueba 3.0, 4.0, 6.0 si quieres más castigo
POS_WEIGHT = 1.0   # peso normal para comida


def make_ds(df, training):
    paths = df["path"].values.astype(str)
    y = df[CLASSES].values.astype(np.float32)

    # Peso por muestra: si es "ninguno" => más peso
    # ninguno = fila con suma 0 (leche=0, arroz=0, fruta=0)
    is_none = (y.sum(axis=1) == 0).astype(np.float32)
    sample_w = np.where(is_none == 1.0, NEG_WEIGHT, POS_WEIGHT).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y, sample_w))

    def _load(p, label, w):
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label, w

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1500)

    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x_in = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(x_in, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Multilabel => sigmoid
    out = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(LABELS_CSV)
    df["path"] = df["filename"].apply(lambda f: os.path.join(IMAGES_DIR, f))

    # Verificación de columnas
    missing_cols = [c for c in CLASSES if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas en labels.csv: {missing_cols}. Columnas actuales: {list(df.columns)}")

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Solo para info
    train_none = int((train_df[CLASSES].sum(axis=1) == 0).sum())
    val_none = int((val_df[CLASSES].sum(axis=1) == 0).sum())
    print(f"\n📌 Train: {len(train_df)} muestras | NINGUNO: {train_none}")
    print(f"📌 Val  : {len(val_df)} muestras | NINGUNO: {val_none}")
    print(f"📌 Pesos: NINGUNO={NEG_WEIGHT} | comida={POS_WEIGHT}\n")

    train_ds = make_ds(train_df, True)
    val_ds = make_ds(val_df, False)

    model = build_model()
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]
    )

    print("\n✅ Modelo guardado en:", MODEL_PATH)


if __name__ == "__main__":
    main()
