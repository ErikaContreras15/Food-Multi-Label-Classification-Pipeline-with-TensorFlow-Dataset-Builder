import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.tensorflow

# ======================
# CONFIG
# ======================
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

LABELS_CSV = "dataset/labels.csv"
IMAGES_DIR = "dataset/images"
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "model.keras")

# ✅ multilabel real (3 clases)
CLASSES = ["lacteos", "arroz", "frutas/verduras"]

# ✅ castigo para falsos positivos en "NINGUNO"
NEG_WEIGHT = 4.0
POS_WEIGHT = 1.0

# ======================
def make_ds(df, training: bool):
    paths = df["path"].values.astype(str)
    y = df[CLASSES].values.astype(np.float32)

    # sample_weight: si es ninguno (000) => más peso
    is_none = (y.sum(axis=1) == 0).astype(np.float32)
    sw = np.where(is_none == 1.0, NEG_WEIGHT, POS_WEIGHT).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y, sw))

    def _load(p, label, w):
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label, w

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000, seed=SEED)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x_in = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(x_in, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    return model


def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def unfreeze_last_layers(model, n_layers=30):
    """
    Descongela las últimas n_layers del backbone MobileNetV2.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        print("⚠️ No encontré el backbone como submodelo.")
        return model

    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    return model


class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            try:
                mlflow.log_metric(k, float(v), step=epoch)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["head", "finetune"], default="head")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--unfreeze", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(LABELS_CSV)
    df["path"] = df["filename"].apply(lambda f: os.path.join(IMAGES_DIR, f))

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)

    train_ds = make_ds(train_df, True)
    val_ds = make_ds(val_df, False)

    # MLflow
    mlflow.set_experiment("multilabel_real_leche_arroz_fruta")

    run_name = f"{args.stage}_epochs{args.epochs}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "stage": args.stage,
            "epochs": args.epochs,
            "batch": BATCH,
            "img_size": IMG_SIZE[0],
            "classes": ",".join(CLASSES),
            "neg_weight": NEG_WEIGHT,
            "pos_weight": POS_WEIGHT,
        })

        # construir/continuar modelo
        if args.stage == "head":
            lr = args.lr if args.lr is not None else 1e-3
            model = build_model()
            model = compile_model(model, lr)
            mlflow.log_param("lr", lr)

        else:
            if not os.path.isfile(MODEL_PATH):
                raise FileNotFoundError(f"No existe {MODEL_PATH}. Primero corre stage=head.")

            lr = args.lr if args.lr is not None else 1e-4
            model = tf.keras.models.load_model(MODEL_PATH)
            model = unfreeze_last_layers(model, n_layers=args.unfreeze)
            model = compile_model(model, lr)

            mlflow.log_param("lr", lr)
            mlflow.log_param("unfreeze_layers", args.unfreeze)

        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
            MLflowMetricsCallback(),
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # artefacto: modelo guardado
        mlflow.log_artifact(MODEL_PATH)

        print("\n✅ Modelo guardado en:", MODEL_PATH)


if __name__ == "__main__":
    main()
