# retrain_service.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf


class IncrementalRetrainer:
    def __init__(
        self,
        model_path="outputs/model.keras",
        corrections_dir="corrections",
        project_root=None,
        dataset_labels_rel=os.path.join("dataset", "labels.csv"),
        dataset_images_rel=os.path.join("dataset", "images"),
    ):
        # paths ABS
        self.model_path = os.path.abspath(model_path)
        self.corrections_dir = os.path.abspath(corrections_dir)

        # root del proyecto (para encontrar dataset/labels.csv siempre)
        if project_root:
            self.project_root = os.path.abspath(project_root)
        else:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.dataset_labels_path = os.path.join(self.project_root, dataset_labels_rel)
        self.dataset_images_dir = os.path.join(self.project_root, dataset_images_rel)

        self.CLASSES = ["lacteos", "arroz", "frutas/verduras"]
        self.MIN_CORRECTIONS = 1  # testing (sube a 10-20 cuando ya esté listo)
        self.IMG_SIZE = (224, 224)
        self.BATCH = 32

        # dirs
        os.makedirs(self.corrections_dir, exist_ok=True)
        os.makedirs(os.path.join(self.corrections_dir, "history"), exist_ok=True)

        # pending.csv
        self.pending_file = os.path.join(self.corrections_dir, "pending.csv")
        if not os.path.exists(self.pending_file):
            pd.DataFrame(
                columns=["timestamp", "image_path", "lacteos", "arroz", "frutas/verduras", "status"]
            ).to_csv(self.pending_file, index=False)

        # status file
        self.status_file = os.path.join(self.corrections_dir, "retraining_status.txt")

    # =========================
    # Utilidad estado
    # =========================
    def _write_status(self, status: str, message: str = ""):
        """
        status: idle | in_progress | completed | error
        Formato:
            STATUS:<status>
            <message>
        """
        os.makedirs(self.corrections_dir, exist_ok=True)
        with open(self.status_file, "w", encoding="utf-8") as f:
            f.write(f"STATUS:{status}\n")
            if message:
                f.write(message.strip() + "\n")

    # =========================
    # Correcciones
    # =========================
    def add_correction(self, image_path: str, corrected_labels: list):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

        corrected_labels = corrected_labels or []

        # Soporte "none" (no pertenece): todo 0
        if "none" in corrected_labels:
            row = {
                "lacteos": 0,
                "arroz": 0,
                "frutas/verduras": 0,
                "status": "pending_none",
            }
        else:
            row = {
                "lacteos": int("lacteos" in corrected_labels),
                "arroz": int("arroz" in corrected_labels),
                "frutas/verduras": int("frutas/verduras" in corrected_labels),
                "status": "pending",
            }

        df_new = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "image_path": os.path.abspath(image_path),
            **row
        }])

        if os.path.exists(self.pending_file) and os.path.getsize(self.pending_file) > 0:
            try:
                existing = pd.read_csv(self.pending_file)
            except Exception:
                existing = pd.DataFrame(columns=df_new.columns)
            df_all = pd.concat([existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_csv(self.pending_file, index=False)
        return len(df_all)

    def get_pending_count(self) -> int:
        if not os.path.exists(self.pending_file) or os.path.getsize(self.pending_file) == 0:
            return 0
        try:
            return len(pd.read_csv(self.pending_file))
        except Exception:
            return 0

    def should_retrain(self) -> bool:
        return self.get_pending_count() >= self.MIN_CORRECTIONS

    # =========================
    # Dataset tf.data (seguro)
    # =========================
    def _make_incremental_ds(self, df: pd.DataFrame, training: bool):
        df = df.copy()
        df = df[df["image_path"].notna() & (df["image_path"] != "")]
        if len(df) == 0:
            raise RuntimeError("No hay ejemplos válidos para entrenar (image_path vacío).")

        paths = df["image_path"].astype(str).values
        y = df[self.CLASSES].values.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((paths, y))

        def _load(p, label):
            img_bytes = tf.io.read_file(p)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, self.IMG_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            return img, label

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

        # ✅ forma nueva (sin deprecated)
        ds = ds.ignore_errors()

        if training:
            ds = ds.shuffle(1000, seed=42, reshuffle_each_iteration=True)

        ds = ds.batch(self.BATCH).prefetch(tf.data.AUTOTUNE)
        return ds

    # =========================
    # Entrenamiento incremental
    # =========================
    def incremental_finetune(self, epochs=3, boost_factor=25):
        """
        Reentrena con:
        - sample del dataset original
        - correcciones (con boost)
        - LR muy bajo (1e-5)
        """

        if not self.should_retrain():
            raise ValueError(
                f"Necesitas al menos {self.MIN_CORRECTIONS} correcciones. "
                f"Actualmente tienes: {self.get_pending_count()}"
            )

        self._write_status(
            "in_progress",
            f"🔄 Iniciando reentrenamiento...\nCorrecciones: {self.get_pending_count()}/{self.MIN_CORRECTIONS}"
        )

        print("🚀 INICIANDO REENTRENAMIENTO INCREMENTAL...")
        print("📊 Correcciones pendientes:", self.get_pending_count())
        print("📁 labels.csv:", self.dataset_labels_path)

        # 1) Dataset original
        if not os.path.exists(self.dataset_labels_path):
            msg = f"❌ No existe labels.csv en:\n{self.dataset_labels_path}"
            self._write_status("error", msg)
            raise RuntimeError(msg)

        try:
            original_df = pd.read_csv(self.dataset_labels_path)
        except Exception as e:
            msg = f"❌ Error leyendo labels.csv:\n{e}"
            self._write_status("error", msg)
            raise RuntimeError(f"Error cargando dataset original: {e}")

        # validar columnas
        for c in ["filename"] + self.CLASSES:
            if c not in original_df.columns:
                msg = f"❌ labels.csv no tiene columna requerida: {c}"
                self._write_status("error", msg)
                raise RuntimeError(msg)

        original_df["image_path"] = original_df["filename"].apply(
            lambda f: os.path.join(self.dataset_images_dir, str(f))
        )

        # 2) Correcciones
        try:
            corrections_df = pd.read_csv(self.pending_file)
        except Exception as e:
            msg = f"❌ Error leyendo pending.csv:\n{e}"
            self._write_status("error", msg)
            raise RuntimeError(f"Error cargando correcciones: {e}")

        corrections_df = corrections_df[
            corrections_df["image_path"].notna() & (corrections_df["image_path"] != "")
        ].copy()

        for c in self.CLASSES:
            if c not in corrections_df.columns:
                corrections_df[c] = 0

        # ✅ Verificar existencia real de imágenes corregidas
        missing = corrections_df[~corrections_df["image_path"].apply(os.path.exists)]
        if len(missing) > 0:
            msg = (
                "❌ Hay correcciones apuntando a imágenes que no existen.\n"
                "Ejemplos:\n" + "\n".join(missing["image_path"].head(10).tolist())
            )
            self._write_status("error", msg)
            raise RuntimeError(msg)

        print("✅ Todas las imágenes de corrección existen y se usarán.")

        # 3) Mezcla + BOOST
        sample_size = int(len(corrections_df) * 2.33)  # original ~70%
        original_sample = original_df.sample(
            n=min(sample_size, len(original_df)),
            random_state=42
        )

        corrections_boost = pd.concat([corrections_df] * int(boost_factor), ignore_index=True)
        mixed_df = pd.concat([original_sample, corrections_boost], ignore_index=True)

        print("📈 Dataset mezclado:", len(mixed_df))
        print("   - Original:", len(original_sample))
        print("   - Correcciones:", len(corrections_df), f"(boost x{boost_factor} => {len(corrections_boost)})")

        train_ds = self._make_incremental_ds(mixed_df, training=True)

        # ✅ evitar "input ran out of data"
        train_ds = train_ds.repeat()
        steps_per_epoch = max(1, int(np.ceil(len(mixed_df) / self.BATCH)))

        # 4) Cargar modelo
        try:
            model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            msg = f"❌ Error cargando modelo:\n{e}"
            self._write_status("error", msg)
            raise RuntimeError(f"Error cargando modelo: {e}")

        model = self._unfreeze_conservative(model, n_layers=15)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="binary_crossentropy",
            metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        # 5) Callback status
        class EpochTracker(tf.keras.callbacks.Callback):
            def __init__(self, outer, total_epochs):
                self.outer = outer
                self.total_epochs = total_epochs

            def on_epoch_begin(self, epoch, logs=None):
                self.outer._write_status(
                    "in_progress",
                    f"🔄 Reentrenando...\nÉpoca {epoch+1}/{self.total_epochs}"
                )
                print(f"  ⏳ Época {epoch+1}/{self.total_epochs}...")

        callbacks = [
            EpochTracker(self, epochs),
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
        ]

        # 6) Entrenar
        try:
            history = model.fit(
                train_ds,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e:
            msg = f"❌ Error durante reentrenamiento:\n{e}"
            self._write_status("error", msg)
            raise RuntimeError(f"Error durante el reentrenamiento: {e}")

        # 7) Guardar backup + nuevo modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.model_path.replace(".keras", f"_backup_{timestamp}.keras")

        try:
            if os.path.exists(self.model_path):
                os.rename(self.model_path, backup_path)
            model.save(self.model_path)
        except Exception as e:
            msg = f"❌ Error guardando modelo:\n{e}"
            self._write_status("error", msg)
            raise RuntimeError(f"Error guardando modelo: {e}")

        # 8) Guardar history
        try:
            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(
                os.path.join(self.corrections_dir, "history", f"retrain_{timestamp}.csv"),
                index=False
            )
        except Exception:
            pass

        # 9) Limpiar pending
        pd.DataFrame(
            columns=["timestamp", "image_path", "lacteos", "arroz", "frutas/verduras", "status"]
        ).to_csv(self.pending_file, index=False)

        final_acc = float(history.history.get("binary_accuracy", [0])[-1])
        msg = (
            "✅ MODELO REENTRENADO EXITOSAMENTE\n"
            f"Backup: {os.path.basename(backup_path)}\n"
            f"Correcciones usadas: {len(corrections_df)}\n"
            f"Épocas: {epochs}\n"
            f"Precisión final: {final_acc:.2%}"
        )
        self._write_status("completed", msg)

        print("✅ REENTRENAMIENTO COMPLETADO")
        print("   Backup:", backup_path)
        print("   Precisión final:", f"{final_acc:.2%}")

        return {
            "status": "success",
            "epochs": history.history,
            "backup_path": backup_path,
            "corrections_used": int(len(corrections_df)),
            "final_accuracy": final_acc,
        }

    def _unfreeze_conservative(self, model, n_layers=15):
        """Descongela SOLO las últimas n capas del backbone si existe como submodelo."""
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break

        if backbone is None:
            print("⚠️ No encontré el backbone como submodelo. (igual entrenará head)")
            return model

        backbone.trainable = True
        for layer in backbone.layers[:-n_layers]:
            layer.trainable = False
        return model

    def get_retraining_status(self) -> dict:
        # si no existe status_file -> idle
        if not os.path.exists(self.status_file):
            pending = self.get_pending_count()
            return {
                "status": "idle",
                "message": f"Esperando correcciones... ({pending}/{self.MIN_CORRECTIONS})",
                "pending_count": pending,
                "threshold": self.MIN_CORRECTIONS,
                "ready_for_retrain": self.should_retrain(),
            }

        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error leyendo estado: {e}",
                "pending_count": self.get_pending_count(),
                "threshold": self.MIN_CORRECTIONS,
                "ready_for_retrain": self.should_retrain(),
            }

        status = "completed"
        message = ""

        if lines:
            if lines[0].startswith("STATUS:"):
                status = lines[0].split("STATUS:", 1)[1].strip() or "completed"
                message = "\n".join(lines[1:]).strip()
            else:
                # compatibilidad con formato viejo
                content = "\n".join(lines)
                if "REENTRENÁNDOSE" in content or "Época" in content or "Reentrenando" in content:
                    status = "in_progress"
                message = content

        return {
            "status": status,
            "message": message,
            "pending_count": self.get_pending_count(),
            "threshold": self.MIN_CORRECTIONS,
            "ready_for_retrain": self.should_retrain(),
        }
