import os
import shutil
import pandas as pd

# =========================
# CONFIG
# =========================

RAW_FOOD11 = "raw/food11"
RAW_FOODORNOT = "raw/foodornot"

OUT_IMAGES = "dataset/images"
OUT_LABELS = "dataset/labels.csv"

# 3 clases que usarás (nombres bonitos para tu informe)
CLASSES = ["leche", "arroz", "fruta"]

# Mapeo EXACTO según tus carpetas de Food-11:
FOOD11_FOLDER_MAP = {
    "leche": "Dairy product",
    "arroz": "Rice",
    "fruta": "Vegetable-Fruit"
}

# límites (None = sin límite)
MAX_PER_SPLIT_PER_CLASS = 1200   # máximo por clase por split en Food-11
MAX_NONFOOD_TOTAL = 2000         # total NO comida a copiar (train+test)

# =========================
# HELPERS
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    ]

def copy_with_newname(src: str, prefix: str, idx: int):
    ext = os.path.splitext(src)[1].lower()
    if ext == "":
        ext = ".jpg"
    newname = f"{prefix}_{idx:07d}{ext}"
    dst = os.path.join(OUT_IMAGES, newname)
    shutil.copy2(src, dst)
    return newname

def food11_collect(split_name: str):
    """
    Food-11 en tu caso viene por carpetas con nombres.
    Devuelve lista (ruta_imagen, clase).
    """
    split_dir = os.path.join(RAW_FOOD11, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"No existe: {split_dir}")

    items = []

    for cls in CLASSES:
        folder_name = FOOD11_FOLDER_MAP[cls]
        class_dir = os.path.join(split_dir, folder_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No existe carpeta de clase: {class_dir}")

        imgs = list_images(class_dir)
        if MAX_PER_SPLIT_PER_CLASS:
            imgs = imgs[:MAX_PER_SPLIT_PER_CLASS]

        for img in imgs:
            items.append((img, cls))

    return items

def foodornot_collect_nonfood():
    """
    Recoge NO comida desde:
      - raw/foodornot/train/negative_non_food (si existe)
      - raw/foodornot/test/negative_non_food  (tu caso seguro existe)
    """
    paths = []

    train_nonfood = os.path.join(RAW_FOODORNOT, "train", "negative_non_food")
    test_nonfood  = os.path.join(RAW_FOODORNOT, "test", "negative_non_food")

    if os.path.isdir(train_nonfood):
        paths.extend(list_images(train_nonfood))

    if os.path.isdir(test_nonfood):
        paths.extend(list_images(test_nonfood))

    if len(paths) == 0:
        raise FileNotFoundError(
            "No encontré 'negative_non_food' en raw/foodornot/train ni en raw/foodornot/test.\n"
            "Revisa que exista: raw/foodornot/test/negative_non_food"
        )

    if MAX_NONFOOD_TOTAL:
        paths = paths[:MAX_NONFOOD_TOTAL]

    return paths

# =========================
# MAIN
# =========================

def main():
    ensure_dir(OUT_IMAGES)

    rows = []
    idx = 0

    # ========= 1) Food-11: 3 clases comida =========
    for split in ["training", "validation", "evaluation"]:
        pairs = food11_collect(split)

        for path, cls in pairs:
            idx += 1
            fn = copy_with_newname(path, f"food11_{split}", idx)

            label = {k: 0 for k in CLASSES}
            label[cls] = 1

            rows.append({
                "filename": fn,
                **label,
                "source": f"food11/{split}/{FOOD11_FOLDER_MAP[cls]}"
            })

    # ========= 2) Food-or-Not: NO comida => “ninguno” (0,0,0) =========
    nonfoods = foodornot_collect_nonfood()

    for path in nonfoods:
        idx += 1
        fn = copy_with_newname(path, "notfood", idx)

        label = {k: 0 for k in CLASSES}  # [0,0,0] => ninguno

        rows.append({
            "filename": fn,
            **label,
            "source": "foodornot/negative_non_food"
        })

    # Guardar CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_LABELS, index=False, encoding="utf-8")

    # Reporte
    print("\n✅ Dataset combinado listo")
    print("Ruta imágenes:", OUT_IMAGES)
    print("CSV:", OUT_LABELS)
    print("Total imágenes:", len(df))

    print("\nConteo positivos por clase:")
    print(df[CLASSES].sum())

    print("\nSin producto (000):", int((df[CLASSES].sum(axis=1) == 0).sum()))

if __name__ == "__main__":
    main()
