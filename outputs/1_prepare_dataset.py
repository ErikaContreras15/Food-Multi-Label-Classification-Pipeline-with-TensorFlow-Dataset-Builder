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

# ✅ 5 clases finales (tus clases)
CLASSES = ["bread", "dairy", "dessert", "rice", "vegfruit"]

# ✅ Mapeo EXACTO a tus carpetas
FOOD11_FOLDER_MAP = {
    "bread": "Bread",
    "dairy": "Dairy product",
    "dessert": "Dessert",
    "rice": "Rice",
    "vegfruit": "Vegetable-Fruit"
}

MAX_PER_SPLIT_PER_CLASS = 1200
MAX_NONFOOD_TOTAL = 3000

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
    paths = []
    train_nonfood = os.path.join(RAW_FOODORNOT, "train", "negative_non_food")
    test_nonfood  = os.path.join(RAW_FOODORNOT, "test", "negative_non_food")

    if os.path.isdir(train_nonfood):
        paths.extend(list_images(train_nonfood))
    if os.path.isdir(test_nonfood):
        paths.extend(list_images(test_nonfood))

    if len(paths) == 0:
        raise FileNotFoundError("No encontré negative_non_food en foodornot/train ni foodornot/test")

    if MAX_NONFOOD_TOTAL:
        paths = paths[:MAX_NONFOOD_TOTAL]

    return paths

def main():
    ensure_dir(OUT_IMAGES)

    rows = []
    idx = 0

    # 1) Food-11 (5 clases)
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

    # 2) No comida => ninguno (0,0,0,0,0)
    nonfoods = foodornot_collect_nonfood()
    for path in nonfoods:
        idx += 1
        fn = copy_with_newname(path, "notfood", idx)

        label = {k: 0 for k in CLASSES}

        rows.append({
            "filename": fn,
            **label,
            "source": "foodornot/negative_non_food"
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_LABELS, index=False, encoding="utf-8")

    print("\n✅ Dataset combinado listo (5 clases)")
    print("Total imágenes:", len(df))
    print("Conteo por clase:")
    print(df[CLASSES].sum())
    print("NINGUNO:", int((df[CLASSES].sum(axis=1) == 0).sum()))
    print("CSV:", OUT_LABELS)

if __name__ == "__main__":
    main()
