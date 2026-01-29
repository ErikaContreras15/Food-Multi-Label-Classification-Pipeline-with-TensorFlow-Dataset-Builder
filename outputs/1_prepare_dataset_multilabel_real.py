import os
import random
import pandas as pd
from PIL import Image

random.seed(42)

# =========================
# CONFIG
# =========================
RAW_FOOD11 = "raw/food11"
RAW_FOODORNOT = "raw/foodornot"

OUT_IMAGES = "dataset/images"
OUT_LABELS = "dataset/labels.csv"

# ✅ 3 clases (multilabel real)
CLASSES = ["lacteos", "arroz", "frutas/verduras"]

FOOD11_FOLDER_MAP = {
    "lacteos": "Dairy product",
    "arroz": "Rice",
    "frutas/verduras": "Vegetable-Fruit",
}

# cantidad de collages por split
N_TRAIN = 3000
N_VAL = 600
N_TEST = 600

# probabilidad de que cada clase aparezca en una imagen compuesta
P_INCLUDE = 0.60

# cantidad de NO comida por split
MAX_NONFOOD_TRAIN = 1200
MAX_NONFOOD_VAL = 200
MAX_NONFOOD_TEST = 200

IMG_SIZE = (224, 224)


# =========================
# HELPERS
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
    ]

def load_pool(split_name: str):
    split_dir = os.path.join(RAW_FOOD11, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"No existe: {split_dir}")

    pools = {}
    for cls in CLASSES:
        class_dir = os.path.join(split_dir, FOOD11_FOLDER_MAP[cls])
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No existe carpeta: {class_dir}")

        pools[cls] = list_images(class_dir)
        if len(pools[cls]) == 0:
            raise RuntimeError(f"Carpeta vacía: {class_dir}")

    return pools

def pick_labels():
    labs = []
    for _ in CLASSES:
        labs.append(1 if random.random() < P_INCLUDE else 0)

    # la mayoría de veces debe haber al menos 1 clase
    if sum(labs) == 0 and random.random() < 0.80:
        labs[random.randrange(len(CLASSES))] = 1

    return labs

def make_collage(selected_imgs):
    canvas = Image.new("RGB", IMG_SIZE, (0, 0, 0))

    # slots para hasta 3 imágenes
    slots = [
        (0, 0, 112, 112),
        (112, 0, 224, 112),
        (0, 112, 112, 224),
    ]

    for img, slot in zip(selected_imgs, slots):
        x1, y1, x2, y2 = slot
        w, h = x2 - x1, y2 - y1
        im = img.convert("RGB").resize((w, h))
        canvas.paste(im, (x1, y1))

    # si solo hay 1 clase, mejor usarla grande
    if len(selected_imgs) == 1:
        canvas = selected_imgs[0].convert("RGB").resize(IMG_SIZE)

    return canvas

def save_image(img, filename):
    img.save(os.path.join(OUT_IMAGES, filename), quality=95)

def collect_nonfood(split_key: str):
    paths = []

    train_non = os.path.join(RAW_FOODORNOT, "train", "negative_non_food")
    test_non = os.path.join(RAW_FOODORNOT, "test", "negative_non_food")

    if split_key == "train" and os.path.isdir(train_non):
        paths.extend(list_images(train_non))

    if os.path.isdir(test_non):
        paths.extend(list_images(test_non))

    random.shuffle(paths)
    return paths


def build_split(food11_split, n_samples, prefix, nonfood_limit, nonfood_split_key):
    pools = load_pool(food11_split)
    rows = []
    idx = 0

    # 1) collages multilabel
    for _ in range(n_samples):
        labs = pick_labels()

        chosen_paths = []
        for cls, flag in zip(CLASSES, labs):
            if flag == 1:
                chosen_paths.append(random.choice(pools[cls]))

        if len(chosen_paths) == 0:
            cls = random.choice(CLASSES)
            labs = [1 if c == cls else 0 for c in CLASSES]
            chosen_paths = [random.choice(pools[cls])]

        imgs = [Image.open(p) for p in chosen_paths]
        collage = make_collage(imgs)

        idx += 1
        fn = f"{prefix}_{idx:07d}.jpg"
        save_image(collage, fn)

        rows.append({
            "filename": fn,
            **{c: int(v) for c, v in zip(CLASSES, labs)},
            "source": f"collage/{food11_split}"
        })

    # 2) NO comida => ninguno (0,0,0)
    nonfoods = collect_nonfood(nonfood_split_key)[:nonfood_limit]

    for p in nonfoods:
        idx += 1
        fn = f"{prefix}_none_{idx:07d}.jpg"
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        save_image(img, fn)

        rows.append({
            "filename": fn,
            **{c: 0 for c in CLASSES},
            "source": "foodornot/negative_non_food"
        })

    return rows


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_IMAGES)

    # limpiar images viejas
    for f in os.listdir(OUT_IMAGES):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            os.remove(os.path.join(OUT_IMAGES, f))

    all_rows = []
    all_rows += build_split("training",   N_TRAIN, "train", MAX_NONFOOD_TRAIN, "train")
    all_rows += build_split("validation", N_VAL,   "val",   MAX_NONFOOD_VAL,   "val")
    all_rows += build_split("evaluation", N_TEST,  "test",  MAX_NONFOOD_TEST,  "test")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_LABELS, index=False, encoding="utf-8")

    print("\n✅ Dataset MULTILABEL REAL listo")
    print("Total:", len(df))
    print("Positivos por clase:\n", df[CLASSES].sum())
    print("NINGUNO:", int((df[CLASSES].sum(axis=1) == 0).sum()))
    print("CSV:", OUT_LABELS)


if __name__ == "__main__":
    main()
