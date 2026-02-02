import os
import csv
import argparse

# Tu dataset usa estas clases
CLASSES = ["lacteos", "arroz", "frutas/verduras"]

# Extensiones válidas
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Columna extra que tienes en tu CSV
DEFAULT_SOURCE_VALUE = "manual"


def find_project_root(start="."):
    here = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(here, "dataset")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError("No encontré la carpeta 'dataset' hacia arriba. Abre el proyecto correcto en VS Code.")
        here = parent


def list_images(images_dir: str):
    files = []
    for name in os.listdir(images_dir):
        ext = os.path.splitext(name.lower())[1]
        if ext in IMG_EXTS and os.path.isfile(os.path.join(images_dir, name)):
            files.append(name)
    return sorted(files)


def read_existing_filenames(labels_csv: str):
    if not os.path.exists(labels_csv):
        return set(), None

    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        existing = set()

        # tu CSV usa "filename"
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if fn:
                existing.add(fn)

    return existing, header


def write_pending(pending_csv: str, missing_files: list, source_value: str):
    os.makedirs(os.path.dirname(pending_csv), exist_ok=True)

    header = ["filename"] + CLASSES + ["source"]

    with open(pending_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for fn in missing_files:
            writer.writerow([fn, 0, 0, 0, source_value])


def append_to_labels(labels_csv: str, missing_files: list, source_value: str):
    file_exists = os.path.exists(labels_csv)

    if not file_exists:
        # Crear el archivo con header correcto
        with open(labels_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename"] + CLASSES + ["source"])

    # Append filas nuevas
    with open(labels_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for fn in missing_files:
            writer.writerow([fn, 0, 0, 0, source_value])


def main():
    parser = argparse.ArgumentParser(
        description="Genera labels_pending.csv con imágenes nuevas que no están en labels.csv"
    )
    parser.add_argument("--labels_csv", default="dataset/labels.csv")
    parser.add_argument("--images_dir", default="dataset/images")
    parser.add_argument("--pending_csv", default="dataset/labels_pending.csv")
    parser.add_argument("--merge", action="store_true",
                        help="Añade filas nuevas directamente a labels.csv (con 0s).")
    parser.add_argument("--source", default=DEFAULT_SOURCE_VALUE,
                        help="Valor para la columna 'source' de nuevas filas. Ej: manual, real, camera, etc.")
    args = parser.parse_args()

    root = find_project_root(".")
    os.chdir(root)

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"No existe la carpeta: {args.images_dir}")

    all_imgs = list_images(args.images_dir)
    existing, header = read_existing_filenames(args.labels_csv)

    missing = [fn for fn in all_imgs if fn not in existing]

    print("✅ Proyecto:", root)
    print("✅ Total imágenes en dataset/images:", len(all_imgs))
    print("✅ Filas existentes en labels.csv:", len(existing))
    print("🆕 Imágenes nuevas (no están en labels.csv):", len(missing))

    if len(missing) == 0:
        print("🎉 No hay nada que agregar. Todo está sincronizado.")
        return

    write_pending(args.pending_csv, missing, args.source)
    print(f"📝 Generado: {args.pending_csv}")
    print("➡️ Edita labels_pending.csv y cambia 0/1 según corresponda.")
    print("   - Si NO es comida: deja lacteos=0, arroz=0, frutas/verduras=0")

    if args.merge:
        append_to_labels(args.labels_csv, missing, args.source)
        print(f"➕ Añadidas {len(missing)} filas nuevas a: {args.labels_csv}")
        print("⚠️ OJO: quedaron en 0s. Debes editar labels.csv para poner las etiquetas correctas.")


if __name__ == "__main__":
    main()
