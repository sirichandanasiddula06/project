import os
import shutil
import random

# ===== SOURCE PATHS (your original data) =====
TUMOR_SRC = r"D:\p1\TUM"
NORMAL_SRC = r"D:\p2\NORM"

# ===== DESTINATION PATH =====
BASE_DIR = r"D:\histopathology_project\dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def split_class(class_name, src_path):
    files = os.listdir(src_path)
    random.shuffle(files)

    total = len(files)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_list in split_files.items():
        dest_dir = os.path.join(BASE_DIR, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        for file in split_list:
            shutil.copy(
                os.path.join(src_path, file),
                os.path.join(dest_dir, file)
            )

    print(f"{class_name} split completed")

# ===== RUN =====
split_class("cancer", TUMOR_SRC)
split_class("normal", NORMAL_SRC)

print("✅ Dataset splitting completed successfully!")