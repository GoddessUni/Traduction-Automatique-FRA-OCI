from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "final_corpus"
SPLIT_DIR = PROJECT_ROOT / "data_splits"
SPLIT_DIR.mkdir(exist_ok=True)

FR_FILE = FINAL_DIR / "train.fr"
OC_FILE = FINAL_DIR / "train.oc"

# Data split: 90% train / 5% validation / 5% test
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05
TEST_RATIO = 0.05

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-6

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

with open(FR_FILE, encoding="utf-8") as f_fr, \
     open(OC_FILE, encoding="utf-8") as f_oc:
    fr_lines = [l.strip() for l in f_fr]
    oc_lines = [l.strip() for l in f_oc]

assert len(fr_lines) == len(oc_lines), "FR / OC ne sont pas correspondants."

pairs = list(zip(fr_lines, oc_lines))
random.shuffle(pairs)

n_total = len(pairs)
n_train = int(n_total * TRAIN_RATIO)
n_valid = int(n_total * VALID_RATIO)

train_pairs = pairs[:n_train]
valid_pairs = pairs[n_train:n_train + n_valid]
test_pairs  = pairs[n_train + n_valid:]

def write_split(pairs, prefix):
    fr_path = SPLIT_DIR / f"{prefix}.fr"
    oc_path = SPLIT_DIR / f"{prefix}.oc"
    with open(fr_path, "w", encoding="utf-8") as f_fr, \
         open(oc_path, "w", encoding="utf-8") as f_oc:
        for fr, oc in pairs:
            f_fr.write(fr + "\n")
            f_oc.write(oc + "\n")

write_split(train_pairs, "train")
write_split(valid_pairs, "valid")
write_split(test_pairs,  "test")

print(f"Total: {n_total}")
print(f"Train: {len(train_pairs)}")
print(f"Valid: {len(valid_pairs)}")
print(f"Test : {len(test_pairs)}")
print(f"Données écrites dans: {SPLIT_DIR}")
