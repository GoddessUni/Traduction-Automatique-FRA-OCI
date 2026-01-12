from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "train_sentences" / "validation_report"
FINAL_DIR = PROJECT_ROOT / "final_corpus"
FINAL_DIR.mkdir(exist_ok=True)

CLEAN_FILE = REPORT_DIR / "clean.csv"
OK_FILE = REPORT_DIR / "ok_romance.csv"

FR_OUT = FINAL_DIR / "train.fr"
OC_OUT = FINAL_DIR / "train.oc"

def load_pairs(csv_path):
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            fr, oc = row[0].strip(), row[1].strip()
            if fr and oc:
                pairs.append((fr, oc))
    return pairs


print("Chargement des paires clean...")
clean_pairs = load_pairs(CLEAN_FILE)

print("Chargement des paires OK (romanes)...")
ok_pairs = load_pairs(OK_FILE)

all_pairs = clean_pairs + ok_pairs

# Dé-duplication finale (sécurité)
all_pairs = list(dict.fromkeys(all_pairs))

print(f"Total final de paires: {len(all_pairs)}")

# Écriture
with open(FR_OUT, "w", encoding="utf-8") as f_fr, \
     open(OC_OUT, "w", encoding="utf-8") as f_oc:
    for fr, oc in all_pairs:
        f_fr.write(fr + "\n")
        f_oc.write(oc + "\n")

print(f"Corpus final écrit dans: {FINAL_DIR}")
print(f" - {FR_OUT.name}")
print(f" - {OC_OUT.name}")