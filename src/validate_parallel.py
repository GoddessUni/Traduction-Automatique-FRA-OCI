from pathlib import Path
import fasttext
import matplotlib.pyplot as plt
import random
import csv

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "train_sentences"

FR_FILE = DATA_DIR / "train_sentences.fr"
OC_FILE = DATA_DIR / "train_sentences.oc"

FASTTEXT_MODEL = PROJECT_ROOT / "lid.176.ftz"

ROMANCE_LANGS = {"fr", "oc", "ca", "es", "it", "pt", "ro"}
HIGH_CONF = 0.80
LOW_CONF = 0.50

MIN_LEN = 2
MAX_LEN = 200
MAX_LEN_RATIO = 0.5

REPORT_DIR = DATA_DIR / "validation_report"
REPORT_DIR.mkdir(exist_ok=True)

with open(FR_FILE, encoding="utf-8") as f_fr, open(OC_FILE, encoding="utf-8") as f_oc:
    pairs = [(f.strip(), o.strip()) for f, o in zip(f_fr, f_oc)]

print(f"Total paires: {len(pairs)}")

# fastText
print("Chargement du modèle fastText...")
lang_model = fasttext.load_model(str(FASTTEXT_MODEL))

# Analyser
clean = []
ok_romance = []
uncertain = []
reject = []

for fr, oc in pairs:
    fr_len = len(fr.split())
    oc_len = len(oc.split())

    if fr_len < MIN_LEN or oc_len < MIN_LEN:
        reject.append((fr, oc, "too_short"))
        continue

    if fr_len > MAX_LEN or oc_len > MAX_LEN:
        uncertain.append((fr, oc, "too_long"))
        continue

    if abs(fr_len - oc_len) / max(fr_len, 1) > MAX_LEN_RATIO:
        uncertain.append((fr, oc, "length_mismatch"))
        continue

    # Detection de langue
    labels, probs = lang_model.predict(oc)
    lang = labels[0].replace("__label__", "")
    prob = probs[0]

    if lang not in ROMANCE_LANGS:
        reject.append((fr, oc, f"non_romance:{lang}"))
    elif prob >= HIGH_CONF:
        clean.append((fr, oc, lang, prob))
    elif prob >= LOW_CONF:
        ok_romance.append((fr, oc, lang, prob))
    else:
        uncertain.append((fr, oc, f"low_conf:{lang}"))

print(f"Paires clean: {len(clean)}")
print(f"Paires OK (romanes): {len(ok_romance)}")
print(f"Paires incertaines: {len(uncertain)}")
print(f"Paires rejetées: {len(reject)}")

# Rapport
def save_csv(name, rows, header):
    with open(REPORT_DIR / name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

save_csv("clean.csv", clean, ["fr", "oc", "lang", "prob"])
save_csv("ok_romance.csv", ok_romance, ["fr", "oc", "lang", "prob"])
save_csv("uncertain.csv", uncertain, ["fr", "oc", "reason"])
save_csv("reject.csv", reject, ["fr", "oc", "reason"])

print(f"\nRapports enregistrés dans {REPORT_DIR}")

