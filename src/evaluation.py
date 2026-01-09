#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Configuration
OUTPUT_DIR = "/data/projets_m2_2526/dcheng/outputs/run_4"  # L'entraînement actuel
MAX_TARGET_LENGTH = 128
TEST_SAMPLE_NUM = 100

# Trouver le checkpoint le plus récent
latest_checkpoint = None
checkpoint_dirs = [
    os.path.join(OUTPUT_DIR, d) 
    for d in os.listdir(OUTPUT_DIR) 
    if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and "checkpoint" in d
]
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    print("Le checkpoint plus récent:", latest_checkpoint)
else:
    print("Checkpoint n'a pas été trouvé，essayez final_model ou mid_training_model")
    latest_checkpoint = os.path.join(OUTPUT_DIR, "final_model")

# Modèle et tokenizer
device = 0 if torch.cuda.is_available() else -1
model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

# Pipeline avec les langues
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    src_lang="fra_Latn",
    tgt_lang="oci_Latn"
)

# Corpus de test
TEST_SRC = "../data_splits/test.bidir.src"
TEST_TGT = "../data_splits/test.bidir.tgt"

with open(TEST_SRC, "r", encoding="utf-8") as f:
    test_src_texts = [line.strip() for line in f.readlines()][:TEST_SAMPLE_NUM]

with open(TEST_TGT, "r", encoding="utf-8") as f:
    test_tgt_texts = [line.strip() for line in f.readlines()][:TEST_SAMPLE_NUM]

# Traduction
print(f"{TEST_SAMPLE_NUM} échantillons de traductions ont été générés...")
translated_texts = []
for text in test_src_texts:
    out = translator(text, max_length=400)  # Pour les phrases plus longues
    translated_texts.append(out[0]['translation_text'])

# Score bleu
bleu_metric = evaluate.load("sacrebleu")
references = [[lbl] for lbl in test_tgt_texts]  # sacreBLEU
bleu_result = bleu_metric.compute(predictions=translated_texts, references=references)
print("mid_training BLEU:", bleu_result["score"])

# Sauvgarder
mid_model_dir = os.path.join(OUTPUT_DIR, "mid_training_model")
os.makedirs(mid_model_dir, exist_ok=True)
model.save_pretrained(mid_model_dir)
tokenizer.save_pretrained(mid_model_dir)

with open(os.path.join(mid_model_dir, "mid_test_translations.txt"), "w", encoding="utf-8") as f:
    for src, pred, ref in zip(test_src_texts, translated_texts, test_tgt_texts):
        f.write(f"SRC: {src}\nPRED: {pred}\nREF: {ref}\n\n")

with open(os.path.join(mid_model_dir, "mid_test_bleu.txt"), "w") as f:
    f.write(f"{bleu_result['score']:.2f}\n")

print(f"Modèle et résultat d'évaluation ont été enregistrés dans {mid_model_dir}")
