#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint
import torch

# Configuration
MODEL_NAME = "facebook/nllb-200-distilled-600M"
BASE_OUTPUT_DIR = "/data/projets_m2_2526/dcheng/outputs"

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 2               
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Data en 2 direction
DATA_DIR = "../data_splits"
TRAIN_SRC = os.path.join(DATA_DIR, "train.bidir.src")
TRAIN_TGT = os.path.join(DATA_DIR, "train.bidir.tgt")
VALID_SRC = os.path.join(DATA_DIR, "valid.bidir.src")
VALID_TGT = os.path.join(DATA_DIR, "valid.bidir.tgt")
TEST_SRC = os.path.join(DATA_DIR, "test.bidir.src")
TEST_TGT = os.path.join(DATA_DIR, "test.bidir.tgt")

# Nettoyage de la mémoire de GPU
torch.cuda.empty_cache()

# La sortie de catalogue
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
existing_runs = [
    int(d.split("_")[-1]) for d in os.listdir(BASE_OUTPUT_DIR)
    if os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d)) and d.startswith("run_") and d.split("_")[-1].isdigit()
]
next_run_id = max(existing_runs, default=0) + 1
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{next_run_id}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Catalogue d'entraînement:", OUTPUT_DIR)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Charger les corpus
print("Chargement des corpus...")
raw_datasets = DatasetDict({
    "train": load_dataset("text", data_files={"src": TRAIN_SRC, "tgt": TRAIN_TGT}),
    "valid": load_dataset("text", data_files={"src": VALID_SRC, "tgt": VALID_TGT}),
    "test": load_dataset("text", data_files={"src": TEST_SRC, "tgt": TEST_TGT}),
})

def merge_src_tgt(raw_ds, split_name):
    src_texts = raw_ds[split_name]["src"]["text"]
    tgt_texts = raw_ds[split_name]["tgt"]["text"]
    return {"input_text": src_texts, "target_text": tgt_texts}

datasets = DatasetDict()
for split in ["train", "valid", "test"]:
    merged = merge_src_tgt(raw_datasets, split)
    datasets[split] = Dataset.from_dict(merged)
    print(f"{split} données ont été chargées, nombre d'échantillons: {len(datasets[split])}")

# Tokenization
SRC_LANG = "fra_Latn"
TGT_LANG = "oci_Latn"
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

def tokenize_function(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenization commence...")
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["input_text", "target_text"],
)
print("Tokenization fini!")

# Charger le modèle
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Collecter les données
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Évaluation bleu
bleu_metric = evaluate.load("sacrebleu")
best_bleu = 0.0

def compute_metrics(eval_preds):
    global best_bleu
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["bleu"] = result.pop("score")
    
    if result["bleu"] > best_bleu:
        best_bleu = result["bleu"]
        with open(os.path.join(OUTPUT_DIR, "best_bleu.txt"), "w") as f:
            f.write(f"{best_bleu:.2f}\n")
    return result

# Arguments d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=200,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    push_to_hub=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Checkpoint
last_ckpt = get_last_checkpoint(OUTPUT_DIR)
if last_ckpt is not None:
    print(f"Aucun checkpoint n'a été détecté, l'entraînement continu: {last_ckpt}")
else:
    print("Aucun checkpoint n'a été détecté, l'entraînement commence depuis le début.")
    last_ckpt = None

# L'entraînement commence
print("L'entraînement commence...")
trainer.train(resume_from_checkpoint=last_ckpt)
print("L'entraînement fini!")

# Sauvgarder le modèle et le tokenizer
print("Enregistrement du modèle finale et du tokenizer...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

# Calculer le score BLEU
print("Calcule le score BLEU...")
test_results = trainer.evaluate(tokenized_datasets["test"])
with open(os.path.join(OUTPUT_DIR, "test_bleu.txt"), "w") as f:
    f.write(f"{test_results['eval_bleu']:.2f}\n")
print("Le score BLEU:", test_results["eval_bleu"])





