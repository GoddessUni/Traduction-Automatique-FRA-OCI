#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

data_dir = Path("../data_splits")  

output_dir = data_dir
output_dir.mkdir(parents=True, exist_ok=True)

splits = ["train", "valid", "test"]

def generate_bilingual(src_file, tgt_file, out_src_file, out_tgt_file):
    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = f.read().splitlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = f.read().splitlines()

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"{src_file} et {tgt_file} n'ont pas même nombre de lignes!")

    # fr -> oc
    src_bidir = [f"<fr> {line}" for line in src_lines]
    tgt_bidir = [line for line in tgt_lines]

    # oc -> fr
    src_bidir += [f"<oc> {line}" for line in tgt_lines]
    tgt_bidir += [line for line in src_lines]

    with open(out_src_file, "w", encoding="utf-8") as f:
        f.write("\n".join(src_bidir) + "\n")
    with open(out_tgt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tgt_bidir) + "\n")

    print(f" {out_src_file} et {out_tgt_file} ont été générés, nombre de lignes: {len(src_bidir)}")

# parcourir tous les fichiers
for split in splits:
    src_file = data_dir / f"{split}.fr"
    tgt_file = data_dir / f"{split}.oc"

    out_src_file = output_dir / f"{split}.bidir.src"
    out_tgt_file = output_dir / f"{split}.bidir.tgt"

    generate_bilingual(src_file, tgt_file, out_src_file, out_tgt_file)

print("Les corpus bidirectionnels ont été générés")
