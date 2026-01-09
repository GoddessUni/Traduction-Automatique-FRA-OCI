import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import spacy

# La direction de données
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "train_sentences"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR = OUTPUT_DIR / "validation_report"
REPORT_DIR.mkdir(exist_ok=True)


nlp_fr = spacy.load("fr_core_news_sm")


def clean_html(text):
    """Nettoyage du fichier html.json"""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    spans = soup.find_all("span", class_="cx-segment")
    text_segments = [s.get_text(separator=" ", strip=True) for s in spans]
    text = " ".join(text_segments)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[\s*[^\]]+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    return text.strip()


def split_fr(text):
    """Segmenter le texte français en phrases avec SpaCy"""
    return [sent.text.strip() for sent in nlp_fr(text).sents if sent.text.strip()]


def split_oc(text, max_len=300):
    """
    Segmenter le texte occitan en phrases de manière robuste.
    Args:
        text (str): texte occitan à segmenter
        max_len (int): longueur maximale d'une phrase avant découpe secondaire
    Returns:
        list[str]: liste de phrases
    """
    if not text:
        return []

    # Liste d'abréviations fréquentes
    abbrevs = [
        r"M", r"Mr", r"Dr", r"Pr", r"St", r"Av", r"etc", r"fig", r"n", r"vol", r"p",
        r"ISBN", r"e\.g", r"i\.e", r"cf", r"p\. ex", r"cfr"
    ]
    abbrev_regex = re.compile(r'\b(?:' + '|'.join(abbrevs) + r')\.', flags=re.IGNORECASE)
    url_email_regex = re.compile(r'\b(?:https?://|www\.)\S+|\b\S+@\S+\b')

    placeholders = {}

    # remplacer URL/email
    def protect(match):
        key = f"__PROT_{len(placeholders)}__"
        placeholders[key] = match.group()
        return key

    text = url_email_regex.sub(protect, text)
    text = abbrev_regex.sub(protect, text)

    # segmentation simple
    sentences = re.split(r'(?<=[.?!…])\s+', text)

    # nettoyage
    sentences = [s.strip().replace('\xa0', ' ').replace('\u200b', '') for s in sentences if s.strip()]

    final_sents = []

    for s in sentences:
        # segmenter les phrases longues
        while len(s) > max_len:
            cut_pos = s.rfind(' ', 0, max_len)
            if cut_pos == -1:
                cut_pos = max_len
            final_sents.append(s[:cut_pos].strip())
            s = s[cut_pos:].strip()
        if s:
            final_sents.append(s)
            
    for i in range(len(final_sents)):
        for key, val in placeholders.items():
            final_sents[i] = final_sents[i].replace(key, val)

    return final_sents


# Extraire les phrases parallèles depuis JSON
def process_json(file_path):
    src_sents_all = []
    tgt_sents_all = []
    print(f"Processing JSON: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        src_text = clean_html(entry['source']['content'])
        tgt_text = clean_html(entry['target']['content'])
        src_sents = split_fr(src_text)
        tgt_sents = split_oc(tgt_text)
        if len(src_sents) == len(tgt_sents) and len(src_sents) > 0:
            src_sents_all.extend(src_sents)
            tgt_sents_all.extend(tgt_sents)
        else:
            if src_text and tgt_text:
                src_sents_all.append(src_text)
                tgt_sents_all.append(tgt_text)
    return src_sents_all, tgt_sents_all

# Extraire les phrases parallèles depuis TMX
def process_tmx(file_path):
    src_sents_all = []
    tgt_sents_all = []
    print(f"Processing TMX: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    for tu in root.findall(".//tu"):
        src_seg = ""
        tgt_seg = ""
        for tuv in tu.findall("tuv"):
            lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            seg_elem = tuv.find("seg")
            seg_text = seg_elem.text.strip() if seg_elem is not None and seg_elem.text else ""
            if lang.startswith("fr"):
                src_seg = seg_text
            elif lang.startswith("oc"):
                tgt_seg = seg_text
        if src_seg and tgt_seg:
            src_sents = split_fr(src_seg)
            tgt_sents = split_oc(tgt_seg)
            if len(src_sents) == len(tgt_sents) and len(src_sents) > 0:
                src_sents_all.extend(src_sents)
                tgt_sents_all.extend(tgt_sents)
            else:
                src_sents_all.append(src_seg)
                tgt_sents_all.append(tgt_seg)
    return src_sents_all, tgt_sents_all


all_source = []
all_target = []

for root_dir, dirs, files in os.walk(DATA_DIR):
    for file in files:
        path = os.path.join(root_dir, file)
        if file.endswith(".json"):
            src, tgt = process_json(path)
        elif file.endswith(".tmx"):
            src, tgt = process_tmx(path)
        else:
            continue
        all_source.extend(src)
        all_target.extend(tgt)

# Garder les paires uniques
pairs = list(set(zip(all_source, all_target)))
all_source, all_target = zip(*pairs) if pairs else ([], [])

with open(OUTPUT_DIR / "train_sentences.fr", "w", encoding="utf-8") as f:
    for line in all_source:
        f.write(line + "\n")

with open(OUTPUT_DIR / "train_sentences.oc", "w", encoding="utf-8") as f:
    for line in all_target:
        f.write(line + "\n")

print(f"Finished. Total unique parallel sentences: {len(all_source)}")
