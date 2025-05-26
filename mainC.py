import spacy
import os
import glob
import json
import re

# load German spaCy model
nlp = spacy.load("de_core_news_sm")
print("Model loaded")

# set up directories
base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "data", "txt")
output_dir = os.path.join(base_dir, "data", "entities_json")
os.makedirs(output_dir, exist_ok=True)

# load custom pattern rules from rules.json
with open(os.path.join(base_dir, "rules.json"), "r", encoding="utf-8") as f:
    rules = json.load(f)

# add custom entity patterns via spaCy EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
ruler.add_patterns(rules["PATTERNS"])  # inject rules from JSON

# fix OCR errors before NLP
def normalize_ocr(text):
    text = (
        text.replace("Pilm", "Film")
            .replace("Rollfiim", "Rollfilm")
            .replace("Rolifiim", "Rollfilm")
            .replace("Leicaf.", "Leicafilm")
            .replace("Aufu", "Aufn")
            .replace("Aufns", "Aufn")
            .replace("Aufa", "Aufn")
            .replace("Aufn,", "Aufn")
            .replace("Jinner", "Jänner")
            .replace("Yanner", "Jänner")
            .replace("Mirz", "März")
            .replace("Feber.", "Feber")
            .replace("Dr,", "Dr. ")  # correct misplaced commas
    )
    # separate glued 'AufnDr' or 'AufnMeisinger' into two tokens
    text = re.sub(r"(Aufn)[.,]?([A-ZÄÖÜ])", r"\1. \2", text)
    return text

# remove 'aufn' prefix, strip month suffixes, clean punctuation
def clean_photographer(name):
    name = re.sub(r"^(aufn\.?|aufnahme)[\s.,]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(j[a-z]+|f[a-z]+|märz?|april|mai|juni?|juli?|aug|sep|okt|nov|dez)[\s.,]*$", "", name, flags=re.IGNORECASE)
    return name.strip(" ,._-")

combined_results = []

# process all OCR text files matching the naming pattern
for txt_file in glob.glob(os.path.join(input_dir, "*crop_otsu_bin_psm6.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    filename = os.path.basename(txt_file)
    text = normalize_ocr(text)  # apply OCR fixes

    # grab lines to extract line-level info like description
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    description_line = lines[1] if len(lines) > 1 else ""

    doc = nlp(text)  # run spaCy NLP pipeline

    # store matched entities by label
    labeled = {"Location": [], "Date": [], "Photographer": [], "Film": [], "Description": []}
    used_spans = set()  # to prevent overlapping entities

    # extract entities
    for ent in doc.ents:
        ent_text = ent.text.strip().rstrip(".,;:")
        if ent.start_char in used_spans:
            continue  # already labeled by a higher-priority rule
        if ent.label_ != "Description":
            used_spans.update(range(ent.start_char, ent.end_char))
        if ent.label_ in labeled:
            labeled[ent.label_].append(ent_text)

    # build output dictionary
    result = {
        "Filename": filename,
        "Text": text,
        "Location": labeled["Location"][0].strip() if labeled["Location"] else "",
        "Description": description_line if not labeled["Description"] else labeled["Description"][0].strip(),
        "Date": labeled["Date"][0].strip() if labeled["Date"] else "",
        "Photographer": clean_photographer(labeled["Photographer"][0]) if labeled["Photographer"] else "",
        "Film": labeled["Film"][0].strip(" ,._-") if labeled["Film"] else ""
    }

    # write per-file JSON output
    out_path = os.path.join(output_dir, filename.replace(".txt", ".json"))
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)

    combined_results.append(result)
    print(f"Processed: {filename}")

# write combined JSON with all results
combined_path = os.path.join(base_dir, "task-c-spacy.json")
with open(combined_path, "w", encoding="utf-8") as out_f:
    json.dump(combined_results, out_f, indent=2, ensure_ascii=False)

print(f"Saved combined results to: {combined_path}")
