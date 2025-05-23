import spacy
import os
import glob
import json
import re

nlp = spacy.load("de_core_news_sm")
print("Model loaded")

base_dir = os.path.dirname(__file__)
gt_dir = os.path.join(base_dir, "data", "ground_truth", "txt")
output_dir = os.path.join(base_dir, "data", "entities_json")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(base_dir, "rules.json"), "r", encoding="utf-8") as f:
    rules = json.load(f)

ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
patterns = []

# add static patterns
patterns.extend(rules["PATTERNS"])
ruler.add_patterns(patterns)

# function to clean photographer names
def clean_photographer(name):
    return re.sub(r"^(aufn\.?|aufnahme)\s*", "", name, flags=re.IGNORECASE).strip(" ,.")

combined_results = []

for txt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    filename = os.path.basename(txt_file)
    doc = nlp(text)

    labeled = {"Location": [], "Date": [], "Photographer": [], "Film": [], "Description": []}

    for ent in doc.ents:
        ent_text = ent.text.strip().rstrip(".,;:")
        if ent.label_ in labeled:
            labeled[ent.label_].append(ent_text)

    result = {
        "Filename": filename,
        "Location": labeled["Location"][0].strip() if labeled["Location"] else "",
        "Description": labeled["Description"][0].strip() if labeled["Description"] else "",
        "Date": labeled["Date"][0].strip() if labeled["Date"] else "",
        "Photographer": clean_photographer(labeled["Photographer"][0]).strip() if labeled["Photographer"] else "",
        "Film": labeled["Film"][0].strip() if labeled["Film"] else ""
    }

    out_path = os.path.join(output_dir, filename.replace(".txt", ".json"))
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)

    combined_results.append(result)
    print(f"Processed: {filename}")

combined_path = os.path.join(base_dir, "entities-combined.json")
with open(combined_path, "w", encoding="utf-8") as out_f:
    json.dump(combined_results, out_f, indent=2, ensure_ascii=False)

print(f"Saved combined results to: {combined_path}")
