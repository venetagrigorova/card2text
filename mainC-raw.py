import spacy, os, glob, json, re

nlp = spacy.load("de_core_news_sm")

LABEL_MAP = {
    "PER": "Photographer",
    "LOC": "Location",
    "GPE": "Location",
    "DATE": "Date",
    "ORG": "Description",
    "MISC": "Description"
}

base_dir = os.path.dirname(__file__)
gt_dir = os.path.join(base_dir, "data", "ground_truth", "txt")
output_dir = os.path.join(base_dir, "data", "entities_json_raw")
os.makedirs(output_dir, exist_ok=True)

combined_results = []

for txt_file in glob.glob(os.path.join(gt_dir, "*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    filename = os.path.basename(txt_file)
    doc = nlp(text)

    labeled = {"Location": [], "Date": [], "Photographer": [], "Film": [], "Description": []}
    for ent in doc.ents:
        label = LABEL_MAP.get(ent.label_, None)
        if label:
            ent_text = ent.text.strip().rstrip(".,;:")
            labeled[label].append(ent_text)

    result = {
        "Filename": filename,
        "Text": text,
        "Location": labeled["Location"][0] if labeled["Location"] else "",
        "Description": labeled["Description"][0] if labeled["Description"] else "",
        "Date": labeled["Date"][0] if labeled["Date"] else "",
        "Photographer": labeled["Photographer"][0] if labeled["Photographer"] else "",
        "Film": labeled["Film"][0] if labeled["Film"] else ""
    }

    out_path = os.path.join(output_dir, filename.replace(".txt", ".json"))
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)

    combined_results.append(result)
    print(f"Processed (raw NER): {filename}")

combined_path = os.path.join(base_dir, "entities-combined-raw.json")
with open(combined_path, "w", encoding="utf-8") as out_f:
    json.dump(combined_results, out_f, indent=2, ensure_ascii=False)

print(f"Saved combined raw results to: {combined_path}")
