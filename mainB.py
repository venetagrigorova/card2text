import os
import cv2
import pytesseract
import editdistance
import pandas as pd


def summarize_results(all_results, output_csv_path):
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Group by variant and compute summary statistics
    summary = df.groupby("variant").agg(
        CER_mean=("CER", "mean"),
        CER_median=("CER", "median"),
        CER_variance=("CER", "var"),
        CER_min=("CER", "min"),
        CER_max=("CER", "max"),
        WER_mean=("WER", "mean"),
        WER_median=("WER", "median"),
        WER_variance=("WER", "var"),
        WER_min=("WER", "min"),
        WER_max=("WER", "max"),
    ).reset_index()

    # Save to CSV
    summary.to_csv(output_csv_path, index=False)
    print(f"Summary statistics saved to {output_csv_path}")

def save_raw_errors(all_results, output_csv_path):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Raw OCR errors saved to {output_csv_path}")

def find_crop_image(crop_dir, full_name):
    prefix = full_name.split("_ImageFile")[0]
    for crop_name in os.listdir(crop_dir):
        if crop_name.startswith(prefix + "_ImageFile_text_1"):
            return os.path.join(crop_dir, crop_name)
    return None

def binarize_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def cer(gt, pred):
    return editdistance.eval(gt, pred) / len(gt) if gt else 1.0

def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return editdistance.eval(gt_words, pred_words) / len(gt_words) if gt_words else 1.0

def run_experiment(img_path, gt_path, output_dir, scope, bin_method, psm):
    if not os.path.exists(img_path):
        return None

    # Load image and GT
    image = cv2.imread(img_path)
    if image is None:
        return None

    if bin_method == "otsu":
        image = binarize_image(image)

    config = f"--psm {psm}"
    ocr_result = pytesseract.image_to_string(image, config=config).strip()

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = f.read().strip()

    cer_score = cer(ground_truth, ocr_result)
    wer_score = wer(ground_truth, ocr_result)

    variant_name = f"{scope}_{bin_method}_psm{psm}"
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_{variant_name}.txt")

    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write(ocr_result)

    return {
        "image": base_name,
        "variant": variant_name,
        "CER": cer_score,
        "WER": wer_score
    }

def main():
    # Config Dirs
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir,"data", "images")
    text_images_dir = os.path.join(current_dir,"data", "text_images")
    gt_text_dir = os.path.join(current_dir,"data", "ground_truth", "txt")
    output_dir = os.path.join(current_dir,"data", "txt")
    os.makedirs(output_dir, exist_ok=True)
    # Config Tesseract Executable Path
    pytesseract.pytesseract.tesseract_cmd = os.path.join("C:\\", "Program Files", "Tesseract-OCR", "tesseract.exe")
    # Config Tesseract run settings
    #TODO
    
    # Experiment config
    #TODO: add otsu for binarization
    bin_methods = ["none"]
    psm_modes = [3, 6]  # or extend with other PSMs
    all_results = []
    
    #count = 0
    #max_images = 3
    # Run experiment
    for fname in sorted(os.listdir(image_dir)):
        # TODO remove
        #count += 1
        #if count >= max_images:
        #    break
        if not fname.lower().endswith('.jpg'):
            continue

        base = os.path.splitext(fname)[0]
        full_img_path = os.path.join(image_dir, fname)
        crop_img_path = find_crop_image(text_images_dir, fname)
        gt_path = os.path.join(gt_text_dir, base + ".txt")

        if not os.path.exists(gt_path):
            continue

        for scope, img_path in [("full", full_img_path), ("crop", crop_img_path)]:
            for bin_method in bin_methods:
                for psm in psm_modes:
                    result = run_experiment(img_path, gt_path, output_dir, scope, bin_method, psm)
                    if result:
                        all_results.append(result)

    
    save_raw_errors(all_results, "ocr_raw_errors.csv")   
    summarize_results(all_results, "ocr_summary_statistics.csv")
    
    

if __name__ == "__main__":
    main()