from ultralytics import YOLO
import os
import cv2

def main():
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir,"data", "images")
    model_dir = os.path.join(current_dir,"models")
    output_dir = os.path.join(current_dir,"data", "text_images")
    
    model = YOLO(os.path.join(model_dir,"yolov8n.pt"))
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        layout_results = model(image)[0]
        prefix = image_name.split("_")[0] + "_ImageFile"
        
        text_regions = []
        for box in layout_results.boxes:
            cls = int(box.cls[0])
            if cls == 2:  # printed text region
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                text_image = image[y1:y2, x1:x2]
                text_regions.append(text_image)

        if len(text_regions) == 0:
            continue
        elif len(text_regions) == 1:
            # Save single region
            text_image_name = f"{prefix}_text_1.jpg"
            cv2.imwrite(os.path.join(output_dir, text_image_name), text_regions[0])
        else:
            # Resize all regions to the same width (e.g., the max width)
            max_width = max(img.shape[1] for img in text_regions)
            resized = [cv2.copyMakeBorder(
                img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255]
            ) for img in text_regions]
            merged_image = cv2.vconcat(resized)
            text_image_name = f"{prefix}_text_merged.jpg"
            cv2.imwrite(os.path.join(output_dir, text_image_name), merged_image)
        
    
    

if __name__ == "__main__":
    main()