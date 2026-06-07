import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def process_single_image(img_path, xml_path, output_images_dir):
    """
    Process a single image: divide into quadrants if possible, or keep original.
    Returns a list of dictionaries with 'filename' and 'label' for each generated image.
    """
    # Load annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get original filename
    original_filename = root.find('filename').text
    base_name = os.path.splitext(original_filename)[0]
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return [], False, 0, 0
    
    h, w, _ = img.shape
    
    # Get all person bounding boxes
    objects = root.findall('object')
    boxes = []
    
    for obj in objects:
        if obj.find('name').text == 'person':
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            boxes.append((xmin, ymin, xmax, ymax))
    
    # Calculate midpoints for quadrant division
    mid_x, mid_y = w // 2, h // 2
    
    # Check if we can divide (no person crosses quadrant boundaries)
    can_divide = True
    quadrant_indices = set()  # Use set to avoid duplicates if multiple people in same quadrant
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        
        # Check if box is fully contained in one quadrant
        in_left = xmax <= mid_x
        in_right = xmin >= mid_x
        in_top = ymax <= mid_y
        in_bottom = ymin >= mid_y
        
        # If box doesn't fit cleanly in one quadrant, cannot divide
        if not ((in_left or in_right) and (in_top or in_bottom)):
            can_divide = False
            break
        
        # Determine which quadrant: 0=Top-Left, 1=Top-Right, 2=Bottom-Left, 3=Bottom-Right
        if in_left and in_top:
            quadrant_indices.add(0)
        elif in_right and in_top:
            quadrant_indices.add(1)
        elif in_left and in_bottom:
            quadrant_indices.add(2)
        elif in_right and in_bottom:
            quadrant_indices.add(3)
    
    results = []
    divided = False
    images_with_people = 0
    images_without_people = 0
    
    if can_divide and len(boxes) > 0:
        # Divide into quadrants
        divided = True
        quads = [
            (0, 0, mid_x, mid_y),      # Top-Left (0)
            (mid_x, 0, w, mid_y),      # Top-Right (1)
            (0, mid_y, mid_x, h),      # Bottom-Left (2)
            (mid_x, mid_y, w, h)       # Bottom-Right (3)
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(quads):
            # Extract quadrant
            part = img[y1:y2, x1:x2]
            part_filename = f"{base_name}_part_{i}.jpg"
            part_path = os.path.join(output_images_dir, part_filename)
            
            # Save quadrant image
            cv2.imwrite(part_path, part)
            
            # Determine label: 1 if person in this quadrant, 0 otherwise
            label = 1 if i in quadrant_indices else 0
            
            results.append({
                "filename": part_filename,
                "label": label
            })
            
            if label == 1:
                images_with_people += 1
            else:
                images_without_people += 1
    
    # Always save the original image with label 1 (contains person)
    original_output_path = os.path.join(output_images_dir, original_filename)
    cv2.imwrite(original_output_path, img)
    
    results.append({
        "filename": original_filename,
        "label": 1
    })
    images_with_people += 1
    
    return results, divided, images_with_people, images_without_people


def process_dataset(base_dir, output_dir):
    """
    Process entire dataset: all images in train folder.
    
    Args:
        base_dir: Root directory containing 'annotations' and 'infrared' folders
        output_dir: Output directory for results
    """
    # Define paths
    annotations_dir = os.path.join(base_dir, 'annotations')
    train_images_dir = os.path.join(base_dir, 'infrared', 'train')
    
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Get all image files in train directory
    image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
    
    # Initialize counters
    total_original_images = 0
    total_divided_images = 0
    total_images_with_people = 0
    total_images_without_people = 0
    
    # Store all results for final CSV
    all_results = []
    
    print(f"Found {len(image_files)} images in train set")
    print("Processing images...")
    
    # Process each image
    for img_filename in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(train_images_dir, img_filename)
        
        # Construct corresponding XML path
        base_name = os.path.splitext(img_filename)[0]
        xml_filename = f"{base_name}.xml"
        xml_path = os.path.join(annotations_dir, xml_filename)
        
        # Check if annotation exists
        if not os.path.exists(xml_path):
            print(f"\nWarning: No annotation found for {img_filename}, skipping...")
            continue
        
        # Process the image
        results, was_divided, imgs_with_people, imgs_without_people = process_single_image(
            img_path, xml_path, output_images_dir
        )
        
        # Update counters
        total_original_images += 1
        if was_divided:
            total_divided_images += 1
        total_images_with_people += imgs_with_people
        total_images_without_people += imgs_without_people
        
        # Add results to master list
        all_results.extend(results)
    
    # Create DataFrame and save as CSV (PyTorch-compatible)
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_labels_dir, 'dataset_labels.csv')
    df.to_csv(csv_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total original images processed:    {total_original_images}")
    print(f"Images that were divided:           {total_divided_images}")
    print(f"Images that were NOT divided:       {total_original_images - total_divided_images}")
    print(f"-" * 60)
    print(f"Total images WITH people (label=1): {total_images_with_people}")
    print(f"Total images WITHOUT people (label=0): {total_images_without_people}")
    print(f"Total generated images:             {len(all_results)}")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  Images: {output_images_dir}")
    print(f"  Labels: {csv_path}")
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df


if __name__ == "__main__":
    # Configure paths
    BASE_DIR = r"C:\Users\mrodri11\Downloads\LLVIP\LLVIP"  # Change this to your dataset root directory
    OUTPUT_DIR = r"C:\Users\mrodri11\Documents\UNI\deep\final_project\results"     # Change this to your desired output directory
    
    # Example structure expected:
    # BASE_DIR/
    # ├── annotations/
    # │   ├── 010008.xml
    # │   ├── 010009.xml
    # │   └── ...
    # └── infrared/
    #     ├── train/
    #     │   ├── 010008.jpg
    #     │   ├── 010009.jpg
    #     │   └── ...
    #     └── test/
    #         └── ...
    
    # Process the dataset
    df = process_dataset(BASE_DIR, OUTPUT_DIR)
    
    print("\n✓ Processing complete! You can now use the generated CSV with PyTorch DataLoader.")
