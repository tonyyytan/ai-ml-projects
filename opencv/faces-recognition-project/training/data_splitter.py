import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATASET_NAME = "vishesh1412/celebrity-face-image-dataset"
OUTPUT_DIR = "data_splits"       # Where to save the split files
RANDOM_SEED = 42

def prepare_data_splits():
    # 1. Download Dataset via KaggleHub
    print(f"Downloading {DATASET_NAME}...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to: {path}")

    # 2. Locate the actual image directory
    # The dataset structure is often: /cache/.../Celebrity Faces Dataset/<Person Name>/<image.jpg>
    base_dir = os.path.join(path, "Celebrity Faces Dataset")
    
    if not os.path.exists(base_dir):
        # Fallback if the folder structure is different
        base_dir = path 

    print(f"Scanning images in: {base_dir}")

    # 3. Collect all image paths and labels
    data = []
    
    # Get all subdirectories (classes)
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    classes.sort()  # Ensure consistent ordering
    
    # Create a mapping from Class Name -> Integer Label
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(class_dir, img_name)
                data.append({
                    'filepath': full_path,
                    'label_name': class_name,
                    'label_idx': class_to_idx[class_name]
                })

    df = pd.DataFrame(data)
    print(f"Found {len(df)} total images across {len(classes)} classes.")

    # 4. Split the Data (Stratified)
    # Split: 70% Train, 15% Val, 15% Test
    
    # First: Separate Test (15%)
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df['label_name'], random_state=RANDOM_SEED
    )

    # Second: Separate Train (70% of total) and Val (15% of total) from the remaining 85%
    # 0.15 / 0.85 ~= 0.1765
    train, val = train_test_split(
        train_val, test_size=0.1765, stratify=train_val['label_name'], random_state=RANDOM_SEED
    )

    # 5. Save Splits to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    
    # Also save the class mapping for reference
    pd.DataFrame(list(class_to_idx.items()), columns=['class_name', 'class_idx'])\
      .to_csv(f"{OUTPUT_DIR}/class_mapping.csv", index=False)

    print("\n--- Processing Complete ---")
    print(f"Train: {len(train)} images -> {OUTPUT_DIR}/train.csv")
    print(f"Val:   {len(val)} images   -> {OUTPUT_DIR}/val.csv")
    print(f"Test:  {len(test)} images  -> {OUTPUT_DIR}/test.csv")
    print(f"Class Map saved to {OUTPUT_DIR}/class_mapping.csv")

if __name__ == "__main__":
    prepare_data_splits()