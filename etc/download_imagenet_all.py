from datasets import load_dataset
import os
import sys

# Load the dataset
dataset = load_dataset("ILSVRC/imagenet-1k")

def save_split(split, out_dir):
    ds = dataset[split]
    os.makedirs(out_dir, exist_ok=True)
    for idx, example in enumerate(ds):
        class_name = ds.features["label"].int2str(example["label"])
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        file_path = os.path.join(class_dir, f"{idx}.jpg")
        image = example["image"]
        # Convert to RGB if the image is not in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(file_path)
        if idx % 1000 == 0:
            print(f"Saved {idx} images in {split} split...")
            sys.stdout.flush()

# Save training and validation splits into separate folders
#save_split("train", "imagenet1k_train")
save_split("validation", "imagenet1k_val")
