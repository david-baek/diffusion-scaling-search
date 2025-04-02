import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch.nn.functional as F
import os
import json
from PIL import Image

class GeneratedImagesDataset(Dataset):
    """Custom Dataset to load generated images from JSON files."""
    def __init__(self, json_folder, transform=None):
        self.json_folder = json_folder
        self.transform = transform
        self.image_paths = []

        # Read all JSON files starting with 'prompt'
        for filename in os.listdir(json_folder):
            if filename.startswith('prompt') and filename.endswith('.json'):
                json_path = os.path.join(json_folder, filename)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    img_path = data.get('best_img_path')
                    if img_path and os.path.exists(img_path):
                        self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def compute_fid_is_from_json(json_folder, real_images_path, batch_size, device='cuda'):
    """
    Compute FID and IS scores using generated images specified in JSON files.

    Parameters:
    - json_folder: Path to the folder containing JSON files starting with 'prompt'.
    - real_images_path: Path to the directory containing real ImageNet images.
    - batch_size: Batch size for processing images.
    - device: Device to perform computations on (default is 'cuda').

    Returns:
    - fid_value (float): The computed FID score (lower is better).
    - is_value (float): The computed IS score (higher is better).
    """
    # Initialize FID and IS metrics
    fid = FrechetInceptionDistance(normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)

    # Define preprocessing for images
    transform = transforms.Compose([
        transforms.Resize(299),       # Resize smallest side to 299
        transforms.CenterCrop(299),   # Crop to 299x299
        transforms.ToTensor(),        # Convert to tensor with values in [0,1]
    ])

    # Load generated images from JSON files
    generated_dataset = GeneratedImagesDataset(json_folder, transform=transform)
    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

    # Process generated images
    for generated_images in generated_loader:
        generated_images = generated_images.to(device)
        fid.update(generated_images, real=False)
        is_score.update(generated_images)

    # Load real ImageNet images
    real_dataset = torchvision.datasets.ImageFolder(real_images_path, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    # Process real images (limit to the number of generated images for fair comparison)
    num_generated = len(generated_dataset)
    real_images_loaded = 0
    for real_images, _ in real_loader:
        real_images = real_images.to(device)
        
        if real_images_loaded + real_images.size(0) > num_generated:
            real_images = real_images[:num_generated - real_images_loaded]
            
        fid.update(real_images, real=True)
        real_images_loaded += real_images.size(0)
        if real_images_loaded >= num_generated:
            break

    # Compute the final FID and IS scores
    fid_value = fid.compute().item()          # FID as a float
    is_value = is_score.compute()[0].item()   # IS mean as a float (ignores std)

    return fid_value, is_value

# Example usage
json_folder = './output/sd-v1.5/gemini/overall_score/20250401_212426/'       # Replace with path to JSON files
real_images_path = '../imagenet1k_val/' # Replace with path to ImageNet images
batch_size = 1                           # Adjust based on memory
fid_score, is_score = compute_fid_is_from_json(json_folder, real_images_path, batch_size)
print(f"FID Score: {fid_score:.4f}")
print(f"IS Score: {is_score:.4f}")