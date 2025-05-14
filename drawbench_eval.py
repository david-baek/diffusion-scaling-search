import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from huggingface_hub import hf_hub_download
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)

class GeneratedImagesWithTextDataset(Dataset):
    """
    Custom Dataset to load generated images and their associated text prompts
    from JSON files. Each JSON file is expected to include:
      - 'best_img_path': path to the generated image.
      - 'prompt': the text prompt used for generating the image.
    """
    def __init__(self, json_folder, transform=None):
        self.json_folder = json_folder
        self.transform = transform
        self.samples = []  # Each sample is a tuple (img_path, prompt)

        # Read all JSON files starting with 'prompt' in the filename.
        for filename in os.listdir(json_folder):
            if filename.startswith('prompt') and filename.endswith('.json'):
                json_path = os.path.join(json_folder, filename)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                img_path = data.get('best_img_path')
                prompt = data.get('prompt')
                if img_path and os.path.exists(img_path) and prompt:
                    self.samples.append((img_path, prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, prompt, img_path  # Added img_path to return value

def load_aesthetic_model(device='cuda'):
    """Load the LAION aesthetic predictor model"""
    model_path = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")

    model = MLP()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_scores(json_folder, batch_size=8, device='cuda'):
    """
    Compute both CLIP scores and aesthetic scores for images in JSON files.

    Parameters:
    - json_folder (str): Path to the folder containing JSON files.
    - batch_size (int): Number of samples to process at once.
    - device (str): Device to run computations on.

    Returns:
    - tuple: (avg_clipscore, avg_aesthetic_score, detailed_scores)
    """
    # Load models
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    aesthetic_model = load_aesthetic_model(device)

    # Prepare the dataset using the CLIP preprocessing transformation
    dataset = GeneratedImagesWithTextDataset(json_folder, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Store scores
    clip_scores = []
    aesthetic_scores = []
    detailed_scores = []  # For per-image results

    # Process batches
    for images, texts, img_paths in dataloader:
        images = images.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        with torch.no_grad():
            # Get CLIP features
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(text_tokens)

            # Normalize features for CLIP score
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate CLIP scores
            cosine_sim = (image_features_norm * text_features_norm).sum(dim=-1)
            clip_score_batch = 2.5 * torch.clamp(cosine_sim, min=0)  # Scale CLIP scores

            # Calculate aesthetic scores
            aesthetic_score_batch = aesthetic_model(image_features).squeeze()

        # Store scores
        clip_scores.extend(clip_score_batch.cpu().tolist())
        aesthetic_scores.extend(aesthetic_score_batch.cpu().tolist())

        # Store detailed results
        for i in range(len(images)):
            detailed_scores.append({
                "image_path": img_paths[i],
                "prompt": texts[i],
                "clip_score": clip_score_batch[i].item(),
                "aesthetic_score": aesthetic_score_batch[i].item()
            })

    # Compute averages
    avg_clipscore = sum(clip_scores) / len(clip_scores)
    avg_aesthetic_score = sum(aesthetic_scores) / len(aesthetic_scores)

    print(f"Average CLIP Score: {avg_clipscore:.4f}")
    print(f"Average Aesthetic Score: {avg_aesthetic_score:.4f}")

    return avg_clipscore, avg_aesthetic_score, detailed_scores

def compute_single_clipscore(image_path, text_prompt, device=None):
    """
    Compute the CLIPscore for a single image and text prompt.

    Parameters:
    - image_path (str): Path to the image file.
    - text_prompt (str): The text prompt associated with the image.
    - device (str, optional): Device to run computations on.

    Returns:
    - clipscore (float): Scaled cosine similarity between image and text features.
    """
    def _truncate_prompt(prompt, max_length=77):
        """Truncate prompt to fit CLIP's context length."""
        words = prompt.split()
        if len(words) == 0:
            return prompt

        # Start with the full prompt and remove words from the end until it fits
        for end in range(len(words), 0, -1):
            candidate = " ".join(words[:end])
            try:
                tokens = clip.tokenize([candidate])
                return candidate
            except RuntimeError:
                # If tokenization fails, try a shorter prompt
                continue

        # If even one word is too long, return an empty string
        return ""

    # Set device to 'cuda' if available and not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CLIP model and preprocessing transform
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

    # Tokenize the text prompt
    text_prompt = _truncate_prompt(text_prompt)
    text_tokens = clip.tokenize([text_prompt]).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    cosine_sim = (image_features * text_features).sum(dim=-1).item()

    return 2.5 * max(cosine_sim, 0)

def compute_aesthetic_score(image_path, device=None):
    """
    Compute the aesthetic score for a single image.

    Parameters:
    - image_path (str): Path to the image file.
    - device (str, optional): Device to run computations on.

    Returns:
    - aesthetic_score (float): The predicted aesthetic score (typically in 1-10 range).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CLIP vision model
    clip_model_path = "openai/clip-vit-large-patch14"
    clip = CLIPVisionModelWithProjection.from_pretrained(clip_model_path)
    clip.eval()
    clip.to(device)

    # Load processor
    processor = CLIPProcessor.from_pretrained(clip_model_path)

    # Load aesthetic MLP
    mlp = MLP()
    model_path = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
    state_dict = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    mlp.load_state_dict(state_dict)
    mlp.eval()
    mlp.to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute aesthetic score
    with torch.no_grad():
        # Get CLIP embeddings
        embed = clip(**inputs)[0]
        # Normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        # Compute aesthetic score
        score = mlp(embed).item()

    return score

# Example usage
if __name__ == '__main__':
    json_folder = './output/sd-v1.5/gemini/overall_score/20250401_212426/'
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Compute both scores for all images
    avg_clip, avg_aesthetic, detailed = compute_scores(json_folder, batch_size=batch_size, device=device)
    print(f"Average CLIPScore: {avg_clip:.4f}")
    print(f"Average Aesthetic Score: {avg_aesthetic:.4f}")

    # Save detailed results to JSON
    with open('image_scores.json', 'w') as f:
        json.dump(detailed, f, indent=2)

    # Example for a single image
    image_path = detailed[0]["image_path"]
    prompt = detailed[0]["prompt"]

    clip_score = compute_single_clipscore(image_path, prompt, device)
    aesthetic_score = compute_aesthetic_score(image_path, device)

    print(f"Single image CLIP score: {clip_score:.4f}")
    print(f"Single image aesthetic score: {aesthetic_score:.4f}")
