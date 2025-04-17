import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip  # Ensure you have the OpenAI CLIP package installed

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
        return image, prompt

def compute_clipscore(json_folder, batch_size=8, device='cuda'):
    """
    Compute the average CLIPscore for a set of generated images (with text prompts)
    saved in JSON files.
    
    Parameters:
    - json_folder (str): Path to the folder containing JSON files.
    - batch_size (int): Number of samples to process at once.
    - device (str): Device to run computations on (e.g., 'cuda' or 'cpu').
    
    Returns:
    - avg_clipscore (float): Average cosine similarity between image and text features.
    """
    # Load the CLIP model and its preprocessing transform.
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # Set the model to evaluation mode

    # Prepare the dataset using the CLIP preprocessing transformation.
    dataset = GeneratedImagesWithTextDataset(json_folder, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    clip_scores = []  # To store cosine similarity scores for each image-text pair

    # Iterate over the dataset in batches.
    for images, texts in dataloader:
        images = images.to(device)
        # Tokenize the list of text prompts; clip.tokenize handles a list of strings.
        text_tokens = clip.tokenize(texts).to(device)

        with torch.no_grad():
            # Compute image and text features.
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

        # Normalize features to unit vectors.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate the cosine similarity for each image-text pair.
        # Since features are normalized, the dot product equals cosine similarity.
        cosine_sim = (image_features * text_features).sum(dim=-1)
        clip_scores.extend(cosine_sim.cpu().tolist())
        
    print("CLIP scores", clip_scores)

    # Compute the average CLIPscore over all samples.
    avg_clipscore = sum(clip_scores) / len(clip_scores)
    return avg_clipscore

def compute_single_clipscore(image_path, text_prompt, device=None):
    """
    Compute the CLIPscore for a single image and text prompt.

    The CLIPscore is a measure of similarity between an image and a text prompt,
    calculated as the cosine similarity between their feature embeddings produced
    by the CLIP model.

    Parameters:
    - image_path (str): Path to the image file.
    - text_prompt (str): The text prompt associated with the image.
    - device (str, optional): Device to run computations on. Defaults to 'cuda' 
      if available, else 'cpu'.

    Returns:
    - clipscore (float): The cosine similarity between the image and text features,
      ranging from -1 to 1 (typically 0 to 1 for meaningful pairs).
    """
    # Set device to 'cuda' if available and not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CLIP model and preprocessing transform
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Tokenize the text prompt
    text_tokens = clip.tokenize([text_prompt]).to(device)
    
    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
    print("SHape:", image_features.shape, text_features.shape)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity
    cosine_sim = (image_features * text_features).sum(dim=-1).item()
    
    return 2.5 * max(cosine_sim, 0)

# Example usage:
if __name__ == '__main__':
    json_folder = './output/sd-v1.5/gemini/overall_score/20250401_212426/'  # Replace with your path to JSON files
    batch_size = 8  # Adjust batch size depending on your GPU/CPU memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_score = compute_clipscore(json_folder, batch_size=batch_size, device=device)
    print(f"Average CLIPScore: {clip_score:.4f}")
