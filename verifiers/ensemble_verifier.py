import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Any
from torchvision import transforms
import scipy.stats as stats

from .base_verifier import BaseVerifier
from .laion_aesthetics import LAIONAestheticVerifier

import clip

class EnsembleVerifier(BaseVerifier):
    """
    Ensemble verifier that uses rank aggregation of component scores.

    Instead of weighted averaging of scores, this implementation:
    1. Ranks images separately by each metric (CLIP and aesthetic)
    2. Averages these ranks to create the final ranking
    3. Normalizes the result to [0,1] scale

    Config parameters:
    - clip_model_path: Path to CLIP model weights file
    - normalize_scores: Whether to normalize final scores to [0,1] (default: True)
    """

    SUPPORTED_METRIC_CHOICES = ["ensemble_score", "clip_score", "aesthetic_score"]

    def __init__(
        self,
        clip_model_path: str = None,
        normalize_scores: bool = True,
        **kwargs
    ):

        # Remove args not used in this implementation
        kwargs.pop("clip_weight", None)
        kwargs.pop("aesthetic_weight", None)
        kwargs.pop("clip_multiplier", None)
        kwargs.pop("name", None)
        kwargs.pop("choice_of_metric", None)

        super().__init__(**kwargs)

        # Store config
        self.normalize_scores = normalize_scores

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize CLIP model
        if clip_model_path and os.path.exists(clip_model_path):
            self.clip_model, self.clip_preprocess = clip.load(clip_model_path, device=self.device)
        else:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_model.eval()

        # Initialize aesthetic scorer
        self.aesthetic_verifier = LAIONAestheticVerifier()

        # Verify LAION verifier supports required metric
        if "laion_aesthetic_score" not in self.aesthetic_verifier.SUPPORTED_METRIC_CHOICES:
            raise ValueError("LAION verifier must support 'laion_aesthetic_score'")

        # Set up preprocessing
        self.aesthetic_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def prepare_inputs(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompts: Union[str, List[str]],
        **kwargs
    ) -> List[Tuple[str, Image.Image]]:
        """Prepares inputs for scoring."""
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        return list(zip(prompts, images))

    @torch.no_grad()
    def score(
        self,
        inputs: List[Tuple[str, Image.Image]],
        **kwargs
    ) -> List[Dict[str, float]]:
        """
        Score images using rank aggregation of CLIP and aesthetic scores.

        Returns list of dictionaries with keys:
        - ensemble_score: Average rank-based score (higher is better)
        - clip_score: Original CLIP similarity score
        - aesthetic_score: Original LAION aesthetic score
        """
        if len(inputs) <= 1:
            # With just one image, rankings don't make sense
            # Return default scores
            return self._score_single_image(inputs)

        # Split prompts and images
        prompts, pil_images = zip(*inputs)
        prompts = [self._truncate_prompt(prompt) for prompt in prompts]

        # Get CLIP scores
        clip_images = torch.stack([self.clip_preprocess(img) for img in pil_images]).to(self.device)
        text_inputs = clip.tokenize(prompts).to(self.device)

        image_features = self.clip_model.encode_image(clip_images)
        text_features = self.clip_model.encode_text(text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        clip_scores = (image_features * text_features).sum(dim=-1).cpu().numpy()

        # Get aesthetic scores
        aesthetic_inputs = []
        for img in pil_images:
            img_tensor = self.aesthetic_preprocess(img).unsqueeze(0).to(self.device)
            aesthetic_inputs.append({"pixel_values": img_tensor})

        aesthetic_outputs = self.aesthetic_verifier.score(aesthetic_inputs)
        aesthetic_scores = np.array([out["laion_aesthetic_score"] for out in aesthetic_outputs])

        # Calculate rankings (lower rank = better)
        # scipy.stats.rankdata uses 1-based ranking (1 = best)
        clip_ranks = stats.rankdata(-clip_scores)  # Negative so higher scores get lower ranks
        aesthetic_ranks = stats.rankdata(-aesthetic_scores)

        # Average the ranks
        average_ranks = (clip_ranks + aesthetic_ranks) / 2

        # Convert to normalized score if requested (higher = better)
        if self.normalize_scores:
            # Convert to range [0, 1] where 1 is best
            normalized_scores = 1 - ((average_ranks - 1) / (len(inputs) - 1)) if len(inputs) > 1 else np.ones_like(average_ranks)
        else:
            # Use raw ranks (lower is better, so invert)
            normalized_scores = len(inputs) + 1 - average_ranks

        # Prepare results
        results = []
        for i in range(len(inputs)):
            results.append({
                "ensemble_score": float(normalized_scores[i]),
                "clip_score": float(clip_scores[i]),
                "aesthetic_score": float(aesthetic_scores[i])
            })

        return results

    def _truncate_prompt(self, prompt, max_length=77):
        """Truncate prompt to fit CLIP's context length."""
        words = prompt.split()
        if len(words) == 0:
            return prompt
        # Start with the full prompt and remove words from the end until it fits
        for end in range(len(words), 0, -1):
            candidate = " ".join(words[:end])
            try:
                tokens = clip.tokenize([candidate])
                if tokens.shape[1] <= max_length:
                    return candidate
            except RuntimeError:
                continue
        # If even one word is too long, return the first word
        return words[0]

    def _score_single_image(self, inputs):
        """Handle the special case of scoring a single image."""
        prompt, image = inputs[0]

        # Truncate prompt if needed
        prompt = self._truncate_prompt(prompt)

        # Get CLIP score
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)

        image_feature = self.clip_model.encode_image(clip_image)
        text_feature = self.clip_model.encode_text(text_input)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        clip_score = (image_feature * text_feature).sum(dim=-1).item()

        # Get aesthetic score
        img_tensor = self.aesthetic_preprocess(image).unsqueeze(0).to(self.device)
        aesthetic_score = self.aesthetic_verifier.score([{"pixel_values": img_tensor}])[0]["laion_aesthetic_score"]

        # With a single image, we'll use the max possible score
        ensemble_score = 1.0 if self.normalize_scores else 1.0

        return [{
            "ensemble_score": ensemble_score,
            "clip_score": float(clip_score),
            "aesthetic_score": float(aesthetic_score)
        }]
