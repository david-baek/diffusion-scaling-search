import torch
import torch.nn as nn
import ImageReward as reward
from peft import LoraConfig, TaskType, inject_adapter_in_model, get_peft_model
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel

class PEFTImageReward(nn.Module):
    def __init__(self, base_model_path="ImageReward-v1.0", timestep_dim=320, text_model_name="roberta-base"):
        super().__init__()
        print(f"Initializing PEFTImageReward - Base: {base_model_path}")
        self.base_reward_model = reward.load(base_model_path)
        self.original_reward_model = reward.load(base_model_path)
        self.original_reward_model.eval()

        # --- Debug: Inspect the loaded base_reward_model --- 
        print("--- Debugging base_reward_model --- ")
        try:
            print(f"Type: {type(self.base_reward_model)}")
            print(f"Attributes/Methods: {dir(self.base_reward_model)}")
            if hasattr(self.base_reward_model, 'blip'):
                 print(f"BLIP component attributes: {dir(self.base_reward_model.blip)}")
        except Exception as e:
            print(f"Error during debug inspection: {e}")
        print("--- End Debugging --- ")
        # --- End Debug ---

        # --- References to Base Model Components ---
        # Vision Encoder (Will be adapted)
        if not hasattr(self.base_reward_model, 'blip') or not hasattr(self.base_reward_model.blip, 'visual_encoder'):
            raise AttributeError("Cannot find '.blip.visual_encoder'.")
        self.vision_encoder = self.base_reward_model.blip.visual_encoder
        print("Identified vision component as '.blip.visual_encoder'")
        self.vision_feature_dim = self.vision_encoder.embed_dim if hasattr(self.vision_encoder, 'embed_dim') else self.vision_encoder.config.hidden_size
        print(f"Vision feature dimension: {self.vision_feature_dim}")

        # --- EDIT: Use Standalone Text Encoder --- 
        # Text Projection Layer (Frozen - from original model)
        if not hasattr(self.base_reward_model.blip, 'text_proj'):
             raise AttributeError("Cannot find '.blip.text_proj'.")
        self.text_proj = self.base_reward_model.blip.text_proj
        self.text_feature_dim = self.text_proj.out_features # This is the *projected* dim
        print(f"Using original Text Projection Layer. Projected Text dim: {self.text_feature_dim}")

        # Load Standalone Text Encoder (Frozen)
        print(f"Loading standalone text encoder: {text_model_name}")
        self.standalone_text_encoder = AutoModel.from_pretrained(text_model_name)
        self.standalone_text_encoder.eval() # Set to eval mode
        # --- End EDIT --- 

        # Reward Head MLP (Frozen for now)
        if not hasattr(self.base_reward_model, 'mlp'):
             raise AttributeError("Cannot find 'mlp' reward head.")
        self.reward_head = self.base_reward_model.mlp
        try: first_layer_of_mlp = self.reward_head.layers[0]; self.reward_head_in_dim = first_layer_of_mlp.in_features
        except Exception: raise AttributeError("Cannot get input dim for 'mlp' reward head.")
        print(f"Identified reward head: 'mlp' (expects input dim: {self.reward_head_in_dim})")

        # --- Image Preprocessing Method (Crucial!) ---
        if not hasattr(self.base_reward_model, 'preprocess'):
             raise AttributeError("Cannot find 'preprocess' method on base_reward_model.")
        self.preprocess_image = self.base_reward_model.preprocess
        print("Found 'preprocess' method on base model.")
        # --- End Preprocessing --- 

        # --- Trainable Components ---
        self.timestep_embedding = TimestepEmbedding(timestep_dim)
        print(f"Timestep Embedding dim: {timestep_dim}")

        total_fused_dim = self.vision_feature_dim + timestep_dim + self.text_feature_dim
        intermediate_fusion_dim = self.vision_feature_dim
        self.fusion_layer = nn.Linear(total_fused_dim, intermediate_fusion_dim)
        print(f"Fusion layer input dim: {total_fused_dim}, output dim: {intermediate_fusion_dim}")

        self.fusion_to_reward_proj = nn.Linear(intermediate_fusion_dim, self.reward_head_in_dim)
        print(f"Added projection layer: {intermediate_fusion_dim} -> {self.reward_head_in_dim}")

        # --- PEFT Configs --- 
        # Vision Encoder PEFT
        peft_config_vision = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["qkv", "proj"] # Common targets for ViT
        )
        print("Applying LoRA to Vision Encoder...")
        self.vision_encoder = inject_adapter_in_model(peft_config_vision, self.vision_encoder)
        
        # --- Freeze Parameters --- 
        # Freeze base model first
        for name, param in self.base_reward_model.named_parameters():
            param.requires_grad = False
            
        # Freeze the standalone text encoder
        for param in self.standalone_text_encoder.parameters():
             param.requires_grad = False

        # --- Unfreeze selected components --- 
        # Unfreeze custom layers 
        for param in self.timestep_embedding.parameters(): param.requires_grad = True
        for param in self.fusion_layer.parameters(): param.requires_grad = True
        for param in self.fusion_to_reward_proj.parameters(): param.requires_grad = True
        
        # Unfreeze LoRA params in vision_encoder (already done by PEFT but explicit)
        for name, param in self.vision_encoder.named_parameters():
             if 'lora' in name:
                  param.requires_grad = True
                  
        # Unfreeze Reward Head MLP (it was frozen by the base_reward_model loop above)
        print("Explicitly unfreezing reward_head MLP parameters...")
        for param in self.reward_head.parameters():
             param.requires_grad = True

        # Keep original_reward_model used for scoring target frozen
        for param in self.original_reward_model.parameters(): param.requires_grad = False

        print("PEFTImageReward model configuration complete (Vision Encoder adapted, Standalone Text Encoder).") # Updated message
        print(f"Total Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, intermediate_image, timestep, input_ids, attention_mask):
        """ Forward pass using preprocessed intermediate image, timestep, and text. Adapts Vision Encoder. """
        device = next(self.parameters()).device 
        batch_size = intermediate_image.shape[0]

        # --- Debug --- 
        # print(f"[DEBUG] Intermediate image input shape: {intermediate_image.shape}") # Input shape is [B, C, H, W]
        # --- End Debug --- 
        
        # 1. Process Image Batch
        # Ensure input is on the correct device
        img_batch_gpu = intermediate_image.to(device)

        # Remove the extra inner dimension if present -> Shape [B, C, H, W]
        if img_batch_gpu.dim() == 5 and img_batch_gpu.shape[1] == 1:
            img_batch_gpu = img_batch_gpu.squeeze(1)
        elif img_batch_gpu.dim() != 4:
            raise ValueError(f"Unexpected image tensor dimensions: {img_batch_gpu.shape}. Expected 4 (B, C, H, W) or 5 (B, 1, C, H, W).")

        # Preprocess images individually as preprocess expects PIL
        processed_images_list = []
        batch_size = img_batch_gpu.shape[0] # Get batch size here
        for i in range(batch_size):
            single_img_tensor = img_batch_gpu[i] # Shape is likely C, H, W or H, W, C
            try:
                # --- Ensure tensor is CHW before converting ---
                if single_img_tensor.ndim == 3 and single_img_tensor.shape[-1] == 3: # Check if it's HWC (e.g., 512, 512, 3)
                    single_img_tensor_chw = single_img_tensor.permute(2, 0, 1).to(dtype=torch.float32)
                elif single_img_tensor.ndim == 3 and single_img_tensor.shape[0] == 3: # Check if it's already CHW (e.g., 3, 512, 512)
                    single_img_tensor_chw = single_img_tensor.to(dtype=torch.float32)
                else:
                    print(f"Warning: Unexpected single image tensor shape: {single_img_tensor.shape}. Skipping preprocessing for image {i}.")
                    # Add a dummy tensor matching the expected *processed* shape [3, 224, 224]
                    processed_images_list.append(torch.zeros((3, 224, 224), dtype=torch.float32, device=device))
                    continue

                # Convert tensor [C, H, W] to PIL Image
                single_img_pil = to_pil_image(single_img_tensor_chw.cpu())

                # Apply the original model's preprocess method (expects PIL)
                # preprocess_image likely returns a [3, 224, 224] tensor
                processed_single_tensor = self.preprocess_image(single_img_pil)

                # Append the processed tensor (ensure it's on the correct device)
                processed_images_list.append(processed_single_tensor.to(device))

            except Exception as e:
                print(f"Error during individual image preprocessing (index {i}): {e}")
                print(f" -> Original single image tensor shape: {single_img_tensor.shape}, dtype: {single_img_tensor.dtype}")
                # Handle error: skip this image by adding a dummy tensor
                print(f" -> Skipping image {i} due to error.")
                processed_images_list.append(torch.zeros((3, 224, 224), dtype=torch.float32, device=device))
                # Optionally re-raise if you want to stop on error: raise

        # Check if any images were successfully processed
        if not any(p.sum() != 0 for p in processed_images_list): # Basic check if all are dummies
             print("Warning: Preprocessing might have failed for all images in the batch (all seem to be zero tensors).")
             # Decide how to handle, maybe return zeros? For now, proceed cautiously.

        # Stack the processed tensors back into a batch -> [B, 3, 224, 224]
        try:
            processed_intermediate_image = torch.stack(processed_images_list, dim=0)
        except Exception as stack_e:
            print(f"Error stacking processed images: {stack_e}")
            # Print shapes for debugging
            # for j, p_img in enumerate(processed_images_list): print(f" Img {j} shape: {p_img.shape}")
            raise RuntimeError("Could not stack processed images.") from stack_e

        # 2. Get Image Features from PEFT-adapted Vision Encoder
        vision_outputs = self.vision_encoder(processed_intermediate_image)
        image_features = vision_outputs[:, 0] # Use CLS token output -> [B, D_vis]
        # print(f"[DEBUG FORWARD] image_features.requires_grad: {image_features.requires_grad}") 

        # --- Use Standalone Text Encoder ---
        # 3. Process Text Batch (using standalone frozen encoder and original projection)
        with torch.no_grad():
             # input_ids/attention_mask arrive as [B, 1, SeqLen] from dataloader/saving,
             # but text encoder expects [B, SeqLen]. Squeeze the middle dimension.
             if input_ids.ndim == 3 and input_ids.shape[1] == 1:
                 input_ids_squeezed = input_ids.squeeze(1)
                 # print(f"Squeezed input_ids shape: {input_ids_squeezed.shape}")
             else:
                  input_ids_squeezed = input_ids # Assume correct shape if not [B, 1, SeqLen]

             if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
                  attention_mask_squeezed = attention_mask.squeeze(1)
                  # print(f"Squeezed attention_mask shape: {attention_mask_squeezed.shape}")
             else:
                   attention_mask_squeezed = attention_mask # Assume correct shape

             standalone_text_output = self.standalone_text_encoder.to(device)(\
                 input_ids=input_ids_squeezed.to(device),\
                 attention_mask=attention_mask_squeezed.to(device),\
                 return_dict=True\
             )\
             # Use CLS token from the standalone encoder output -> [B, D_text_raw]
             # Use pooler_output for RoBERTa-style models
             text_features = standalone_text_output.pooler_output

             # Pass through the original model's frozen projection layer -> [B, D_text_proj]
             projected_text_features = self.text_proj.to(device)(text_features)
        # --- End Standalone Text Encoder ---

        # 4. Get Timestep Embedding (trainable)
        # timestep should be [B]
        timestep_emb = self.timestep_embedding(timestep.to(device)) # -> [B, D_ts]
        # print(f"[DEBUG FORWARD] timestep_emb.requires_grad: {timestep_emb.requires_grad}") 

        # 5. Fuse Features (trainable layers)
        # Concatenate along the feature dimension (dim=1)
        combined_features = torch.cat([
            image_features,                 # [B, D_vis]
            projected_text_features,        # [B, D_text_proj]
            timestep_emb                    # [B, D_ts]
        ], dim=1) # Result shape [B, D_vis + D_text_proj + D_ts]
        fused_features = self.fusion_layer(combined_features) # -> [B, D_fusion_out]
        # print(f"[DEBUG FORWARD] fused_features.requires_grad: {fused_features.requires_grad}") 

        # 6. Project to Reward Head Input Dimension (trainable layer)
        fused_features_projected = self.fusion_to_reward_proj(fused_features) # -> [B, D_mlp_in]
        # print(f"[DEBUG FORWARD] fused_features_projected.requires_grad: {fused_features_projected.requires_grad}") 

        # 7. Pass through Reward Head MLP (now trainable)
        reward_score = self.reward_head(fused_features_projected) # -> [B, 1]
        # print(f"[DEBUG FORWARD] reward_score.requires_grad: {reward_score.requires_grad}") 

        return reward_score # Shape [B, 1]

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, timestep):
        half_dim = self.dim // 2; timestep = timestep.float() / 1000.0
        freqs = torch.exp(-torch.arange(half_dim, device=timestep.device) * torch.log(torch.tensor(10000.0)) / half_dim)
        timestep = timestep.view(-1, 1); freqs = freqs.view(1, -1); args = timestep * freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding) 