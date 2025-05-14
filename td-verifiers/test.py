import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline
import random
import os
import argparse
from huggingface_hub import login
from PIL import Image
import numpy as np
import traceback
import ImageReward as RM # Added ImageReward import

def plot_images_side_by_side(image1, title1, image2, title2):
    """Plots two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert tensors to PIL Images if necessary
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().permute(1, 2, 0).numpy()
        image1 = (image1 * 255).astype(np.uint8)
        image1 = Image.fromarray(image1)
        
    if isinstance(image2, torch.Tensor):
        image2 = image2.cpu().permute(1, 2, 0).numpy()
        image2 = (image2 * 255).astype(np.uint8)
        image2 = Image.fromarray(image2)

    axes[0].imshow(image1)
    axes[0].set_title(title1, fontsize=10) # Adjusted fontsize for potentially longer titles
    axes[0].axis('off')
    
    axes[1].imshow(image2)
    axes[1].set_title(title2, fontsize=10) # Adjusted fontsize
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("generated_plot.png") # Save the figure
    print("Plot saved to generated_plot.png")
    plt.show()

def generate_and_decode_latent(hf_token=None, image_size=512, num_inference_steps=50, prompt=""):
    """
    Generates an image with FLUX.1-dev, captures an intermediate latent, 
    decodes it, scores both images, and plots them.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and device == "cuda" else torch.float32

    # --- Authentication ---
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token provided. You might encounter authentication errors.")
        print("Please set the HF_TOKEN environment variable or pass --token.")
        print("Get your token from: https://huggingface.co/settings/tokens")

    # --- Load FLUX Pipeline ---
    print(f"Loading FLUX.1-dev pipeline (this might take a while)...")
    pipeline = None
    try:
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch_dtype,
        )
        if device == "cuda":
            print("CUDA device detected. Enabling model CPU offload to potentially save memory.")
            pipeline.enable_model_cpu_offload() 
            pipeline.to(device) 
        else: 
            print("CUDA not available or not primary. Enabling model CPU offload for the pipeline.")
            pipeline.enable_model_cpu_offload()
        print("FLUX.1-dev pipeline loaded.")
    except Exception as e:
        print(f"Error loading FLUX.1-dev pipeline: {e}")
        traceback.print_exc()
        print("Make sure you have a valid Hugging Face token with access to this model.")
        print("And that 'diffusers', 'transformers', 'accelerate' are installed.")
        return

    # --- Load ImageReward Model ---
    reward_model = None
    print("Loading ImageReward model...")
    try:
        # Ensure ImageReward is loaded to the same device if it doesn't handle it internally
        reward_model = RM.load("ImageReward-v1.0") # device=device argument might be supported
        if device == "cuda": # Explicitly move if not handled by load
            reward_model.to(device)
        reward_model.eval() # Set to evaluation mode
        print("ImageReward model loaded successfully.")
    except Exception as e:
        print(f"Error loading ImageReward model: {e}")
        traceback.print_exc()
        print("ImageReward scoring will be skipped.")
        reward_model = None


    # --- Prepare for image generation and capturing intermediate latent ---
    intermediate_latent_image = None
    captured_timestep_value = None
    final_image_score = None
    intermediate_image_score = None # Must be nonlocal for callback
    
    capture_step = random.randint(1, num_inference_steps - 1) 
    print(f"Will attempt to capture latent at inference step: {capture_step}")

    # prompt, image_size, reward_model are accessible here due to closure
    def callback_fn(pipe, step, timestep, callback_kwargs):
        nonlocal intermediate_latent_image, captured_timestep_value, intermediate_image_score 
        current_latents_from_callback = callback_kwargs["latents"].to(pipe.device)

        if step == capture_step:
            print(f"Callback triggered at step {step}, timestep {timestep}")
            with torch.no_grad():
                latents_to_process_for_vae = None
                if hasattr(pipe, '_unpack_latents'):
                    try:
                        vae_scale_factor = pipe.vae_scale_factor 
                        print(f"Attempting to unpack latents with shape: {current_latents_from_callback.shape} using image_size: {image_size} and vae_scale_factor: {vae_scale_factor}")
                        unpacked_latents = pipe._unpack_latents(current_latents_from_callback, image_size, image_size, vae_scale_factor)
                        print(f"Unpacked latents shape: {unpacked_latents.shape}")
                        latents_to_process_for_vae = unpacked_latents
                    except Exception as unpack_e:
                        print(f"Warning: Failed to unpack latents using pipe._unpack_latents: {unpack_e}")
                        traceback.print_exc()
                        latents_to_process_for_vae = current_latents_from_callback
                else:
                    print("Warning: pipe._unpack_latents method not found. Proceeding with original latents.")
                    latents_to_process_for_vae = current_latents_from_callback

                latents_for_vae_decode = latents_to_process_for_vae.to(pipe.vae.device, dtype=pipe.vae.dtype)
                scaling_factor = getattr(pipe.vae.config, 'scaling_factor', None)
                shift_factor = getattr(pipe.vae.config, 'shift_factor', 0.0) 
                if scaling_factor is not None:
                    print(f"Applying VAE scaling_factor: {scaling_factor} and shift_factor: {shift_factor}")
                    latents_for_vae_decode = (latents_for_vae_decode / scaling_factor) + shift_factor
                else:
                    print("Warning: VAE config.scaling_factor not found.")
                
                print(f"Decoding latents with shape: {latents_for_vae_decode.shape}, device: {latents_for_vae_decode.device}, dtype: {latents_for_vae_decode.dtype}")
                decoded_output = pipe.vae.decode(latents_for_vae_decode, return_dict=False)
                image_tensor = decoded_output[0]
                print(f"Decoded image tensor shape: {image_tensor.shape}")
                if image_tensor.ndim == 3: 
                    image_tensor = image_tensor.unsqueeze(0) 
                pil_images = pipe.image_processor.postprocess(image_tensor, output_type="pil")

                if pil_images and len(pil_images) > 0:
                    intermediate_latent_image = pil_images[0]
                    captured_timestep_value = timestep.item() if torch.is_tensor(timestep) else timestep
                    print(f"Captured and decoded intermediate latent image at step {step}.")
                    if reward_model:
                        try:
                            print(f"Scoring intermediate image at step {step} (Prompt: '{prompt}')...")
                            current_prompt_list = [prompt] if isinstance(prompt, str) else prompt
                            # ImageReward.score expects a list of prompts and a list of PIL images
                            score_data = reward_model.score(current_prompt_list, [intermediate_latent_image])
                            if isinstance(score_data, list) and len(score_data) > 0:
                                intermediate_image_score = score_data[0]
                                print(f"Intermediate image score at step {step}: {intermediate_image_score:.4f}")
                            else: # Handle scalar or unexpected return
                                intermediate_image_score = float(score_data) # Try to convert
                                print(f"Intermediate image score (adjusted) at step {step}: {intermediate_image_score:.4f}")

                        except Exception as score_e:
                            print(f"Error scoring intermediate image: {score_e}")
                            traceback.print_exc()
                            intermediate_image_score = None # Ensure it's None on error
        return callback_kwargs

    # --- Generate Image ---
    final_image = None
    print(f"Generating image with prompt: '{prompt}' (Size: {image_size}x{image_size}, Steps: {num_inference_steps})")
    try:
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0, 
                callback_on_step_end=callback_fn,
                callback_on_step_end_tensor_inputs=["latents"], 
            )
        final_image = output.images[0]
        print("Image generation complete.")
        
        # Score final image
        if reward_model and final_image:
            try:
                print(f"Scoring final image (Prompt: '{prompt}')...")
                current_prompt_list = [prompt] if isinstance(prompt, str) else prompt
                score_data = reward_model.score(current_prompt_list, [final_image])
                if isinstance(score_data, list) and len(score_data) > 0:
                    final_image_score = score_data[0]
                    print(f"Final image score: {final_image_score:.4f}")
                else: # Handle scalar or unexpected return
                    final_image_score = float(score_data)
                    print(f"Final image score (adjusted): {final_image_score:.4f}")
            except Exception as score_e:
                print(f"Error scoring final image: {score_e}")
                traceback.print_exc()
                final_image_score = None

    except Exception as e:
        print(f"Error during image generation: {e}")
        traceback.print_exc()
        return # Exit if generation fails

    # --- Plot Results ---
    if final_image and intermediate_latent_image:
        title1 = f"Final Image"
        if final_image_score is not None:
            title1 += f"\nScore: {final_image_score:.4f}"
        
        title2 = f"Decoded Latent (Step {capture_step}, T {captured_timestep_value:.0f})"
        if intermediate_image_score is not None:
            title2 += f"\nScore: {intermediate_image_score:.4f}"
            
        plot_images_side_by_side(final_image, title1, intermediate_latent_image, title2)

    elif final_image:
        title1 = "Final Image"
        if final_image_score is not None:
            title1 += f" (Score: {final_image_score:.4f})"
        print("Final image generated, but intermediate latent was not captured.")
        final_image.show(title=title1) # Show with score if available
    else:
        print("Image generation failed or no images were produced.")

    # Ensure scores are printed if not included in plot or if plotting fails
    print(f"--- Summary of Scores ---")
    if final_image_score is not None:
        print(f"Final Image Score: {final_image_score:.4f}")
    else:
        print(f"Final Image Score: Not available")
    
    if intermediate_image_score is not None:
        print(f"Intermediate Image (Step {capture_step}) Score: {intermediate_image_score:.4f}")
    else:
        print(f"Intermediate Image (Step {capture_step}) Score: Not available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FLUX.1-dev latent decoding and ImageReward scoring.")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token for authentication.")
    parser.add_argument("--prompt", type=str, default="A photo of a cat wearing a small hat", help="Prompt for image generation.")
    parser.add_argument("--size", type=int, default=512, help="Image size (square).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps (e.g., 20-50).")
    
    args = parser.parse_args()

    if not args.token and not os.environ.get("HF_TOKEN_PATH"): # Also check for token path if direct token not given
        print("Error: Hugging Face token is required. Set HF_TOKEN, HF_TOKEN_PATH, or use --token argument.")
        parser.print_help()
    else:
        generate_and_decode_latent(
            hf_token=args.token, # Will be None if not provided, then os.environ.get is used
            image_size=args.size,
            num_inference_steps=args.steps,
            prompt=args.prompt
        ) 