import ImageReward as reward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import traceback
import random # Added for random step selection
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, logging as hf_logging
from diffusers import FluxPipeline # Added FluxPipeline import

# Suppress tokenizer warnings about legacy behavior
hf_logging.set_verbosity_error()

# Add parent directory path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) if __file__ else '.', '..')))
from source.peft_image_reward_model import PEFTImageReward # Import model

def plot_loss_to_terminal(steps, losses, width=80, title="Training Loss per Step"):
    if not steps or not losses or len(steps) != len(losses): print("Plotting skipped: Invalid data."); return
    avg_loss_val = np.mean(losses[-50:]) if len(losses) > 0 else float('nan') # Avg last 50 or available
    print(f"\n--- {title} (Avg last 50: {avg_loss_val:.6f}) ---")
    try:
        plt.figure(figsize=(12, 4)); plt.plot(steps, losses, linestyle='-', color='b', alpha=0.7)
        plt.title(title); plt.xlabel('Training Step'); plt.ylabel('Batch Loss')
        plt.grid(True, alpha=0.5)
        min_positive_loss = min((l for l in losses if l > 1e-9), default=1e-9) # Avoid zero/negative for log
        if max(losses) / min_positive_loss > 50: plt.yscale('log'); plt.ylabel('Batch Loss (log scale)')
        plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); img = Image.open(buf); plt.close()
        aspect_ratio = img.width / img.height; new_height = min(30, int(width / aspect_ratio / 2))
        img = img.resize((width, new_height)).convert('L'); ascii_chars = ' .:-=+*#%@'; pixels = np.array(img)
        ascii_img = [''.join([ascii_chars[int(p * (len(ascii_chars) - 1) / 255)] for p in r]) for r in pixels]
        for row in ascii_img: print(row)
        print("--- End Loss Curve ---\n")
    except Exception as e: print(f"Failed to generate terminal plot: {e}\n{traceback.format_exc()}")

def train(
    model: PEFTImageReward, 
    flux_pipe: FluxPipeline,
    prompts_file="prompts.txt",
    batch_size=1,
    learning_rate=1e-4,
    image_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
    plot_every_n_steps=1,
    tokenizer_name="roberta-base",
    max_prompt_length=77
):
    # --- Ensure models are on the correct device (passed models might be on CPU) ---
    try:
        model.to(device)
        # Flux pipe should already be loaded with offload enabled, 
        # but we ensure its core components expected on GPU are there.
        # This might be implicitly handled by offload, but can be explicit if needed.
        # flux_pipe.to(device) # Typically enable_model_cpu_offload handles this
    except Exception as e:
        print(f"ERROR moving models passed as arguments to device '{device}': {e}")
        return None
    # --- End Ensure Device ---
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints directory: {checkpoint_dir}")

    # Load Tokenizer (Keep this here, it's fast)
    try: tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e: print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}"); return None
    print(f"Tokenizer '{tokenizer_name}' loaded.")

    # --- Load Prompts Directly ---
    prompts = []
    try:
        # Check paths for prompts file
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.' # Handle case where script is run directly
        paths_to_check = [
            prompts_file,
            os.path.join(script_dir, prompts_file),
            os.path.join(script_dir, '..', prompts_file)
        ]
        found_path = None
        for path in paths_to_check:
             if os.path.exists(path):
                  found_path = path
                  break
        if found_path is None:
             raise FileNotFoundError(f"Prompts file not found in checked paths: {paths_to_check}")
        prompts_file = found_path # Use the found path

        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
             raise ValueError(f"No valid prompts found in {prompts_file}")
        print(f"Loaded {len(prompts)} prompts directly from {prompts_file}")
    except Exception as e: 
        print(f"ERROR: Failed to load prompts: {e}\n{traceback.format_exc()}")
        return None
    dataset_size = len(prompts)
    num_steps = (dataset_size + batch_size - 1) // batch_size # Calculate steps based on batch size
    print(f"Data loaded. Size: {dataset_size}. Calculated Steps/Pass: {num_steps}")
    # --- End Load Prompts --- 

    # Setup Optimizer & Loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params: print("ERROR: No trainable params."); return None
    print(f"Optimizing {len(trainable_params)} params ({sum(p.numel() for p in trainable_params):,} total).")
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    
    # Metrics storage
    step_losses, global_steps_list = [], []; global_step_counter = 0
    print(f"Starting training for one pass ({num_steps} steps)... Plotting every {plot_every_n_steps} steps.")

    model.train(); total_loss_accum, batches_processed = 0.0, 0

    # The dataloader now yields batches of prompt strings
    # Loop directly over prompts, creating batches manually
    for i in range(0, dataset_size, batch_size):
        global_step_counter += 1
        prompts_text_batch = prompts[i:min(i + batch_size, dataset_size)]
        current_batch_size = len(prompts_text_batch) # Actual size of this batch
        if not prompts_text_batch: continue # Skip if somehow empty
        
        try:
            # --- Generate Images On-the-fly --- 
            batch_final_images_cpu = []
            batch_intermediate_images_cpu = []
            batch_timesteps_cpu = []
            num_inference_steps = 30 # Define inference steps

            # Process each prompt in the batch individually (simplest approach for now)
            for prompt_text in prompts_text_batch:
                random_step = random.randint(1, num_inference_steps - 1)
                captured_timestep = torch.tensor(random_step) # Keep on CPU for now
                intermediate_latents = None

                # Define callback within the loop to capture intermediate latents
                def capture_callback(pipe, step, timestep, callback_kwargs):
                    nonlocal intermediate_latents
                    if step == random_step:
                        # Capture latents (will be on pipeline's compute device, likely GPU)
                        intermediate_latents = callback_kwargs["latents"].detach().clone()
                    return callback_kwargs

                # Generate image using the single pipeline instance
                with torch.no_grad():
                    # Note: For batch size > 1, ideally call pipeline once per batch.
                    # Current loop processes prompts individually, might be inefficient.
                    # Consider batching the flux_pipe call if performance is critical.
                    output = flux_pipe(
                        prompt=prompt_text, height=image_size, width=image_size,
                        guidance_scale=3.5, num_inference_steps=num_inference_steps,
                        max_sequence_length=512, output_type="pt", # Get PyTorch tensors
                        callback_on_step_end=capture_callback,
                        callback_on_step_end_tensor_inputs=["latents"]
                    )
                # Need to ensure final_image_tensor has batch dim if batch_size > 1 was intended here
                # If output.images is already [B, C, H, W] for batch prompts, adjust indexing
                final_image_tensor = output.images[0] # Assumes output.images is [C, H, W] for single prompt

                # Decode intermediate latents (if captured)
                intermediate_image_tensor = None
                if intermediate_latents is not None:
                    # Move latents to target device (necessary if offloading happened)
                    latents_to_process = intermediate_latents.to(device, dtype=flux_pipe.dtype)

                    # Unpack/Reshape (copied from old dataloader)
                    latent_channels = flux_pipe.vae.config.latent_channels
                    latent_height = image_size // 8; latent_width = image_size // 8
                    expected_shape = (1, latent_channels, latent_height, latent_width)
                    numel_expected = 1 * latent_channels * latent_height * latent_width
                    if hasattr(flux_pipe, "_unpack_latents"): # Check if unpack method exists
                        try:
                            vae_scale_factor = 2 ** (len(flux_pipe.vae.config.block_out_channels) - 1)
                            latents_to_process = flux_pipe._unpack_latents(latents_to_process, image_size, image_size, vae_scale_factor)
                        except Exception as unpack_e:
                            if latents_to_process.numel() == numel_expected: latents_to_process = latents_to_process.reshape(expected_shape)
                            else: raise ValueError(f"Cannot unpack/reshape {latents_to_process.shape} to {expected_shape}.") from unpack_e
                    elif latents_to_process.numel() == numel_expected:
                        latents_to_process = latents_to_process.reshape(expected_shape)
                    else: raise ValueError(f"Cannot reshape {latents_to_process.shape} to {expected_shape}.")

                    # Inverse Scale & Shift 
                    scaling_factor = flux_pipe.vae.config.scaling_factor
                    shift_factor = getattr(flux_pipe.vae.config, "shift_factor", 0.0)
                    latents_for_decode = latents_to_process / scaling_factor + shift_factor
                    
                    # VAE Decode
                    with torch.no_grad():
                         # Ensure VAE is on the correct device (may need .to(device))
                        decoded_output = flux_pipe.vae.decode(latents_for_decode.to(flux_pipe.vae.dtype), return_dict=False)
                        decoded_image_tensor = decoded_output[0]
                    
                    # Post-process
                    intermediate_image = flux_pipe.image_processor.postprocess(decoded_image_tensor, output_type="pt")
                    if isinstance(intermediate_image, list): intermediate_image = intermediate_image[0]
                    if intermediate_image.dim() == 4 and intermediate_image.shape[0] == 1: intermediate_image = intermediate_image.squeeze(0)
                    intermediate_image_tensor = intermediate_image # Shape [C, H, W]

                else:
                     print(f"W: Intermediate latents not captured for step {global_step_counter}. Using final image.")
                     intermediate_image_tensor = final_image_tensor.clone()
                
                # Append results for this prompt to the batch lists (move to CPU)
                batch_final_images_cpu.append(final_image_tensor.cpu())
                batch_intermediate_images_cpu.append(intermediate_image_tensor.cpu())
                batch_timesteps_cpu.append(captured_timestep.cpu()) # Timestep was already CPU
                
            # --- End Generate Images On-the-fly ---
            
            # --- Create Batches from generated data ---
            if not batch_final_images_cpu:
                print(f"W: Skipping step {global_step_counter}, no images generated for batch."); continue

            final_images_cpu = torch.stack(batch_final_images_cpu) # Shape [B, C, H, W]
            intermediate_images_cpu = torch.stack(batch_intermediate_images_cpu) # Shape [B, C, H, W]
            timesteps_cpu = torch.stack(batch_timesteps_cpu) # Shape [B]
            # prompts_text is already prompts_text_batch (List[str])
            prompts_text = prompts_text_batch 
            # --- End Create Batches --- 

            # Tokenize Prompts (prompts_text is now the batch list)
            tokenized_prompts = tokenizer(
                prompts_text, padding="max_length", truncation=True,
                max_length=max_prompt_length, return_tensors="pt"
            )
            # Move necessary tokens to device
            input_ids = tokenized_prompts["input_ids"].to(device)
            attention_mask = tokenized_prompts["attention_mask"].to(device)

            # --- EDIT: Move FINAL images and timesteps to device for model input ---
            final_images = final_images_cpu.to(device)
            timesteps = timesteps_cpu.to(device)
            # intermediate_images are now only needed for calculating the loss target? No, target uses final PIL.
            # So intermediate_images_cpu aren't used further in this setup.

            # Convert final images (already on CPU) to PIL for score method (target reward)
            final_images_pil = [to_pil_image(img.to(torch.float32)) for img in final_images_cpu]

            # Target rewards using the original model's score method (on FINAL images)
            with torch.no_grad():
                model.original_reward_model.to(device).eval()
                target_rewards_output = model.original_reward_model.score(prompts_text, final_images_pil)
                if target_rewards_output is None: print(f"W: Skip step {global_step_counter}, None target"); continue

                # --- EDIT: Handle scalar or list return from score ---
                # Ensure we have a list of scores before creating the tensor
                if isinstance(target_rewards_output, list):
                    target_rewards_list = target_rewards_output
                else: # Assume scalar if not list
                    target_rewards_list = [target_rewards_output] 

                # Convert list (even if single item) to tensor and add dim
                target_rewards = torch.tensor(target_rewards_list, device=device, dtype=torch.float32).unsqueeze(1) #[B, 1]
                # --- End EDIT ---

            # Corrected Call: Pass INTERMEDIATE image tensor, timesteps, and text tokens to model
            predicted_rewards = model(intermediate_images_cpu, timesteps_cpu, input_ids, attention_mask)

            if predicted_rewards is None: print(f"W: Skip step {global_step_counter}, None prediction"); continue
            if predicted_rewards.shape != target_rewards.shape: 
                print(f"W: Skip step {global_step_counter}, Shape mismatch. Predicted: {predicted_rewards.shape}, Target: {target_rewards.shape}"); 
                continue

            loss = loss_fn(predicted_rewards, target_rewards)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            batch_loss = loss.item(); total_loss_accum += batch_loss; batches_processed += 1
            # Note: global_steps_list uses the batch index, not individual image index
            step_losses.append(batch_loss); global_steps_list.append(global_step_counter)

            # Plotting Condition
            if global_step_counter % plot_every_n_steps == 0:
                print(f"\n--- Plotting at Step {global_step_counter} ---")
                print(f"  Step {global_step_counter}/{num_steps}, Batch Loss: {batch_loss:.6f}, Running Avg Loss: {total_loss_accum/batches_processed:.6f}")
                plot_loss_to_terminal(global_steps_list, step_losses)

        except Exception as batch_e: print(f"\nERROR Step {global_step_counter}: {batch_e}\n{traceback.format_exc()}\nSkipping..."); continue

    # Final wrap-up
    avg_loss_final = total_loss_accum / batches_processed if batches_processed > 0 else float('nan')
    print(f"\n--- Training Pass Finished. Avg Loss: {avg_loss_final:.6f} ---")
    plot_loss_to_terminal(global_steps_list, step_losses, title="Final Training Loss per Step")
    checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint_onepass.pt")
    try:
        checkpoint_data = {'steps': global_step_counter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'final_avg_loss': avg_loss_final}
        torch.save(checkpoint_data, checkpoint_path); print(f"Final checkpoint saved: {checkpoint_path}")
    except Exception as save_e: print(f"ERROR saving final checkpoint: {save_e}")
    print(f"Model processed {dataset_size} unique prompts in {global_step_counter} steps.")
    return model

if __name__ == "__main__":
    # --- This block now needs to perform the setup and model loading --- 
    print("--- Running Train Script Directly (Setup & Training) ---")
    
    # --- Configuration --- 
    prompts_file_path = "prompts.txt"
    batch_size=1
    plot_every_n_steps=1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name="roberta-base"
    # Add other necessary configs like lr, image_size, checkpoint_dir if needed by train()
    learning_rate = 1e-4 
    image_size = 512
    checkpoint_dir = "checkpoints"
    max_prompt_length = 77
    
    # --- Create Dummy Prompts --- 
    if not os.path.exists(prompts_file_path):
        print(f"INFO: {prompts_file_path} not found. Creating a dummy file with one prompt.")
        with open(prompts_file_path, 'w') as f:
            f.write("a photograph of an astronaut riding a horse\n")

    # --- Load Models (SLOW PART) --- 
    loaded_model = None
    loaded_flux_pipe = None

    # Load PEFT Model
    try: 
        print("Loading PEFTImageReward model...")
        # Load initially to CPU potentially to manage memory during sequential load
        loaded_model = PEFTImageReward(text_model_name=tokenizer_name) 
        print(f"PEFTImageReward model loaded initially.")
    except Exception as e: 
        print(f"ERROR: Failed to init model: {e}\n{traceback.format_exc()}"); exit()
    
    # Temporarily Move Reward Model to CPU (redundant if loaded to CPU, but ensures state)
    try:
        print("Ensuring PEFTImageReward model is on CPU...")
        loaded_model.to('cpu')
        print("Attempting to clear CUDA cache before loading Flux pipeline...")
        torch.cuda.empty_cache() 
    except Exception as e:
        print(f"Warning: Error moving model to CPU or clearing cache: {e}")
        
    # Load Flux Pipeline 
    flux_model_name="black-forest-labs/FLUX.1-dev"
    try:
        print(f"Loading Flux pipeline ({flux_model_name}) to device: {device}...")
        pipeline_dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
        print(f"Using dtype: {pipeline_dtype}")
        loaded_flux_pipe = FluxPipeline.from_pretrained(
            flux_model_name,
            torch_dtype=pipeline_dtype,
        ).to(device)
        loaded_flux_pipe.enable_model_cpu_offload()
        print("Flux pipeline loaded and configured with model CPU offload.")
    except Exception as e:
        print(f"ERROR: Failed to load Flux pipeline: {e}\n{traceback.format_exc()}"); exit()

    # Move Reward Model back to GPU 
    try:
        print("Moving PEFTImageReward model to GPU...")
        loaded_model.to(device)
        print("PEFTImageReward model on GPU.")
    except Exception as e:
        print(f"ERROR: Failed to move PEFTImageReward model to GPU: {e}\n{traceback.format_exc()}"); exit()
    
    # --- Call Training Function --- 
    print("\n--- Starting Training --- ")
    train(
        model=loaded_model,
        flux_pipe=loaded_flux_pipe,
        prompts_file=prompts_file_path, 
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        device=device,
        checkpoint_dir=checkpoint_dir,
        plot_every_n_steps=plot_every_n_steps,
        tokenizer_name=tokenizer_name,
        max_prompt_length=max_prompt_length
    ) 
    print("--- Training Script Finished ---")

