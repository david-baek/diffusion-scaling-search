import json
import os
import itertools
import numpy as np
from tqdm import tqdm
import torch
import random
import optuna
import sys
import argparse
from utils import prompt_to_filename

optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""

    print(f"\n\n=== Starting Trial #{trial.number} ===")

    # Define hyperparameter search space
    config = {
        "pipeline_call_args": {"height": 512, "width": 512},
        "verifier_args": {
            "name": "ensemble",
            "choice_of_metric": "ensemble_score",
            "clip_model_path": "ViT-B/32"
        },
        "search_args": {
            "search_method": "rejection",
            "search_rounds": 5,
            "noise_scale": trial.suggest_float("noise_scale", 0.2, 1.0),
            "adaptation_strength": trial.suggest_float("adaptation_strength", 0.1, 0.7),
            "num_samples": 6
        },
        "batch_size_for_img_gen": 1,
        "use_low_gpu_vram": True,
        "torch_dtype": "fp16",
        "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"
    }

    # Print the hyperparameters being tested
    print("Current hyperparameters:")
    for param_name, param_value in trial.params.items():
        print(f"  {param_name}: {param_value}")

    result_dir = f"hpo/results/hpo_trial_{trial.number}"
    os.makedirs(result_dir, exist_ok=True)

    # Write config to file for experiment
    config_path = f"{result_dir}/hpo_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Load specific subset of prompts for HPO
    with open("prompts_drawbench.txt", "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
        # Use consistent subset for all HPO trials
        random.seed(1994)
        hpo_prompts = random.sample(all_prompts, 10)

    with open("prompts_drawbench.txt", "w") as f: # OVERRIDES DEFAULT !!!
        f.write("\n".join(hpo_prompts))

    # Save original command line arguments
    original_argv = sys.argv.copy()

    # Set up command line arguments for main.py
    sys.argv = [
        "main.py",
        f"--pipeline_config_path={config_path}",
        "--num_prompts=all"
    ]

    # Import and run main function
    from main import main as run_experiment
    try:
        run_experiment()  # Call main() with no arguments as it parses sys.argv
    except Exception as e:
        print(f"Error running experiment: {e}")
        return 0.0  # Return 0 score on failure
    finally:
        # Restore original command line arguments
        sys.argv = original_argv

    # Modified code to find the correct metrics file
    model_name = "sd-v1.5"
    verifier_name = "ensemble"
    choice_of_metric = "ensemble_score"

    output_base_dir = os.path.join(
        "output",
        model_name,
        verifier_name,
        choice_of_metric
    )

    # Find the most recent timestamp directory
    if not os.path.exists(output_base_dir):
        print(f"Output directory not found: {output_base_dir}")
        return 0.0

    timestamp_dirs = [d for d in os.listdir(output_base_dir)
                    if os.path.isdir(os.path.join(output_base_dir, d))]

    if not timestamp_dirs:
        print("No timestamp directories found")
        return 0.0

    # Get the most recent timestamp directory
    latest_timestamp = max(timestamp_dirs)
    metrics_file = os.path.join(output_base_dir, latest_timestamp, "all_nfe_data.json")

    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return 0.0

    # Calculate percentage improvements
    try:
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        max_rounds = config["search_args"]["search_rounds"]
        total_clip_percent_increase = 0
        total_aesthetic_percent_increase = 0
        clip_count = 0
        aesthetic_count = 0

        print(f"Trial {trial.number} - Calculating percent improvements:")

        for prompt in metrics_data:
            # Check if we have both baseline and final round data
            if "0" in metrics_data[prompt] and str(max_rounds) in metrics_data[prompt]:
                # Process CLIP score improvement
                try:
                    initial_clip = float(metrics_data[prompt]["0"]["clip_score"])
                    final_clip = float(metrics_data[prompt][str(max_rounds)]["clip_score"])

                    if initial_clip > 0:
                        clip_percent_increase = (final_clip - initial_clip) / initial_clip * 100
                    else:
                        clip_percent_increase = final_clip * 100

                    total_clip_percent_increase += clip_percent_increase
                    clip_count += 1

                    print(f"  Prompt: {prompt[:30]}...")
                    print(f"    CLIP: {initial_clip:.4f} -> {final_clip:.4f} ({clip_percent_increase:+.2f}%)")
                except (ValueError, KeyError) as e:
                    print(f"Error with CLIP metrics for prompt {prompt}: {e}")

                # Process aesthetic score improvement
                try:
                    initial_aesthetic = float(metrics_data[prompt]["0"]["aesthetic_score"])
                    final_aesthetic = float(metrics_data[prompt][str(max_rounds)]["aesthetic_score"])

                    if initial_aesthetic > 0:
                        aesthetic_percent_increase = (final_aesthetic - initial_aesthetic) / initial_aesthetic * 100
                    else:
                        aesthetic_percent_increase = final_aesthetic * 100

                    total_aesthetic_percent_increase += aesthetic_percent_increase
                    aesthetic_count += 1

                    print(f"    Aesthetic: {initial_aesthetic:.4f} -> {final_aesthetic:.4f} ({aesthetic_percent_increase:+.2f}%)")
                except (ValueError, KeyError) as e:
                    print(f"Error with aesthetic metrics for prompt {prompt}: {e}")

        # Calculate combined score - average of both percentage increases
        combined_score = 0
        metrics_used = 0

        if clip_count > 0:
            avg_clip_increase = total_clip_percent_increase / clip_count
            combined_score += avg_clip_increase
            metrics_used += 1
            print(f"Average CLIP score increase: {avg_clip_increase:.2f}%")

        if aesthetic_count > 0:
            avg_aesthetic_increase = total_aesthetic_percent_increase / aesthetic_count
            combined_score += avg_aesthetic_increase
            metrics_used += 1
            print(f"Average aesthetic score increase: {avg_aesthetic_increase:.2f}%")

        if metrics_used > 0:
            combined_score /= metrics_used
            print(f"Trial {trial.number} combined improvement score: {combined_score:.2f}%")

            # Save trial results for reference
            result_dir = f"results/hpo_trial_{trial.number}"
            os.makedirs(result_dir, exist_ok=True)
            with open(f"{result_dir}/improvement_summary.json", "w") as f:
                json.dump({
                    "clip_improvement": float(avg_clip_increase if clip_count > 0 else 0),
                    "aesthetic_improvement": float(avg_aesthetic_increase if aesthetic_count > 0 else 0),
                    "combined_score": float(combined_score),
                    "timestamp": latest_timestamp
                }, f, indent=4)

            return combined_score
        else:
            print(f"Trial {trial.number} failed: No valid metrics collected")
            return 0.0

    except Exception as e:
        print(f"Error processing metrics: {e}")
        traceback.print_exc()
        return 0.0

# Callback function to print intermediate results
def print_best_callback(study, frozen_trial):
    # Print whenever we find a new best value
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(f"\n>> New best value found: {study.best_value:.4f} (Trial {frozen_trial.number})")
        print(f">> Best parameters so far: {study.best_params}")

    # Print progress every trial
    completed = len(study.trials)
    total = study.user_attrs.get("total_trials", "unknown")
    print(f"\n>> Completed: {completed}/{total} trials")

def run_hyperparameter_tuning():
    # Create study with maximize direction (higher ensemble score is better)
    study = optuna.create_study(direction="maximize")
    n_trials = 20

    # Store total trials count for the callback
    study.set_user_attr("total_trials", n_trials)

    # Use the pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Number of trials to run before pruning starts
        n_warmup_steps=0,    # Number of steps to run before pruning starts
        interval_steps=1,    # Minimum interval between prunings
    )

    # Run optimization with callback for intermediate output
    print(f"Starting hyperparameter optimization with {n_trials} trials")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=86400,  # 24 hour timeout
        callbacks=[print_best_callback]
    )

    # Add custom JSON encoder for serializing study results
    def json_serializer(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Save best parameters
    with open("hpo/best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, default=json_serializer)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    print("\n=== Summary of trials ===")
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    return study.best_params

if __name__ == "__main__":
    run_hyperparameter_tuning()
