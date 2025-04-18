import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm
import tempfile
from diffusers.utils import export_to_video

from utils import (
    generate_neighbors,
    prompt_to_filename,
    get_noises,
    TORCH_DTYPE_MAP,
    get_latent_prep_fn,
    parse_cli_args,
    serialize_artifacts,
    MODEL_NAME_MAP,
    prepare_video_frames,
)
from verifiers import SUPPORTED_VERIFIERS
from search_algorithms import (
    RandomSearch,
    ZeroOrderSearch,
    EvolutionarySearch,
    RejectionSamplingSearch,
)

from drawbench_eval import compute_single_clipscore

# Non-configurable constants
TOPK = 3  # Always selecting the top-1 noise for the next round
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds


def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
) -> dict:
    """
    For a given prompt, generate images using all provided noises in batches,
    score them with the verifier, and select the top-K noise.
    The images and JSON artifacts are serialized via `serialize_artifacts`.
    """
    use_low_gpu_vram = config.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config.get("batch_size_for_img_gen", 1)
    verifier_args = config.get("verifier_args")
    choice_of_metric = verifier_args.get("choice_of_metric", None)
    verifier_to_use = verifier_args.get("name", "gemini")
    search_args = config.get("search_args", None)

    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": None,
        "best_noise": None,
        "best_score": None,
        "top10_noise": [],
        "choice_of_metric": choice_of_metric,
        "best_img_path": "",
        "acceptance_rate": 0.0  # Add default value
    }

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    images_info = []  # Will collect (seed, noise, image, filename) tuples for serialization.
    prompt_filename = prompt_to_filename(prompt)

    # Convert the noises dictionary into a list of (seed, noise) tuples.
    noise_items = list(noises.items())

    # Process the noises in batches.
    # TODO: find better way
    extension_to_use = "png"
    if "LTX" in pipe.__class__.__name__:
        extension_to_use = "mp4"
    elif "Wan" in pipe.__class__.__name__:
        extension_to_use = "mp4"
    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i : i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(root_dir, f"{prompt_filename}_i@{search_round}_s@{seed}.{extension_to_use}")
            for seed in seeds_batch
        ]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {list(seeds_batch)}.")

        # Create a batched prompt list and stack the latents.
        batched_prompts = [prompt] * len(noises_batch)
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)

        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, **config["pipeline_call_args"])
        if hasattr(batch_result, "images"):
            batch_images = batch_result.images
        elif hasattr(batch_result, "frames"):
            batch_images = [vid for vid in batch_result.frames]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        # Collect the images and corresponding info.
        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            images_info.append((seed, noise, image, filename))

    export_args = config.get("export_args", {})
    # Prepare verifier inputs and perform inference.
    if isinstance(images_for_prompt[0], Image.Image):
        verifier_inputs = verifier.prepare_inputs(images=images_for_prompt, prompts=[prompt] * len(images_for_prompt))
    else:
        export_args = config.get("export_args", None) or {}
        if export_args:
            fps = export_args.get("fps", 24)
        else:
            fps = 24
        temp_vid_paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, vid in enumerate(images_for_prompt):
                vid_path = os.path.join(tmpdir, f"{idx}.mp4")
                export_to_video(vid, vid_path, fps=fps)
                temp_vid_paths.append(vid_path)

            verifier_inputs = []
            for vid_path in temp_vid_paths:
                frames = prepare_video_frames(vid_path)
                verifier_inputs.append(verifier.prepare_inputs(images=frames, prompts=[prompt] * len(frames)))

    print("Scoring with the verifier.")
    outputs = verifier.score(inputs=verifier_inputs)
    for o in outputs:
        assert choice_of_metric in o, o.keys()

    assert (
        len(outputs) == len(images_for_prompt)
    ), f"Expected len(outputs) to be same as len(images_for_prompt) but got {len(outputs)=} & {len(images_for_prompt)=}"

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Merge verifier outputs with noise info.
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    def f(x):
        # If the verifier output is a dict, assume it contains a "score" key.
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    # Add rejection sampling search filtering
    search_method = search_args.get("search_method", "random")
    if search_method == "rejection":
        acceptance_threshold = config["search_args"].get("initial_threshold", 0.5)
        accepted_samples = [
            res for res in results
            if res[choice_of_metric] >= acceptance_threshold
        ]
        acceptance_rate = len(accepted_samples)/len(results) if results else 0
        datapoint["acceptance_rate"] = acceptance_rate
        if accepted_samples:
            sorted_list = sorted(accepted_samples, key=lambda x: f(x), reverse=True)
        else:
            # Fallback to best of rejected samples
            sorted_list = sorted(results, key=lambda x: f(x), reverse=True)
    else:
        sorted_list = sorted(results, key=lambda x: f(x), reverse=True)

    topk_scores = sorted_list[:topk]

    # Print debug information.
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | Score={ts[choice_of_metric]}")

    best_img_path = os.path.join(
        root_dir, f"{prompt_filename}_i@{search_round}_s@{topk_scores[0]['seed']}.{extension_to_use}"
    )
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]["seed"],
        "best_noise": topk_scores[0]["noise"],
        "best_score": topk_scores[0][choice_of_metric],
        "top10_noise": [topk_scores[i]["noise"] for i in range(min(10, len(topk_scores)))],
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
    }

    # Check if the neighbors have any improvements (zero-order only).
    search_method = search_args.get("search_method", "random") if search_args else "random"
    if search_args and search_method == "zero-order":
        first_score = f(results[0])
        neighbors_with_better_score = any(f(item) > first_score for item in results[1:])
        datapoint["neighbors_improvement"] = neighbors_with_better_score

    # Serialize.
    if search_method == "zero-order":
        if datapoint["neighbors_improvement"]:
            datapoint_new = datapoint.copy()
            datapoint_new.pop("top10_noise", None)
            serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint_new, **export_args)
        else:
            print("Skipping serialization as there was no improvement in this round.")
    elif search_method in ["random", "evolutionary", "rejection"]:
        datapoint_new = datapoint.copy()
        datapoint_new.pop("top10_noise", None)
        serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint_new, **export_args)

    return datapoint


@torch.no_grad()
def main():
    # === Load configuration and CLI arguments ===
    args = parse_cli_args()
    with open(args.pipeline_config_path, "r") as f:
        config = json.load(f)
    config.update(vars(args))

    search_args = config["search_args"]
    search_rounds = search_args["search_rounds"]
    search_method = search_args.get("search_method", "random")
    num_prompts = config["num_prompts"]

    # === Create output directory ===
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = config.pop("pretrained_model_name_or_path")
    verifier_name = config["verifier_args"]["name"]
    choice_of_metric = config["verifier_args"]["choice_of_metric"]
    output_dir = os.path.join(
        "output",
        MODEL_NAME_MAP[pipeline_name],
        verifier_name,
        choice_of_metric,
        current_datetime,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # === Load prompts ===
    if args.prompt is None:
        with open("prompts_drawbench.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if num_prompts != "all":
            prompts = prompts[:num_prompts]
    else:
        prompts = [args.prompt]
    print(f"Using {len(prompts)} prompt(s).")

    # === Set up the image-generation pipeline ===
    torch_dtype = TORCH_DTYPE_MAP[config.pop("torch_dtype")]
    fp_kwargs = {"pretrained_model_name_or_path": pipeline_name, "torch_dtype": torch_dtype}
    if "Wan" in pipeline_name:
        # As per recommendations from https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan.
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(pipeline_name, subfolder="vae", torch_dtype=torch.float32)
        fp_kwargs.update({"vae": vae})
    pipe = DiffusionPipeline.from_pretrained(**fp_kwargs, local_files_only=True) #IF APPLICABLE!
    if not config.get("use_low_gpu_vram", False):
        pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    
    # ----- NFE Counter using Forward Hook -----

    def counting_hook(module, input, output):
        # Only count UNet forward passes during actual image generation
        if not hasattr(counting_hook, "sampling_phase"):
            counting_hook.counter += 1
        
    counting_hook.counter = 0

    # Register the hook on the UNet inside the pipeline.
    if hasattr(pipe, 'unet'):
        hook_handle = pipe.unet.register_forward_hook(counting_hook)
    elif hasattr(pipe, 'transformer'):
        hook_handle = pipe.transformer.register_forward_hook(counting_hook)

    # === Load verifier model ===
    verifier_args = config["verifier_args"]
    verifier_cls = SUPPORTED_VERIFIERS.get(verifier_args["name"])
    if verifier_cls is None:
        raise ValueError("Verifier class evaluated to be `None`. Make sure the dependencies are installed properly.")

    verifier = verifier_cls(**verifier_args)
    
    print(config)
    def get_search_algorithm(search_method, config):
        if search_method == "random":
            return RandomSearch(config)
        elif search_method == "zero-order":
            return ZeroOrderSearch(config)
        elif search_method == "evolutionary":
            return EvolutionarySearch(config)
        elif search_method == "rejection":
            return RejectionSamplingSearch(config)
        else:
            raise ValueError(f"Unsupported search method: {search_method}")

    # In your main function:
    search_method = config["search_args"].get("search_method", "random")
    search_algo = get_search_algorithm(search_method, config)
    
    all_nfe_data = {}

    # === Main loop: For each prompt and each search round ===
    pipeline_call_args = config["pipeline_call_args"].copy()
    for prompt in tqdm(prompts, desc="Processing prompts"):
        previous_data = None
        counting_hook.counter = 0
        all_nfe_data[prompt] = {}
        for search_round in range(1, config["search_args"]["search_rounds"] + 1):
            print(f"\n=== Prompt: {prompt} | Round: {search_round} ===")


            if search_method in ["random", "evolutionary"]:
                search_algo.config["num_samples"] = 2 ** search_round
            elif search_method == "rejection":
                search_algo.config["num_samples"] = config["search_args"].get("num_samples", 10)
            else:
                search_algo.config["num_samples"] = 1

            search_algo.config["torch_dtype"] = torch_dtype
            search_algo.config["pipeline_name"] = pipeline_name
            print(search_algo.config)
            noises = search_algo.generate_noise_candidates(previous_data)
            datapoint = sample(
                noises=noises,
                prompt=prompt,
                search_round=search_round,
                pipe=pipe,
                verifier=verifier,
                topk=TOPK,
                root_dir=output_dir,
                config=config,
            )
            # Update previous_data if needed (for zero-order or evolutionary strategies)
            print(f"Total function evaluations (NFEs) so far: {counting_hook.counter}")
            
            all_nfe_data[prompt][search_round] = {}
            all_nfe_data[prompt][search_round]["NFE"] = str(counting_hook.counter)
            all_nfe_data[prompt][search_round]["clip_score"] = str(compute_single_clipscore(datapoint["best_img_path"], prompt))
            previous_data = datapoint
            print(all_nfe_data)
            with open(os.path.join(output_dir, "all_nfe_data.json"), "w") as f:
                json.dump(all_nfe_data, f, indent=4)

    

if __name__ == "__main__":
    main()
