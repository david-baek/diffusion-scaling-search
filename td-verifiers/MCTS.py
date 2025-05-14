# mcts.py

import math
import random
import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional, NamedTuple
from PIL import Image
from torchvision.transforms.functional import to_pil_image # Added for PIL conversion

# Assuming diffusers and transformers are installed
try:
    from diffusers import FluxPipeline
    from diffusers.schedulers.scheduling_utils import SchedulerOutput
except ImportError:
    print("Warning: diffusers library not found. Flux specific classes will not work.")
    FluxPipeline = None # Placeholder
    SchedulerOutput = None # Placeholder

# --- Configuration & Constants ---
EXPLORATION_CONSTANT = 1.414 # c value in UCB1 formula (sqrt(2) is common)
MIN_TIMESTEP_INDEX = 0 # Use integer timestep indices (0 to num_train_timesteps-1)

# --- Action Definition ---
# Represents a choice at a given node
class Action(NamedTuple):
    # delta_t_index: int   # How many scheduler steps to jump back
    cfg_scale: float     # Guidance scale for this step
    noise_sigma: float   # Optional noise level to add *before* the step

    # Make hashable for use as dictionary keys
    def __hash__(self):
        # Removed delta_t_index as it's now implicit (always step back by 1)
        return hash((self.cfg_scale, self.noise_sigma))

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        # Removed delta_t_index comparison
        return (self.cfg_scale == other.cfg_scale and
                self.noise_sigma == other.noise_sigma)

# --- Abstract Interfaces for External Components ---

class RewardModel(ABC):
    """Abstract base class for the reward model."""
    @abstractmethod
    def score(self, image: Image.Image, prompt_text: str, t_index: int) -> float:
        """
        Scores a given decoded image based on the prompt and timestep index.
        Args:
            image: The decoded PIL Image.
            prompt_text: The conditional prompt text.
            t_index: The timestep index associated with the state before decoding.
        Returns:
            A scalar reward value. Higher is better.
        """
        pass

# --- Concrete Implementations for Flux ---

class FluxFlowIntegrator:
    """Concrete implementation for Flux-1.dev backward flow step."""
    def __init__(self, pipeline: FluxPipeline, prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor, device: torch.device):
        if pipeline is None:
             raise ValueError("Diffusers library not found or FluxPipeline not available.")
        self.pipeline = pipeline
        self.scheduler = pipeline.scheduler
        self.unet = pipeline.transformer # FLUX uses a Transformer, not a UNet
        self.device = device
        self.dtype = pipeline.dtype # Use pipeline's dtype

        # Ensure embeddings are on the correct device and expanded for batch dim
        self.prompt_embeds = prompt_embeds.to(device, dtype=self.dtype)
        self.negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.dtype)

        # Assume batch size of 1 for MCTS node expansion for now
        if self.prompt_embeds.dim() == 2: # If shape is [seq_len, embed_dim]
            self.prompt_embeds = self.prompt_embeds.unsqueeze(0) # Add batch dim -> [1, seq_len, embed_dim]
        if self.negative_prompt_embeds.dim() == 2:
             self.negative_prompt_embeds = self.negative_prompt_embeds.unsqueeze(0)

        print(f"FluxFlowIntegrator initialized. Embeddings shape: {self.prompt_embeds.shape}")


    def step(self, z_t: torch.Tensor, t_index: int, action: Action) -> torch.Tensor:
        """
        Performs one step backward using the scheduler and model.
        Args:
            z_t: The current latent state tensor (on self.device).
            t_index: The current timestep index (integer).
            action: The Action object containing cfg_scale for this step.
        Returns:
            The resulting latent state tensor z_prev at the previous timestep index.
        """
        if z_t.device != self.device:
             z_t = z_t.to(self.device)

        # Get the actual timestep value from the index for the scheduler/model
        timestep = self.scheduler.timesteps[t_index].to(self.device)

        # Classifier-Free Guidance: Run model twice (cond and uncond)
        # Prepare latent input for the model (some schedulers require scaling)
        # Assuming FLUX transformer takes latent and timestep directly? Need to confirm API.
        # Let's assume direct input for now, adjust if needed based on FluxPipeline source.
        latent_model_input = torch.cat([z_t] * 2) # Duplicate latent for CFG
        timestep_model_input = torch.cat([timestep.unsqueeze(0)] * 2) # Duplicate timestep

        # Concatenate prompt embeddings
        text_embeddings = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)

        # Predict noise (or direct state prediction, depending on scheduler type)
        # The FLUX transformer API might differ slightly, adapt as needed.
        # This assumes transformer takes 'hidden_states', 'timestep', 'encoder_hidden_states'
        # Ensure arguments match pipeline.transformer.forward() signature
        noise_pred_uncond, noise_pred_text = self.unet(
            hidden_states=latent_model_input,
            timestep=timestep_model_input,
            encoder_hidden_states=text_embeddings,
            return_dict=False
        )[0].chunk(2) # Chunk the result back into uncond and cond predictions


        # Perform guidance
        noise_pred = noise_pred_uncond + action.cfg_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous state using the scheduler
        # The scheduler step function signature varies. Adapt as needed.
        # Common signature: step(model_output, timestep, sample, ...)
        scheduler_output = self.scheduler.step(noise_pred, timestep, z_t, return_dict=True)
        z_t_prev = scheduler_output.prev_sample

        return z_t_prev.to(self.dtype) # Ensure consistent dtype

    def apply_noise(self, z_t: torch.Tensor, noise_sigma: float, rng_seed: Optional[int] = None) -> torch.Tensor:
        """Adds Gaussian noise to the latent state."""
        if noise_sigma <= 0:
            return z_t
        
        generator = None
        if rng_seed is not None:
             generator = torch.Generator(device=self.device).manual_seed(rng_seed)
             
        noise = torch.randn_like(z_t, device=self.device, generator=generator) * noise_sigma
        noisy_z_t = z_t + noise
        return noisy_z_t.to(z_t.dtype)


class FluxDecoder:
    """Concrete implementation for decoding Flux latents using the VAE."""
    def __init__(self, pipeline: FluxPipeline, device: torch.device):
        if pipeline is None:
             raise ValueError("Diffusers library not found or FluxPipeline not available.")
        self.pipeline = pipeline
        self.vae = pipeline.vae
        self.image_processor = pipeline.image_processor
        self.device = device
        self.dtype = pipeline.vae.dtype # Use VAE's dtype for decoding

        # Store VAE scaling factors
        self.scaling_factor = self.vae.config.scaling_factor
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        print(f"FluxDecoder initialized. VAE Scale: {self.scaling_factor}, Shift: {self.shift_factor}")


    @torch.no_grad()
    def decode(self, z_t: torch.Tensor) -> Image.Image:
        """Decodes a latent state tensor into a PIL Image."""
        if z_t.device != self.device:
             z_t = z_t.to(self.device)
             
        # Inverse scale and shift before decoding
        latents_for_decode = 1.0 / self.scaling_factor * (z_t - self.shift_factor)
        latents_for_decode = latents_for_decode.to(self.dtype) # Ensure correct dtype for VAE

        # Decode using VAE
        # Ensure VAE is on the correct device (handled by pipeline's offloading usually)
        decoded_output = self.vae.decode(latents_for_decode, return_dict=False)[0]

        # Post-process to get PIL Image
        image = self.image_processor.postprocess(decoded_output, output_type="pil")[0]
        return image


# --- Core MCTS Data Structures ---

class Node:
    """Represents a node in the Monte Carlo Tree."""
    def __init__(self, t_index: int, z_t: torch.Tensor, parent: Optional['Node'] = None,
                 action_taken: Optional[Action] = None, available_actions: List[Action] = [],
                 noise_seed: Optional[int] = None):
        self.t_index = t_index          # Current timestep index (integer)
        self.z_t = z_t                  # Latent state (torch.Tensor on device)
        self.parent = parent
        self.action_taken = action_taken # Action that led to this node from parent
        self.noise_seed = noise_seed     # Optional seed used if noise was applied

        self.children: Dict[Action, Node] = {} # Action -> Child Node map

        self.visit_count: int = 0     # N
        self.total_value: float = 0.0 # Q (sum of rewards from evaluations)

        # Actions to explore from this node
        print(f"Creating node at t_index={t_index} with {len(available_actions) if available_actions is not None else 0} available actions")
        if available_actions is None:
            print("WARNING: available_actions is None, using empty list")
            available_actions = []
        self.untried_actions = available_actions[:] # Make a copy
        random.shuffle(self.untried_actions) # Shuffle for random exploration order
        print(f"Node created with {len(self.untried_actions)} untried actions")

    @property
    def value_estimate(self) -> float:
        """Returns the current average value (Q/N) of the node."""
        if self.visit_count == 0:
            return 0.0 # Avoid division by zero
        return self.total_value / self.visit_count

    def is_terminal(self) -> bool:
        """Checks if the node represents a terminal state (t_index=0)."""
        return self.t_index <= MIN_TIMESTEP_INDEX

    def is_fully_expanded(self) -> bool:
        """Checks if all possible actions from this node have been tried."""
        return len(self.untried_actions) == 0

    def select_child(self, exploration_constant: float) -> 'Node':
        """Selects the best child node using the UCB1 formula."""
        best_score = -float('inf')
        best_child = None

        log_parent_visits = math.log(self.visit_count) if self.visit_count > 0 else 0

        for action, child in self.children.items():
            if child.visit_count == 0:
                score = float('inf') # Prioritize unvisited children
            else:
                exploit_term = child.value_estimate
                explore_term = exploration_constant * math.sqrt(log_parent_visits / child.visit_count)
                score = exploit_term + explore_term

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            # Fallback needed if selection fails (e.g., only child has 0 visits but log_parent is 0)
            if self.children:
                print(f"Warning: select_child found no best child for Node(t={self.t_index}, N={self.visit_count}). Returning first child.")
                return next(iter(self.children.values()))
            else:
                raise RuntimeError(f"select_child called on node {self.t_index} with no children")

        return best_child

    def expand(self, action: Action, child_node: 'Node') -> None:
        """Adds a new child node corresponding to an action."""
        if action in self.children:
            # This might happen if hash collision occurs or logic error
            print(f"Warning: Action {action} already expanded for node at t_index={self.t_index}. Overwriting.")
        self.children[action] = child_node

    def update(self, reward: float) -> None:
        """Updates the node's visit count and total value."""
        self.visit_count += 1
        self.total_value += reward
        
    def __repr__(self):
        latent_info = f"z_t shape={self.z_t.shape}"
        return (f"Node(t_idx={self.t_index}, N={self.visit_count}, Q={self.total_value:.3f}, "
                f"Value={self.value_estimate:.3f}, Children={len(self.children)}, "
                f"Untried={len(self.untried_actions)}, {latent_info})")


class MCTS:
    """Monte Carlo Tree Search implementation for optimizing diffusion trajectories."""

    def __init__(self,
                 initial_t_index: int,
                 initial_z_t: torch.Tensor,
                 pipeline: FluxPipeline, # Pass the whole pipeline
                 reward_model: RewardModel,
                 action_space: List[Action],
                 prompt: str, # Need prompt for reward model and embeddings
                 negative_prompt: str = "",
                 exploration_constant: float = EXPLORATION_CONSTANT,
                 initial_noise_seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initializes the MCTS.
        Args:
            initial_t_index: Starting timestep index.
            initial_z_t: Starting latent state tensor.
            pipeline: The loaded FluxPipeline object.
            reward_model: Concrete implementation of RewardModel.
            action_space: List of possible Action objects to take at any step.
            prompt: The conditioning prompt text.
            negative_prompt: The negative conditioning prompt text.
            exploration_constant: The 'c' value for the UCB1 selection formula.
            initial_noise_seed: Optional root seed for noise generation.
            device: The torch device ('cuda' or 'cpu'). Autodetect if None.
        """
        if not action_space:
             raise ValueError("Action space cannot be empty.")
        if pipeline is None:
             raise ValueError("FluxPipeline object is required.")
             
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MCTS using device: {self.device}")
        
        self.pipeline = pipeline.to(self.device) # Ensure pipeline is on the correct device
        self.reward_model = reward_model # Assume reward model handles its own device placement
        self.action_space = action_space
        self.exploration_constant = exploration_constant
        self.initial_noise_seed = initial_noise_seed
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        
        # Pre-compute prompt embeddings (move this outside if running MCTS multiple times with same prompt)
        print("Encoding prompts...")
        # Need to know max sequence length expected by Flux text encoders
        # Assuming 512 like in train.py example
        # Need prompt_embeds and potentially pooled embeddings depending on FLUX model specifics
        # Let's assume pipeline.encode_prompt handles it correctly for now
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
             prompt=self.prompt,
             negative_prompt=self.negative_prompt,
             device=self.device,
             num_images_per_prompt=1, # MCTS expands one path at a time
             do_classifier_free_guidance=True, # We need both cond and uncond
             max_sequence_length=512 # Adjust if needed
        )
        # Store the embeddings needed by the FlowIntegrator (likely non-pooled)
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        print(f"Prompt embeddings computed. Shape: {self.prompt_embeds.shape}")


        # Instantiate concrete components
        self.flow_integrator = FluxFlowIntegrator(
             pipeline=self.pipeline,
             prompt_embeds=self.prompt_embeds,
             negative_prompt_embeds=self.negative_prompt_embeds,
             device=self.device
        )
        self.decoder = FluxDecoder(pipeline=self.pipeline, device=self.device)

        # Ensure initial latent is on the correct device
        initial_z_t = initial_z_t.to(self.device, dtype=self.pipeline.dtype)

        self.root = Node(t_index=initial_t_index, z_t=initial_z_t,
                         available_actions=self.action_space,
                         noise_seed=self.initial_noise_seed)
        print(f"MCTS initialized with root node at t_index={self.root.t_index}")
        print(f"Initial latent shape: {initial_z_t.shape}, dtype: {initial_z_t.dtype}, device: {initial_z_t.device}")

        # --- NFE Tracking ---
        self.nfe_count = 0 
        # --- End NFE Tracking ---

        print(f"MCTS initialized. NFE Count: {self.nfe_count}")


    def _select(self, node: Node) -> Node:
        """Phase 1: Selects a node to expand using UCB1."""
        current_node = node
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node
            else:
                # Ensure children exist before selecting
                if not current_node.children:
                     print(f"Warning: Node {current_node.t_index} is fully expanded but has no children. Stopping selection.")
                     return current_node # Return the node itself, indicates a potential dead end
                current_node = current_node.select_child(self.exploration_constant)
        return current_node # Reached a terminal node

    def _expand_and_evaluate(self, node: Node) -> Tuple[Node, float]:
        """
        Phase 2: Expands an un-tried action, performs flow step, decodes, scores,
                 and creates the new child node.
        Returns:
             The newly created child node and its immediate reward.
        """
        if node.is_terminal():
            raise ValueError("Cannot expand a terminal node.")
        if node.is_fully_expanded():
            raise ValueError("Node is already fully expanded.")

        print(f"Expanding node at t_index={node.t_index} with {len(node.untried_actions)} untried actions")
        if not node.untried_actions:
            print("ERROR: No untried actions available for expansion")
            raise ValueError("No untried actions available for expansion")

        action = node.untried_actions.pop()
        print(f"Selected action for expansion: {action}")

        # --- Simulate the action ---
        z_t_current = node.z_t.clone().to(self.device) # Ensure it's on device
        child_noise_seed = None
        if action.noise_sigma > 0:
            if node.noise_seed is not None:
                child_noise_seed = hash((node.noise_seed, action)) & ((1 << 32) -1)
            
            z_t_current = self.flow_integrator.apply_noise(
                z_t_current, action.noise_sigma, rng_seed=child_noise_seed
            )

        # Perform the diffusion flow step (implicitly steps back by 1)
        t_prev_index = node.t_index - 1
        try:
            # Ensure z_t_current is correct dtype before passing to step
            z_t_prev = self.flow_integrator.step(z_t_current.to(self.pipeline.dtype), node.t_index, action)
            
            # --- NFE Tracking ---
            self.nfe_count += 1 # Increment AFTER successful step
            # --- End NFE Tracking ---
            
        except Exception as e:
            print(f"ERROR during flow integrator step from {node.t_index} to {t_prev_index}: {e}")
            # Create a dummy node to mark this path as failed
            dummy_node = Node(t_index=t_prev_index, z_t=torch.zeros_like(node.z_t), parent=node, action_taken=action, available_actions=self.action_space)
            node.expand(action, dummy_node)
            return dummy_node, -float('inf') # Signal failure

        # --- Decode and Evaluate ---
        reward = -float('inf') # Default to worst reward
        try:
            decoded_image = self.decoder.decode(z_t_prev)
            reward = self.reward_model.score(decoded_image, self.prompt, t_prev_index)
        except Exception as e:
            print(f"ERROR during decode/reward scoring at t_prev_idx={t_prev_index}: {e}")

        # --- Create new child node ---
        child_node = Node(
            t_index=t_prev_index,
            z_t=z_t_prev.detach(), # Store the resulting latent
            parent=node,
            action_taken=action,
            available_actions=self.action_space if t_prev_index > MIN_TIMESTEP_INDEX else [],
            noise_seed=child_noise_seed
        )
        node.expand(action, child_node)

        return child_node, reward

    def _backpropagate(self, node: Node, reward: float) -> None:
        """Phase 3: Propagates the reward back up the tree."""
        current_node = node
        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent

    def search(self, 
               num_iterations: Optional[int] = None, 
               time_limit_secs: Optional[float] = None,
               nfe_limit: Optional[int] = None) -> None:
        """Runs the MCTS algorithm for a specified number of iterations, time limit, or NFE limit."""
        if num_iterations is None and time_limit_secs is None and nfe_limit is None:
            raise ValueError("Must provide at least one limit: num_iterations, time_limit_secs, or nfe_limit.")

        start_time = time.time()
        iterations_done = 0

        print(f"Starting MCTS Search. Limits: Iter={num_iterations}, Time={time_limit_secs}s, NFE={nfe_limit}")

        while True:
            # Check termination conditions (Order matters if multiple limits are hit simultaneously)
            if time_limit_secs is not None and (time.time() - start_time) >= time_limit_secs:
                 print(f"\nReached time limit ({time_limit_secs:.2f}s)."); break
            if nfe_limit is not None and self.nfe_count >= nfe_limit:
                 print(f"\nReached NFE limit ({self.nfe_count}/{nfe_limit})."); break
            if num_iterations is not None and iterations_done >= num_iterations:
                 print(f"\nReached iteration limit ({iterations_done}/{num_iterations})."); break

            # 1. Selection
            selected_node = self._select(self.root)

            # 2. Expansion & Evaluation
            reward = 0.0
            new_node = selected_node # Default if terminal or expansion fails

            if selected_node.is_terminal():
                 # Selected a terminal node. Backpropagate its stored value estimate.
                 reward = selected_node.value_estimate
            elif selected_node.is_fully_expanded() and not selected_node.children:
                 # Selected a non-terminal node that's fully expanded but has no children
                 # This implies previous expansions failed or it's a dead end.
                 print(f"Warning: Selected fully expanded node {selected_node.t_index} with no children. Backpropagating current value.")
                 reward = selected_node.value_estimate
            elif not selected_node.is_fully_expanded():
                 # Expand the selected node
                 try:
                     new_node, reward = self._expand_and_evaluate(selected_node)
                     if reward == -float('inf'): pass # Failure handled in _expand_and_evaluate print("Backpropagating failure reward...")
                 except (ValueError, IndexError) as e:
                     print(f"Error during expansion phase from node {selected_node.t_index}: {e}")
                     reward = -float('inf') # Penalize heavily
                     new_node = selected_node # Backpropagate from the node where error occurred
            else:
                 # This case should ideally not be reached if selection logic is correct
                 # (i.e., selects an expandable node or a terminal node)
                 print(f"Warning: Unexpected state in search loop. Selected node: {selected_node}")
                 reward = selected_node.value_estimate # Fallback to current estimate


            # 3. Backpropagation
            self._backpropagate(new_node, reward)

            iterations_done += 1
            if iterations_done % 100 == 0: # Print progress periodically
                 elapsed = time.time() - start_time
                 # Update progress print to include NFE
                 print(f"Iter: {iterations_done}, NFE: {self.nfe_count}, Elapsed: {elapsed:.2f}s, Root Value: {self.root.value_estimate:.4f} ({self.root.visit_count} visits)", end='\r')

        elapsed_total = time.time() - start_time
        print(f"\nMCTS finished. Iterations: {iterations_done}, NFEs: {self.nfe_count}, Time: {elapsed_total:.2f}s")
        print(f"Root Node Stats: {self.root}")


    def get_best_trajectory(self, criteria='visits') -> Tuple[List[Node], List[Action]]:
        """Extracts the best trajectory based on visit count or value."""
        trajectory_nodes = [self.root]
        trajectory_actions = []
        current_node = self.root

        while not current_node.is_terminal() and current_node.children:
            best_child = None
            best_action = None # Initialize best_action here

            if criteria == 'visits':
                max_visits = -1
                for action, child in current_node.children.items():
                    if child.visit_count > max_visits:
                        max_visits = child.visit_count
                        best_child = child
                        best_action = action # Store the action leading to the best child
            elif criteria == 'value':
                max_value = -float('inf')
                # Prioritize visited children for value comparison
                visited_children = {a: c for a, c in current_node.children.items() if c.visit_count > 0}
                if visited_children:
                    for action, child in visited_children.items():
                        if child.value_estimate > max_value:
                            max_value = child.value_estimate
                            best_child = child
                            best_action = action # Store action
                else:
                    # If no children visited, maybe pick the first one? Or based on initial reward?
                    # Let's break if no visited children provide a basis for value comparison
                    print(f"Warning: Criteria 'value' selected for node {current_node.t_index}, but no children visited. Stopping trajectory.")
                    break
            else:
                raise ValueError("Criteria must be 'visits' or 'value'")

            if best_child is None:
                print(f"Warning: Could not determine best child for node {current_node} with criteria '{criteria}'. Stopping trajectory here.")
                break

            trajectory_nodes.append(best_child)
            if best_action is not None: # Ensure action was found
                trajectory_actions.append(best_action)
            else: # Should not happen if best_child was found
                print(f"Error: Best child found but corresponding action missing at node {current_node.t_index}")
                break
            current_node = best_child

        return trajectory_nodes, trajectory_actions


# --- Example Usage ---

if __name__ == "__main__":
    print("--- MCTS Example with Flux Components ---")

    if FluxPipeline is None:
        print("Cannot run example: diffusers library not found or FluxPipeline unavailable.")
        exit()

    # --- Dummy Reward Model ---
    class DummyRewardModel(RewardModel):
        def score(self, image: Image.Image, prompt_text: str, t_index: int) -> float:
            # Simulate reward: higher score for brighter images, penalize later timesteps
            try:
                avg_brightness = np.mean(np.array(image.convert("L"))) / 255.0
                # Reward decay based on timestep index (assuming 1000 steps total)
                time_penalty = (t_index / 1000.0) * 0.5
                reward = avg_brightness - time_penalty
                # print(f"  Dummy Reward: t_idx={t_index}, brightness={avg_brightness:.3f} -> reward={reward:.3f}")
            return reward
            except Exception as e:
                print(f"Error in dummy reward scoring: {e}")
                return -1.0 # Penalize errors

    # --- Setup ---
    # IMPORTANT: This requires a downloaded Flux model
    flux_model_name = "black-forest-labs/FLUX.1-dev" # Or path to local model
    prompt = "A watercolor painting of a cozy cabin in a snowy forest"
    negative_prompt = "low quality, blurry, noisy, text, signature"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    # Load Pipeline (can be slow)
    print(f"Loading pipeline {flux_model_name}...")
    try:
        # Load initially to CPU to manage memory if needed, then move to device
        pipe = FluxPipeline.from_pretrained(flux_model_name, torch_dtype=pipeline_dtype)
        # pipe.enable_model_cpu_offload() # Optional: if GPU memory is tight
        pipe = pipe.to(device)
        print("Pipeline loaded.")
    except Exception as e:
        print(f"ERROR loading pipeline: {e}")
        print("Ensure the model is downloaded and dependencies are installed.")
        exit()

    # Set timesteps (important for mapping t_index)
    num_inference_steps = 30 # Example number of steps for generation
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    initial_t_index = len(pipe.scheduler.timesteps) - 1 # Start from the last index

    # Define Action Space
    action_space = [
        Action(cfg_scale=1.5, noise_sigma=0.0),
        Action(cfg_scale=3.5, noise_sigma=0.0), # Reference CFG from train.py
        Action(cfg_scale=5.0, noise_sigma=0.0),
        Action(cfg_scale=3.5, noise_sigma=0.01), # Add slight noise
        # Action(cfg_scale=3.5, noise_sigma=0.05), # Add more noise
    ]

    # Prepare initial latent state (random noise)
    # Need to know the expected latent shape for FLUX
    # Infer from VAE? Often (Batch, Channels, Height//VAE_Factor, Width//VAE_Factor)
    # Example: If 512x512 image and VAE factor 8 -> (1, 4, 64, 64)? Let's assume 4 channels.
    # FLUX uses a different latent structure. Consult pipeline/model docs.
    # Let's *assume* we can get the shape from the text encoder output dim or unet input.
    # For FLUX.1-dev, the transformer operates on patchified embeddings, not a direct Conv VAE latent.
    # Getting the initial z_t is tricky without running the forward diffusion process.
    # Let's use a placeholder shape based on common diffusion models for now.
    # You MUST replace this with the correct shape/method for FLUX initial latents.
    latent_height = 64 # Placeholder
    latent_width = 64 # Placeholder
    latent_channels = pipe.transformer.config.in_channels # Try to infer channels
    initial_shape = (1, latent_channels, latent_height, latent_width)
    initial_z = torch.randn(initial_shape, device=device, dtype=pipeline_dtype)
    # Scaling initial noise might be needed depending on scheduler (e.g., * pipe.scheduler.init_noise_sigma)
    if hasattr(pipe.scheduler, 'init_noise_sigma'):
         initial_z = initial_z * pipe.scheduler.init_noise_sigma


    print(f"Initial latent shape (guessed): {initial_z.shape}")


    # Instantiate Dummy Reward Model
    reward_model = DummyRewardModel()

    # --- Run MCTS ---
    print("\nInstantiating MCTS...")
    try:
        mcts_instance = MCTS(
            initial_t_index=initial_t_index,
            initial_z_t=initial_z,
            pipeline=pipe,
            reward_model=reward_model,
            action_space=action_space,
            prompt=prompt,
            negative_prompt=negative_prompt,
            exploration_constant=1.0, # Lower exploration for demo
            device=device
        )
    except Exception as e:
        print(f"ERROR during MCTS instantiation: {e}")
        import traceback
        traceback.print_exc()
        exit()


    print("\nStarting MCTS Search...")
    # Example: Run for max 1000 NFEs OR 500 iterations, whichever comes first
    mcts_instance.search(num_iterations=500, nfe_limit=1000) 
    
    # Example: Run for max 60 seconds OR 2000 NFEs
    # mcts_instance.search(time_limit_secs=60.0, nfe_limit=2000)

    print("\n--- Results ---")
    print(f"Root Node Final: {mcts_instance.root}")
    # for act, child in mcts_instance.root.children.items():
    #      print(f"  Child Action {act}: {child}")

    # Get best path based on visits
    try:
        best_nodes_visits, best_actions_visits = mcts_instance.get_best_trajectory(criteria='visits')
        print("\nBest Trajectory (Most Visits):")
        for i, node in enumerate(best_nodes_visits):
            print(f"  Step {i}: Node(t_idx={node.t_index}, V={node.value_estimate:.3f}, N={node.visit_count})")
        if i < len(best_actions_visits):
            print(f"    Action Taken: {best_actions_visits[i]}")
        
        # Optionally decode the final image from the best trajectory
        if best_nodes_visits:
            final_node = best_nodes_visits[-1]
            if final_node.t_index <= MIN_TIMESTEP_INDEX :
                print("\nDecoding final image from best trajectory (visits)...")
                final_image = mcts_instance.decoder.decode(final_node.z_t)
                final_image.save("mcts_best_visits_final.png")
                print("Saved final image as mcts_best_visits_final.png")

    except Exception as e:
        print(f"Error getting/decoding best trajectory (visits): {e}")

    # Get best path based on value
    try:
        best_nodes_value, best_actions_value = mcts_instance.get_best_trajectory(criteria='value')
        print("\nBest Trajectory (Highest Value):")
        for i, node in enumerate(best_nodes_value):
            print(f"  Step {i}: Node(t_idx={node.t_index}, V={node.value_estimate:.3f}, N={node.visit_count})")
        if i < len(best_actions_value):
            print(f"    Action Taken: {best_actions_value[i]}")

        if best_nodes_value:
            final_node_val = best_nodes_value[-1]
            if final_node_val.t_index <= MIN_TIMESTEP_INDEX:
                print("\nDecoding final image from best trajectory (value)...")
                final_image_val = mcts_instance.decoder.decode(final_node_val.z_t)
                final_image_val.save("mcts_best_value_final.png")
                print("Saved final image as mcts_best_value_final.png")
                  
    except Exception as e:
        print(f"Error getting/decoding best trajectory (value): {e}")
