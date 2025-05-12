from abc import ABC, abstractmethod
from utils import get_noises, get_latent_prep_fn, generate_neighbors

import numpy as np
import random

import torch

MAX_SEED = np.iinfo(np.int32).max

class SearchAlgorithm(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate_noise_candidates(self, previous_data=None):
        """
        Should return a dictionary of noise candidates.
        `previous_data` can be used to seed the next round (if applicable).
        """
        pass

class RandomSearch(SearchAlgorithm):
    def generate_noise_candidates(self, previous_data=None):
        # Sample new noise candidates randomly.
        return get_noises(
            max_seed=MAX_SEED,
            num_samples=self.config["num_samples"],
            dtype=self.config["torch_dtype"],
            fn=get_latent_prep_fn(self.config["pipeline_name"]),
            **self.config["pipeline_call_args"],
        )

class ZeroOrderSearch(SearchAlgorithm):
    def generate_noise_candidates(self, previous_data=None):
        # Assume `previous_data` includes the best noise from the last round.
        if previous_data is None:
            base_noise = get_noises(
                max_seed=MAX_SEED,
                num_samples=1,
                dtype=self.config["torch_dtype"],
                fn=get_latent_prep_fn(self.config["pipeline_name"]),
                **self.config["pipeline_call_args"],
            )
            base_noise = next(iter(base_noise.values()))
        else:
            base_noise = previous_data["best_noise"]
        neighbors = generate_neighbors(
            base_noise, 
            threshold=self.config["search_args"]["threshold"], 
            num_neighbors=self.config["search_args"]["num_neighbors"]
        ).squeeze(0)
        # Concatenate the base noise with its neighbors.
        neighbors_and_noise = torch.cat([base_noise, neighbors], dim=0)
        new_noises = {}
        for i, noise_tensor in enumerate(neighbors_and_noise):
            new_noises[i] = noise_tensor.unsqueeze(0)
        return new_noises

class EvolutionarySearch(SearchAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.population_size = config.get("evo_population_size", 10)
        self.mutation_rate = config.get("evo_mutation_rate", 0.03)
        self.num_generations = config.get("evo_num_generations", 5)
        self.current_generation = 0
        self.current_population = None
        self.torch_dtype = config.get("torch_dtype", None)
        self.latent_prep_fn = None

    def generate_noise_candidates(self, previous_data=None):
        """
        Generates noise candidates via an evolutionary search.
        
        - If no previous_data exists, then an initial population is created using get_noises.
        - If previous_data is available, the algorithm fetches the top10 noise candidates from 
          previous_data, applies selection and crossover between randomly chosen parents, 
          then perturbs the new candidates using Gaussian mutation.
        """
        # Set latent preprocessing function if not already set.
        if self.latent_prep_fn is None:
            self.latent_prep_fn = get_latent_prep_fn(self.config["pipeline_name"])
        # Set torch dtype if not already set.
        if self.torch_dtype is None:
            self.torch_dtype = self.config.get("torch_dtype", torch.float16)
        
        # If no previous data or current_population exists, generate an initial population.
        if self.current_population is None or previous_data is None:
            self.current_population = get_noises(
                max_seed=MAX_SEED,
                num_samples=self.population_size,
                dtype=self.torch_dtype,
                fn=self.latent_prep_fn,
                **self.config["pipeline_call_args"],
            )
            return self.current_population

        # Retrieve top10 noise candidates from the previous iteration.
        top10_noise = previous_data["top10_noise"]

        new_population = self.current_population.copy()
        for i in range(self.population_size):
            # Selection: Randomly pick two parent candidates from the top10 list.
            parent1 = random.choice(top10_noise)
            parent2 = random.choice(top10_noise)
            # Crossover: Create a child by taking a weighted sum of the two parents.
            alpha = random.uniform(0, 1)
            child = alpha * parent1 + (1 - alpha) * parent2
            # Mutation: Add Gaussian noise to the child to explore new regions.
            child = child + self.mutation_rate * torch.randn_like(child)
            # Ensure the child has a batch dimension if it's a 2D tensor.
            if child.dim() == 2:
                child = child.unsqueeze(0)
            new_population[(self.current_generation, i)] = child

        self.current_population = new_population
        self.current_generation += 1
        print("LENGTH OF NEW POPULATION: ", len(new_population))
        return new_population

class RejectionSamplingSearch(SearchAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        search_args = config["search_args"]

        # Single dynamic noise scale instead of three separate parameters
        self.noise_scale = search_args.get("noise_scale", 0.5)

        # Single unified adaptation parameter
        self.adaptation_strength = search_args.get("adaptation_strength", 0.3)

        # Search mode - replaces boolean flags
        # 0: basic, 1: anisotropic only, 2: adaptive only, 3: full features
        self.search_mode = search_args.get("search_mode", 3)

        # Runtime state
        self.round = 0
        self.best_noise = None
        self.previous_best_noise = None
        self.noise_history = []  # Track more history for better adaptation
        self.acceptance_rates = []
        self.score_history = []

        # Internal derived parameters
        self._min_scale_factor = 0.2  # Minimum scale relative to initial noise_scale
        self._max_scale_factor = 2.0  # Maximum scale relative to initial noise_scale

    def generate_noise_candidates(self, previous_data=None):
        """Generate noise candidates using adaptive rejection sampling."""
        latent_prep_fn = get_latent_prep_fn(self.config["pipeline_name"])
        num_samples = self.config["search_args"].get("num_samples", 10)

        self.round += 1

        # First round or no previous best: random initialization
        if previous_data is None or self.best_noise is None:
            return get_noises(
                max_seed=MAX_SEED,
                num_samples=num_samples,
                dtype=self.config["torch_dtype"],
                fn=latent_prep_fn,
                **self.config["pipeline_call_args"],
            )

        # Update best noise and search parameters
        self.previous_best_noise = self.best_noise
        self.best_noise = previous_data["best_noise"]

        # Track history for improved direction estimation
        self.noise_history.append(self.best_noise.clone())
        if len(self.noise_history) > 5:  # Keep history manageable
            self.noise_history.pop(0)

        # Track score history for all modes except basic
        if self.search_mode > 0:
            self.score_history.append(previous_data["best_score"])

        self._update_search_parameters(previous_data)

        # Generate candidates centered on best_noise
        return self._generate_centered_samples(num_samples)

    def _generate_centered_samples(self, num_samples):
        """Generate samples centered on best_noise with adaptive noise scale."""
        samples = {}
        for i in range(num_samples):
            seed = random.randint(0, MAX_SEED)

            # Use anisotropic noise in modes 1 and 3
            use_anisotropic = self.search_mode in [1, 3] and self.round > 2 and len(self.noise_history) >= 2

            if use_anisotropic:
                perturbation = self._generate_directional_noise()
            else:
                perturbation = torch.randn_like(self.best_noise) * self.noise_scale

            new_noise = torch.clamp(self.best_noise + perturbation, -3, 3)
            samples[seed] = new_noise
        return samples

    def _generate_directional_noise(self):
        """Generate noise that incorporates historical improvement directions."""
        # Base noise component
        base_noise = torch.randn_like(self.best_noise) * self.noise_scale

        # With history of at least 2 points, we can estimate direction
        if len(self.noise_history) >= 2:
            # Use exponential weighted average of previous directions for stability
            weighted_direction = torch.zeros_like(self.best_noise)
            weights = []

            # Calculate weights with exponential decay
            history_len = len(self.noise_history)
            for i in range(history_len - 1):
                # More recent directions get higher weight
                weights.append(0.7 ** (history_len - i - 2))

            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]

            # Compute weighted direction
            for i, w in enumerate(weights):
                # Get direction between consecutive noise points
                idx = i + 1
                direction = self.noise_history[idx] - self.noise_history[idx-1]
                dir_magnitude = torch.norm(direction)

                if dir_magnitude > 1e-6:
                    norm_direction = direction / dir_magnitude
                    weighted_direction += w * norm_direction * dir_magnitude

            # Blend random noise with directional component
            direction_weight = min(0.8, 0.4 + (self.round / 20))  # Increase direction influence over time
            random_weight = 1.0 - direction_weight

            return direction_weight * weighted_direction + random_weight * base_noise

        return base_noise  # Fall back to random noise

    def _update_search_parameters(self, previous_data):
        """Update noise scale and other parameters based on unified adaptation."""
        # Get acceptance rate from previous round
        acceptance_rate = previous_data.get("acceptance_rate", 0.0)
        self.acceptance_rates.append(acceptance_rate)

        # Skip adaptation in basic mode
        if self.search_mode == 0:
            return

        # Dynamic adaptation based on unified parameter
        if len(self.acceptance_rates) >= 2:
            # Calculate recent average acceptance rate
            window = min(5, len(self.acceptance_rates))
            recent_rate = np.mean(self.acceptance_rates[-window:])

            # Target acceptance derived from adaptation strength
            target_acceptance = 0.3 * self.adaptation_strength

            # Adjust noise scale
            if recent_rate < target_acceptance:
                # Increase scale to explore more when acceptance is too low
                adjustment = 1.0 + (0.1 * self.adaptation_strength)
                self.noise_scale = min(self.noise_scale * adjustment,
                                      self._max_scale_factor * self.noise_scale)
            else:
                # Decrease scale to focus search when acceptance is high
                adjustment = 1.0 - (0.1 * self.adaptation_strength)
                self.noise_scale = max(self.noise_scale * adjustment,
                                      self._min_scale_factor * self.noise_scale)

        # Adjust based on improvement trend (modes 2 and 3)
        if self.search_mode in [2, 3] and len(self.score_history) >= 3:
            # Calculate improvement trend
            recent_improvements = [
                (self.score_history[-(i)] - self.score_history[-(i+1)])
                for i in range(1, min(3, len(self.score_history)-1) + 1)
            ]
            improvement_trend = np.mean(recent_improvements)

            # If improvement slowing down, adapt more aggressively
            if improvement_trend <= 0:
                # Shrink search radius to focus on refinement
                self.noise_scale *= (1.0 - 0.2 * self.adaptation_strength)
            elif improvement_trend > 0.05 * self.adaptation_strength:
                # If strong improvement, slightly expand to capitalize
                self.noise_scale *= (1.0 + 0.05 * self.adaptation_strength)
