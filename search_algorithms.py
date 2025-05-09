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
