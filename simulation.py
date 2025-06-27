# language_evolution/simulation.py

"""
Main simulation script for Language Evolution.

This script initializes a population of languages and simulates their evolution
based on the principle of efficient communication, using the Information
Bottleneck (IB) framework for fitness calculation.
"""

import numpy as np
import config
from environment import Environment
import utils
import simulation_initializations
INIT_FUNCTION_NUMBER = config.INIT_FUNCTION_NUMBER


class LanguageEvolutionSimulation:
    """
    Manages the setup and execution of the evolutionary simulation.
    """

    def __init__(self):
        """
        Initializes the simulation environment and the population of languages.
        """
        print("Initializing simulation...")

        self.env = Environment.uniform(
            cielab_data_path=config.CIELAB_DATA_PATH,
            sigma=config.SIGMA
        )

        # Default: random lexicon size per language
        # self.population = self._initialize_population(
        #     num_languages=config.POPULATION_SIZE,
        #     num_meanings=self.env.num_meanings,
        #     min_lexicon_size=config.MIN_LEXICON_SIZE,
        #     max_lexicon_size=config.MAX_LEXICON_SIZE
        # )
        self.population = self._initialize_population(
            num_languages=config.POPULATION_SIZE,
            num_meanings=self.env.num_meanings,
            min_lexicon_size=config.MIN_LEXICON_SIZE,
            max_lexicon_size=config.MAX_LEXICON_SIZE
        )
        
        self.proportions = np.full(config.POPULATION_SIZE, 1.0 / config.POPULATION_SIZE)

        print(f"Initialization complete. Simulating {config.POPULATION_SIZE} languages.")

    def _initialize_population(self, num_languages, num_meanings, min_lexicon_size, max_lexicon_size):
        """
        Creates the initial population of languages using the selected initialization function.
        """
        env = self.env
        init_func = simulation_initializations.get_initialization_function_by_number(INIT_FUNCTION_NUMBER)
        if INIT_FUNCTION_NUMBER == 1:
            sharpness = getattr(config, 'VARYING_KMEANS_SHARPNESS', 100.0)
            return init_func(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env, sharpness=sharpness)
        elif INIT_FUNCTION_NUMBER == 2:
            sharpness_min = getattr(config, 'VARYING_SHARPNESS_MIN', 10.0)
            sharpness_max = getattr(config, 'VARYING_SHARPNESS_MAX', 200.0)
            return init_func(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env, sharpness_min=sharpness_min, sharpness_max=sharpness_max)
        else:
            return init_func(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env)

    def run(self):
        """
        Executes the main evolutionary loop for a specified number of generations.
        """
        print("\n--- Starting Evolution ---")
        # Report initial fitness values for all languages, including strategies and parameters
        initial_fitness_scores = [
            utils.calculate_fitness(
                encoder=lang.encoder,
                cognitive_source=self.env.cognitive_source,
                meaning_space=self.env.meaning_space,
                sigma=self.env.sigma,
                beta=config.BETA
            ) for lang in self.population
        ]
        print("Initial fitness values for all languages:")
        for i, (lang, fitness) in enumerate(zip(self.population, initial_fitness_scores)):
            print(self._get_language_report(lang, i, fitness=fitness))

        for generation in range(config.NUM_GENERATIONS):
            self._step(generation)

        print("\n--- Simulation Finished ---")
        self._report_final_results()

    def _get_language_report(self, lang, idx, fitness=None, accuracy=None, complexity=None, proportion=None):
        s = f"  Language {idx:2d}: "
        if proportion is not None:
            s += f"Proportion = {proportion:.4%} | "
        s += f"Strategy: {lang.strategy} | Lexicon Size: {lang.lexicon_size}"
        if lang.strategy == 1 and hasattr(lang, 'strategy_params'):
            sharpness = lang.strategy_params.get('sharpness', None)
            if sharpness is not None:
                s += f" | Sharpness: {sharpness:.2f}"
        if fitness is not None:
            s += f" | Fitness: {float(fitness):.4f}"
        if accuracy is not None:
            s += f" | Accuracy: {accuracy:.4f}"
        if complexity is not None:
            s += f" | Complexity: {complexity:.4f}"
        return s

    def _step(self, generation):
        """
        Performs a single step of the evolutionary simulation.
        """
        # 1. Calculate the fitness for each language in the population
        fitness_scores = np.array([
            utils.calculate_fitness(
                encoder=lang.encoder,
                cognitive_source=self.env.cognitive_source,
                meaning_space=self.env.meaning_space,
                sigma=self.env.sigma,
                beta=config.BETA
            ) for lang in self.population
        ])

        # The replicator equation requires positive fitness values. We shift all
        # scores to ensure they are positive while preserving relative differences.
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            fitness_scores -= (min_fitness - config.SHIFT_VALUE)  # Shift to be positive using config value

        # 2. Calculate the average population fitness (phi)
        avg_fitness_phi = np.sum(self.proportions * fitness_scores)

        # 3. Update the population proportions using the replicator equation
        # x_i(t+1) = x_i(t) * f_i / phi
        if avg_fitness_phi > 0:
            self.proportions = self.proportions * fitness_scores / avg_fitness_phi
            # Re-normalize to handle any floating point inaccuracies
            self.proportions /= np.sum(self.proportions)

        # Report progress periodically
        if (generation + 1) % 100 == 0:
            self._report_progress(generation, avg_fitness_phi)

    def _report_progress(self, generation, avg_fitness):
        """Prints a summary of the current state of the simulation."""
        dominant_lang_idx = np.argmax(self.proportions)
        dominant_lang_prop = self.proportions[dominant_lang_idx]
        dominant_lang = self.population[dominant_lang_idx]
        print(
            f"Generation: {generation + 1:5d} | "
            f"Avg Fitness: {avg_fitness:10.4f} | "
            f"Dominant Language: #{dominant_lang_idx:2d} ({dominant_lang_prop:.2%}) | "
            f"Strategy: {dominant_lang.strategy} | "
            f"Lexicon Size: {dominant_lang.lexicon_size}"
            + (f" | Sharpness: {dominant_lang.strategy_params.get('sharpness', None):.2f}" if dominant_lang.strategy == 1 and hasattr(dominant_lang, 'strategy_params') and 'sharpness' in dominant_lang.strategy_params else "")
        )

    def _report_final_results(self):
        """Prints the final state of the population after the simulation."""
        dominant_lang_idx = np.argmax(self.proportions)
        lowest_lang_idx = np.argmin(self.proportions)
        print("\nFinal Population Proportions:")
        for i, prop in enumerate(self.proportions):
            print(self._get_language_report(self.population[i], i, proportion=prop))

        dominant_lang = self.population[dominant_lang_idx]
        lowest_lang = self.population[lowest_lang_idx]
        # Calculate fitness, accuracy, complexity for both
        dom_fit = utils.calculate_fitness(dominant_lang.encoder, self.env.cognitive_source, self.env.meaning_space, self.env.sigma, config.BETA)
        dom_acc, _ = utils.calculate_accuracy(dominant_lang.encoder, self.env.cognitive_source, self.env.meaning_space, self.env.sigma)
        dom_comp = utils.calculate_complexity(dominant_lang.encoder, self.env.cognitive_source)
        low_fit = utils.calculate_fitness(lowest_lang.encoder, self.env.cognitive_source, self.env.meaning_space, self.env.sigma, config.BETA)
        low_acc, _ = utils.calculate_accuracy(lowest_lang.encoder, self.env.cognitive_source, self.env.meaning_space, self.env.sigma)
        low_comp = utils.calculate_complexity(lowest_lang.encoder, self.env.cognitive_source)

        print(f"\nDominant Language: #{dominant_lang_idx}")
        print(self._get_language_report(dominant_lang, dominant_lang_idx, fitness=dom_fit, accuracy=dom_acc, complexity=dom_comp, proportion=self.proportions[dominant_lang_idx]))
        print(f"\nLowest Proportion Language: #{lowest_lang_idx}")
        print(self._get_language_report(lowest_lang, lowest_lang_idx, fitness=low_fit, accuracy=low_acc, complexity=low_comp, proportion=self.proportions[lowest_lang_idx]))
