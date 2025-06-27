import numpy as np
import random
import config
import encoder_strategies
from language import Language

# Mapping from integer to initialization function
# 0: all_random_parameters
# 1: varying_lexicon_kmeans
# 2: varying_sharpness_kmeans_same_lexicon
INITIALIZATION_FUNCTIONS = {}


def all_random_parameters(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env):
    """
    Default: random strategy, random lexicon size, random sharpness for clustered.
    """
    population = []
    for _ in range(num_languages):
        lexicon_size = random.randint(min_lexicon_size, max_lexicon_size)
        strategy = random.choice(encoder_strategies.AVAILABLE_STRATEGIES)
        strategy_params = {}
        if strategy == 1:
            strategy_params['meaning_space'] = env.meaning_space
            strategy_params['sharpness'] = random.uniform(*config.CLUSTERED_SHARPNESS_RANGE)
        population.append(Language(num_meanings, lexicon_size, strategy, strategy_params))
    return population


INITIALIZATION_FUNCTIONS[0] = all_random_parameters


def varying_lexicon_kmeans(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env, sharpness=100.0):
    """
    All languages use strategy 1 (clustered), same sharpness, and lexicon sizes increase controllably.
    """
    population = []
    lexicon_sizes = [int(round(min_lexicon_size + i * (max_lexicon_size - min_lexicon_size) / (num_languages - 1))) for i in range(num_languages)]
    for lexicon_size in lexicon_sizes:
        strategy = 1
        strategy_params = {'meaning_space': env.meaning_space, 'sharpness': sharpness}
        population.append(Language(num_meanings, lexicon_size, strategy, strategy_params))
    return population


INITIALIZATION_FUNCTIONS[1] = varying_lexicon_kmeans


def varying_sharpness_kmeans_same_lexicon(num_languages, num_meanings, min_lexicon_size, max_lexicon_size, env, sharpness_min=10.0, sharpness_max=200.0):
    """
    All languages use strategy 1 (clustered), same lexicon size, but each has a unique sharpness value (linearly spaced).
    """
    population = []
    # Use the max_lexicon_size for all languages (or min_lexicon_size, or a config value)
    lexicon_size = int(round((max_lexicon_size + min_lexicon_size) / 2))
    sharpness_values = np.linspace(sharpness_min, sharpness_max, num_languages)
    for sharpness in sharpness_values:
        strategy = 1
        strategy_params = {'meaning_space': env.meaning_space, 'sharpness': float(sharpness)}
        population.append(Language(num_meanings, lexicon_size, strategy, strategy_params))
    return population


INITIALIZATION_FUNCTIONS[2] = varying_sharpness_kmeans_same_lexicon


def get_initialization_function_by_number(init_number):
    if init_number not in INITIALIZATION_FUNCTIONS:
        raise ValueError(f"Unknown initialization number: {init_number}")
    return INITIALIZATION_FUNCTIONS[init_number] 