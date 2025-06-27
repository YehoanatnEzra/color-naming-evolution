# language_evolution/config.py

"""
Configuration file for the language evolution simulation.
All static parameters and constants are defined here.
"""

import numpy as np

# --- Data Paths ---
# Path to the CSV file containing the CIELAB coordinates for the 330 WCS color chips.
# The CSV should have columns like: 'chip_id', 'L*', 'a*', 'b*'
CIELAB_DATA_PATH = "data/wcs_cielab_data.csv"

# --- Environment Parameters ---
# The meaning of a color is a Gaussian distribution in CIELAB space.
# SIGMA defines the standard deviation of this Gaussian.
SIGMA = 8.0  # Corresponds to sigma^2 = 64

# --- IB Fitness Parameters ---
# The beta parameter controls the trade-off between accuracy and complexity.
# A key parameter for experimentation.
BETA = 1.09

# --- Simulation Parameters ---
# The number of generations to run the simulation for.
NUM_GENERATIONS = 700

# The number of competing languages in the population.
POPULATION_SIZE = 10

# The maximum number of words allowed in a language's lexicon.
MAX_LEXICON_SIZE = 100
# The minimum number of words allowed in a language's lexicon.
MIN_LEXICON_SIZE = 10

# Fitness shift value to ensure all fitness values are positive and to control selection pressure
SHIFT_VALUE = 30

# Sharpness range for clustered encoder strategy (min, max)
CLUSTERED_SHARPNESS_RANGE = (10, 200)

# Simulation initialization function number (see simulation_initializations.py)
# 0: all_random_parameters
# 1: varying_lexicon_kmeans
# 2: varying_sharpness_kmeans_same_lexicon
INIT_FUNCTION_NUMBER = 2
# Sharpness for varying_lexicon_kmeans initialization
VARYING_KMEANS_SHARPNESS = 100.0
VARYING_SHARPNESS_MIN = 10.0
VARYING_SHARPNESS_MAX = 200.0