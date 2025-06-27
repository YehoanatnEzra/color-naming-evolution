# environment.py

import numpy as np
import pandas as pd


class Environment:
    """
    Represents the environment of color meanings,
    with different factory methods for variants.
    """

    def __init__(self, cielab_data_path, sigma):
        """
        Private constructor: loads the CIELAB coords and
        initializes a uniform cognitive source.
        """
        self.cielab_coords = self._load_cielab_coords(cielab_data_path)
        self.num_meanings  = len(self.cielab_coords)
        self.sigma         = sigma

        # meaning_space: centers of Gaussian meanings in CIELAB
        self.meaning_space     = self.cielab_coords
        # uniform cognitive source by default
        self.cognitive_source  = np.full(self.num_meanings, 1/self.num_meanings)


    @staticmethod
    def _load_cielab_coords(file_path):
        """Loads CIELAB coords (L*, a*, b*) from CSV or returns random dummy."""
        try:
            df = pd.read_csv(file_path)
            return df[['L*','a*','b*']].values
        except FileNotFoundError:
            print(f"CIELAB file not found at {file_path}; using dummy data.")
            return np.random.rand(330, 3)

    @classmethod
    def uniform(cls, cielab_data_path, sigma):
        """
        Factory for uniform environment (default).
        """
        return cls(cielab_data_path, sigma)

    @classmethod
    def beach(cls, cielab_data_path, sigma, weight=5.0, pct=75):
        """
        Factory for “beach” environment:
        biases towards high b* (blue hues).
        """
        env = cls(cielab_data_path, sigma)
        # choose top pct percentile of b* channel
        thresh    = np.percentile(env.meaning_space[:,2], pct)
        blue_mask = env.meaning_space[:,2] >= thresh
        weights   = np.where(blue_mask, weight, 1.0)
        env.cognitive_source = weights / weights.sum()
        return env

    @classmethod
    def forest(cls, cielab_data_path, sigma, weight=5.0, pct=25):
        """
        Factory for “forest” environment:
        biases towards low a* (green hues).
        """
        env = cls(cielab_data_path, sigma)
        thresh     = np.percentile(env.meaning_space[:,1], pct)
        green_mask = env.meaning_space[:,1] <= thresh
        weights    = np.where(green_mask, weight, 1.0)
        env.cognitive_source = weights / weights.sum()
        return env

    @classmethod
    def urban(cls, cielab_data_path, sigma, weight=5.0, pct=25):
        """
        Factory for “urban” environment:
        biases towards low-chroma (gray/neutral) colors.
        """
        env = cls(cielab_data_path, sigma)
        # chroma = sqrt(a*^2 + b*^2); low chroma = neutral grays
        chroma = np.linalg.norm(env.meaning_space[:, 1:3], axis=1)
        thresh = np.percentile(chroma, pct)
        gray_mask = chroma <= thresh  # bottom pct% by chroma
        weights = np.where(gray_mask, weight, 1.0)
        env.cognitive_source = weights / weights.sum()
        return env

    @classmethod
    def sunset(cls, cielab_data_path, sigma, weight=5.0, pct=75):
        """
        Factory for “sunset” environment:
        biases towards high a* (red hues) and/or high b* (yellow hues).
        """
        env = cls(cielab_data_path, sigma)
        # pick top pct percentile of a* (reds) and b* (yellows)
        thresh_a = np.percentile(env.meaning_space[:, 1], pct)
        thresh_b = np.percentile(env.meaning_space[:, 2], pct)
        red_mask = env.meaning_space[:, 1] >= thresh_a
        yellow_mask = env.meaning_space[:, 2] >= thresh_b
        # any color that is either very red or very yellow
        mask = red_mask | yellow_mask
        weights = np.where(mask, weight, 1.0)
        env.cognitive_source = weights / weights.sum()
        return env


