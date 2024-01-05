import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class MineSweeper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 640  # The size of the PyGame window
        SCREEN_WIDTH = 640
        SCREEN_HEIGHT = 640
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        #Watch game screen
        # RGB or gray-scale?
        self.observation_space = spaces.Box(low=0,high=255,shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3),dtype=np.uint8)

        # Choose any tile on 20x20 grid
        self.action_space = spaces.Discrete(400)

    def step(self, action):
        a = 2
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        a = 2
        return observation, info

    def render(self):
        a = 2

    def close(self):
        a = 2