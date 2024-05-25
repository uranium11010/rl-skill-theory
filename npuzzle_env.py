from math import factorial

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces


class NPuzzleEnv(gym.Env):
    gods_numbers = {  # https://en.wikipedia.org/wiki/15_Puzzle
        2: 6,
        3: 31,
        4: 80,
    }
    def __init__(self, size: int):
        super().__init__()
        self._size = size
        self._solved_grid = list(range(1, self._size ** 2)) + [0]
        self._grid: list[int]
        self._pos: int

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return gym.spaces.Box(0, self._size ** 2 - 1, shape=(self._size, self._size), dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._grid = self._solved_grid[:]
        self._pos = len(self._grid) - 1
        scramble_len = self.np_random.integers(1, self.gods_numbers[self._size] + 1)
        for i in range(scramble_len):
            action = self.np_random.choice(self._get_legal_actions(self._size, self._pos))
            self._pos = self._step(self._size, self._grid, self._pos, action)
        return np.array(self._grid, dtype=np.int64).reshape(self._size, self._size), {}

    def step(self, action):
        self._pos = self._step(self._size, self._grid, self._pos, action)
        terminated = self._grid == self._solved_grid
        reward = float(terminated)
        return np.array(self._grid, dtype=np.int64).reshape(self._size, self._size), reward, terminated, False, {}
    
    @classmethod
    def _get_legal_actions(cls, size, pos):
        legal_actions = []
        if pos >= size:
            legal_actions.append(0)
        if pos % size < size - 1:
            legal_actions.append(1)
        if pos < size * (size - 1):
            legal_actions.append(2)
        if pos % size >= 1:
            legal_actions.append(3)
        return legal_actions

    @classmethod
    def _step(cls, size, grid, pos, action):
        new_pos = pos
        if action == 0:  # empty square up
            if pos >= size:
                new_pos = pos - size
        elif action == 1:  # empty square right
            if pos % size < size - 1:
                new_pos = pos + 1
        elif action == 2:  # empty square down
            if pos < size * (size - 1):
                new_pos = pos + size
        elif action == 3:  # empty square left
            if pos % size >= 1:
                new_pos = pos - 1
        else:
            raise Exception(f"Action {action} out of bounds")
        grid[pos], grid[new_pos] = grid[new_pos], grid[pos]
        return new_pos

    def __str__(self):
        out_str = "CURRENT GRID:\n"
        cell_width = len(str(self._size ** 2))
        for i in range(self._size):
            out_str += '|' + '|'.join([f"{{:{cell_width}}}".format(self._grid[self._size * i + j]) for j in range(self._size)]) + '|\n'
        return out_str
