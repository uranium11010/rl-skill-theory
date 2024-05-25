import os
from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import copy
from queue import Queue
from tqdm import tqdm
import time
from itertools import permutations
from math import factorial
from typing import Callable, Optional, Union, TypeVar
import warnings

import pickle as pkl
import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy import optimize, sparse
import torch
from torch import nn

import gymnasium as gym
from gymnasium.envs.toy_text import CliffWalkingEnv
import rubiks_cube_gym
from rubiks_cube_gym.envs import RubiksCube222Env
from compile_env import CompILEEnv
from npuzzle_env import NPuzzleEnv

from utils import flexible_batch_dims


CACHE_DIR = os.getenv("RLDIFFENV_CACHE_DIR", ".cache/")
RUBIKS_DIR = os.path.dirname(rubiks_cube_gym.__file__)


class RLDiffEnv(ABC, gym.Env):
    def __init__(self):
        # cache for (public) immutable properties
        self._avg_sol_len = None
        self._max_sol_len = None
        self._transition_table = None
        self._rewards_table = None
        self._relevant_inds = None
        self._state_dist = None
        self._state_dist_entropy = None

        self._cache = {}  # cache for any private immutable properties
                          # that are preserved across wrapping (other than AbsWrapper)

    @property
    def config_string(self) -> Optional[str]:
        return None

    @abstractproperty
    def state_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_array_obs(self) -> NDArray:
        raise NotImplementedError()

    @abstractmethod
    def get_state_embedder(self) -> nn.Module:
        raise NotImplementedError()

    @abstractproperty
    def terminal_states(self) -> NDArray[np.int64]:
        raise NotImplementedError()

    @property
    def transition_table(self) -> NDArray[np.int64]:
        if self._transition_table is None:
            self._transition_table = self._get_transition_table(self)
        return self._transition_table

    @property
    def rewards_table(self) -> NDArray:
        if self._rewards_table is None:
            self._rewards_table = self._get_rewards_table(self)
        return self._rewards_table

    @property
    def relevant_inds(self) -> Optional[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        if self._relevant_inds is None:
            self._relevant_inds = self._get_relevant_inds(self)
        return self._relevant_inds

    @property
    def state_dist(self) -> NDArray:
        if self._state_dist is None:
            self._state_dist = self._get_state_dist()
        return self._state_dist

    @property
    def state_dist_entropy(self) -> float:
        if self._state_dist_entropy is None:
            self._state_dist_entropy = self._get_state_dist_entropy()
        return self._state_dist_entropy

    @property
    def avg_sol_len(self) -> float:
        if self._avg_sol_len is None:
            self._avg_sol_len = self._get_avg_sol_len(self)
        return self._avg_sol_len

    @property
    def max_sol_len(self) -> float:
        if self._max_sol_len is None:
            self._max_sol_len = self._get_max_sol_len(self)
        return self._max_sol_len

    def get_sum_inv_subopt_gaps(self, gamma: float, **kwargs) -> float:
        return self._get_sum_inv_subopt_gaps(self, gamma, **kwargs)

    def get_incompress(self) -> float:
        def get_incompress_eps(eps):
            numerator = self.state_dist_entropy - np.log((1 - eps) / eps)
            denominator = self.avg_sol_len * np.log(self.action_space.n / (1 - eps))
            return numerator / denominator
        best_eps = optimize.minimize(lambda eps: -get_incompress_eps(eps),
                x0 = 1 / (1 + get_incompress_eps(0.5) * self.avg_sol_len),
                bounds = [(0, 1)]).x[0]
        return get_incompress_eps(best_eps)

    def get_expldiff(self, eps: float, take_log: bool=True, **kwargs) -> Optional[float]:
        return self._get_expldiff(self, eps, take_log, **kwargs)

    @abstractclassmethod
    def _get_transition_table(cls, env) -> NDArray[np.int64]:
        raise NotImplementedError()

    @abstractclassmethod
    def _get_rewards_table(cls, env) -> NDArray:
        raise NotImplementedError()

    @classmethod
    def _get_relevant_inds(cls, env) -> Optional[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        return None

    @abstractmethod
    def _get_state_dist(self) -> NDArray:
        raise NotImplementedError()

    @abstractmethod
    def _get_state_dist_entropy(self) -> float:
        raise NotImplementedError()

    @classmethod
    def _get_avg_sol_len(cls, env) -> float:
        if "_distances" not in env._cache:
            env._cache["_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table)
        mask = env.state_dist > 0
        masked_weighted_distances = env.state_dist[mask] * env._cache["_distances"][mask]
        return np.sum(masked_weighted_distances).item()

    @classmethod
    def _get_max_sol_len(cls, env) -> float:
        if "_distances" not in env._cache:
            env._cache["_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table)
        masked_distances = env._cache["_distances"][env.state_dist > 0]
        return np.max(masked_distances).item()

    @classmethod
    def _get_sum_inv_subopt_gaps(cls, env, gamma: float, threshold: float=1e-11) -> float:
        if "_distances" not in env._cache:
            env._cache["_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table)
        distances = env._cache["_distances"]
        values = gamma ** (distances - 1)
        q_values = gamma * values[env.transition_table]
        subopt_gaps = values.reshape((-1, 1)) - q_values
        subopt_gaps[env.terminal_states] = np.inf
        subopt_gaps[np.abs(subopt_gaps) < threshold] = np.inf
        assert np.all(subopt_gaps >= threshold)
        assert np.all(values[np.isinf(distances)] == 0) and np.all(q_values[np.isinf(distances)] == 0)
        inv_subopt_gaps = 1 / subopt_gaps
        return np.sum(inv_subopt_gaps).item()

    @abstractclassmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, **kwargs) -> float:
        raise NotImplementedError()


class RLDiffCliffWalkingEnv(RLDiffEnv, CliffWalkingEnv):
    def __init__(self, *args, **kwargs):
        super(RLDiffEnv, self).__init__(*args, **kwargs)
        self.goal_state_index = self.nS - 1
        super().__init__()

    def step(self, action):
        next_state, _, terminated, truncated, info = super().step(action)
        reward = float(terminated)
        return next_state, reward, terminated, truncated, info

    @property
    def state_index(self):
        return self.s

    def get_array_obs(self) -> NDArray:
        """ Return obs array as multihot vector at every cell. 0 is player, 1 is goal, 2 is cliff """
        obs = np.zeros((self.nS, 3))
        obs[self.s, 0] = 1
        obs[self.goal_state_index, 1] = 1
        obs = obs.reshape((*self.shape, 3))
        obs[:,:,2] = self._cliff
        return obs

    def get_state_embedder(self, embed_dim=16, hidden_dim=32, kernel_size=3, padding=1):
        class StateEmbedder(nn.Module):
            def __init__(self, embed_dim, hidden_dim, kernel_size, padding):
                super().__init__()

                self.embed_dim = embed_dim
                self.model = nn.Sequential(
                    nn.Conv2d(in_channels=3,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Flatten(start_dim=-3, end_dim=-1),
                    nn.Linear(in_features=(4 + 4 * padding - 2 * (kernel_size - 1)) * (12 + 4 * padding - 2 * (kernel_size - 1)) * hidden_dim,
                              out_features=hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_dim,
                              out_features=embed_dim)
                )

            @flexible_batch_dims(3)
            def forward(self, obs):
                obs = obs.swapaxes(-1, -2).swapaxes(-2, -3)
                return self.model(obs)

        return StateEmbedder(embed_dim, hidden_dim, kernel_size, padding)

    @property
    def terminal_states(self):
        return np.array([self.goal_state_index], dtype=np.int64)

    @classmethod
    def _get_transition_table(cls, env):
        """ Get deterministic transition table T(s, a) """
        env = copy.deepcopy(env)
        env.reset()
        transition_table = np.empty((env.unwrapped.nS, env.action_space.n), dtype=np.int64)
        for state in range(env.unwrapped.nS):
            for action in range(env.action_space.n):
                env.unwrapped.s = state
                env.step(action)
                transition_table[state, action] = env.unwrapped.s
        return transition_table

    @classmethod
    def _get_relevant_inds(cls, env):
        return bfs(env.unwrapped.start_state_index,
                   env.unwrapped.goal_state_index,
                   env.transition_table,
                   start_value=(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
                   value_update=lambda relevant_inds, state, action, next_state:
                                (np.concatenate((relevant_inds[0], np.array([state], dtype=np.int64))),
                                 np.concatenate((relevant_inds[1], np.array([action], dtype=np.int64)))),
                   default_value=(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))

    @classmethod
    def _get_rewards_table(cls, env):
        """ Get deterministic rewards table R(s, a) """
        env = copy.deepcopy(env)
        env.reset()
        rewards_table = np.empty((env.unwrapped.nS, env.action_space.n), dtype=np.int64)
        for state in range(env.unwrapped.nS):
            for action in range(env.action_space.n):
                env.unwrapped.s = state
                rewards_table[state, action] = env.step(action)[1]
        return rewards_table

    def _get_state_dist(self):
        state_dist = np.zeros(self.observation_space.n)
        state_dist[36] = 1
        return state_dist

    def _get_state_dist_entropy(self):
        return 0.

    def get_incompress(self) -> float:
        return 0.

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, iterate=False, rtol: float=1e-7, use_torch: bool=False):
        """ `rtol` only applies to `iterate` being True (i.e., iteratively solve for q) """
        prob_table = get_prob_table(env.transition_table,
                                    env.unwrapped.goal_state_index,
                                    eps,
                                    iterate=iterate,
                                    rtol=rtol,
                                    use_torch=use_torch)
        np_or_torch = torch if use_torch else np
        transform = lambda x: -np_or_torch.log(x) if take_log else 1 / x
        return transform(prob_table[env.unwrapped.start_state_index]).item()


class RLDiffRubiksCube222Env(RLDiffEnv, RubiksCube222Env):
    COLOR_TO_INDEX = {"W": 0, "Y": 1, "G": 2, "B": 3, "R": 4, "O": 5}
    CORNERS = [(0, 11, 4), (2, 5, 6), (3, 7, 8), (1, 9, 10), (20, 14, 13), (21, 16, 15), (23, 18, 17), (22, 12, 19)]
    CORNERS_REDUCED = ["WBO", "WOG", "WGR", "WRB", "YGO", "YRG", "YBR", "YOB"]
    CORNER_TO_INDICES = {
            **{corner_reduced: corner for corner_reduced, corner in zip(CORNERS_REDUCED, CORNERS)},
            **{corner_reduced[1:] + corner_reduced[:1]: corner[1:] + corner[:1]
                for corner_reduced, corner in zip(CORNERS_REDUCED, CORNERS)},
            **{corner_reduced[2:] + corner_reduced[:2]: corner[2:] + corner[:2]
                for corner_reduced, corner in zip(CORNERS_REDUCED, CORNERS)},
            }

    def __init__(self, *args, **kwargs):
        super(RLDiffEnv, self).__init__(*args, **kwargs)
        self.goal_state_index = 0
        super().__init__()

    def generate_scramble(self):
        """
        Random scramble generation
        Originally scramble length is always 11 and agent has trouble with exploration,
        so change scramble length to uniform between 1..11
        """
        scramble_len_target = self.np_random.choice(11) + 1  # uniform 1..11

        scramble_len = 0
        prev_move = None
        scramble = ""
        moves = ['F', 'R', 'U']
        move_type = ['', '2', "'"]

        while scramble_len < scramble_len_target:
            move = self.np_random.choice(moves)
            while move == prev_move:
                move = self.np_random.choice(moves)
            scramble += move + self.np_random.choice(move_type) + " "
            prev_move = move
            scramble_len += 1

        return scramble[:-1]

    def step(self, action):
        next_state, _, terminated, truncated, info = super().step(action)
        reward = float(terminated)
        return next_state, reward, terminated, truncated, info

    @property
    def state_index(self):
        return self.cube_state

    def get_array_obs(self) -> NDArray[np.int32]:
        """ Return obs array as color index at every tile. """
        return np.array([self.COLOR_TO_INDEX[color] for color in self.cube_reduced])

    def get_state_embedder(self, embed_dim=32, hidden_dim=64):
        class StateEmbedder(nn.Module):
            def __init__(self, embed_dim, hidden_dim):
                super().__init__()

                self.embed_dim = embed_dim
                self.model = nn.Sequential(
                    nn.Embedding(num_embeddings=6, embedding_dim=hidden_dim),
                    nn.Flatten(start_dim=-2, end_dim=-1),
                    nn.Linear(in_features=24 * hidden_dim,
                              out_features=hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_dim,
                              out_features=hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_dim,
                              out_features=hidden_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_dim,
                              out_features=embed_dim)
                )

            @flexible_batch_dims(1)
            def forward(self, obs):
                return self.model(obs)

        return StateEmbedder(embed_dim, hidden_dim)

    @property
    def terminal_states(self):
        return np.array([self.goal_state_index], dtype=np.int64)

    @classmethod
    def _cube_from_reduced(cls, reduced_state):
        cube = np.empty(shape=(24,), dtype=np.uint8)
        for corner in cls.CORNERS:
            corner_colors = ''.join(reduced_state[idx] for idx in corner)
            corner_idxs = cls.CORNER_TO_INDICES[corner_colors]
            for loc, idx in zip(corner, corner_idxs):
                cube[loc] = idx
        return cube

    @classmethod
    def _get_transition_table(cls, env):
        """ Get deterministic transition table T(s, a) """
        if env.action_space.n == env.unwrapped.action_space.n:
            # `env` is base environment (without abstractions)
            transitions_path = os.path.join(CACHE_DIR, "rubiks_cube_222_transitions.npy")
            if os.path.isfile(transitions_path):
                return np.load(transitions_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(os.path.join(RUBIKS_DIR, "envs/rubiks_cube_222_states_FRU.pickle"), 'rb') as f:
                cube_states = pkl.load(f)
            cube_reduced_states = [None] * len(cube_states)
            for reduced_state, idx in cube_states.items():
                cube_reduced_states[idx] = reduced_state
            env = copy.deepcopy(env.unwrapped)
            env.reset()
            transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
            print("Computing base environment transition table...")
            for state in tqdm(range(env.observation_space.n)):
                for action in range(env.action_space.n):
                    env.cube = cls._cube_from_reduced(cube_reduced_states[state])
                    env.step(action)
                    transition_table[state, action] = env.cube_state
            np.save(transitions_path, transition_table)
            return transition_table
        # `env` is environment with abstractions
        transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
        transition_table[:,:env.unwrapped.action_space.n] = env.unwrapped.transition_table
        for state in range(env.observation_space.n):
            for ab, action in zip(env.abstractions, range(env.unwrapped.action_space.n, env.action_space.n)):
                next_state = state
                for base_action in ab:
                    next_state = transition_table[next_state, base_action.name]
                transition_table[state, action] = next_state
        return transition_table

    @classmethod
    def _get_rewards_table(cls, env):
        """ Get deterministic rewards table R(s, a) """
        return (env.transition_table == env.unwrapped.goal_state_index).astype(float)

    def _get_state_dist(self):
        half_turn_transition_table = self.transition_table[self.transition_table].diagonal(axis1=1, axis2=2)
        ccw_turn_transition_table = half_turn_transition_table[self.transition_table].diagonal(axis1=1, axis2=2)
        # Get T[s, n, f] = result of turning face f clockwise n times to s
        scramble_transition_table = np.stack((self.transition_table, half_turn_transition_table, ccw_turn_transition_table), axis=1)
        max_scramble_len = 11
        prob_table = np.zeros((max_scramble_len + 1, self.observation_space.n, 3))
        prob_table[0, self.goal_state_index] = 1/3
        diff_face_mask = np.ones((3, 3), dtype=bool) ^ np.identity(3, dtype=bool)
        for scramble_len in range(1, max_scramble_len + 1):
            # Get move_probs[s, n, f, f'] = P[l-1, T[s, n, f], f']
            move_probs = prob_table[scramble_len - 1, scramble_transition_table]
            # Take mean over n and f' != f
            prob_table[scramble_len] = np.mean(move_probs, axis=(1, 3), where=diff_face_mask)
        state_dist = np.sum(prob_table, axis=(0, 2))
        state_dist[self.goal_state_index] -= 1
        state_dist /= max_scramble_len
        assert abs(np.sum(state_dist) - 1) < 1e-7
        return state_dist

    def _get_state_dist_entropy(self):
        return -np.sum(self.state_dist * np.log(self.state_dist)).item()

    @classmethod
    def _get_avg_sol_len(cls, env) -> float:
        if "_inv_distances" not in env._cache:
            env._cache["_inv_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table, inverted_transitions=True)
        weighted_inv_distances = env.state_dist * env._cache["_inv_distances"]
        return np.sum(weighted_inv_distances).item()

    @classmethod
    def _get_max_sol_len(cls, env) -> float:
        if "_inv_distances" not in env._cache:
            env._cache["_inv_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table, inverted_transitions=True)
        return np.max(env._cache["_inv_distances"]).item()

    @classmethod
    def _get_sum_inv_subopt_gaps(cls, env, gamma: float, threshold: float=1e-11) -> float:
        if "_inv_distances" not in env._cache:
            env._cache["_inv_distances"] = get_distances(env.unwrapped.goal_state_index, env.transition_table, inverted_transitions=True)
        inv_distances = env._cache["_inv_distances"]
        values = gamma ** (inv_distances - 1)
        q_values = gamma * values[env.transition_table]
        subopt_gaps = values.reshape((-1, 1)) - q_values
        subopt_gaps[env.terminal_states] = np.inf
        subopt_gaps[np.abs(subopt_gaps) < threshold] = np.inf
        assert np.all(subopt_gaps >= threshold)
        inv_subopt_gaps = 1 / subopt_gaps
        return np.sum(inv_subopt_gaps).item()

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, iterate=True, rtol: float=1e-3):
        """ `rtol` only applies to `iterate` being True (i.e., iteratively solve for q) """
        prob_table = get_prob_table(env.transition_table,
                                    env.unwrapped.goal_state_index,
                                    eps,
                                    iterate=iterate,
                                    rtol=rtol,
                                    use_torch=True)
        transform = lambda x: -torch.log(x) if take_log else 1 / x
        return torch.sum(torch.tensor(env.state_dist) * transform(prob_table)).item()


class RLDiffCompILEEnv(RLDiffEnv, CompILEEnv):
    def __init__(self, seed=1, width=10, height=10, num_objects=6, visit_length=3):
        super(RLDiffEnv, self).__init__(seed=seed, max_steps=float('inf'),
                                        width=width, height=height, num_objects=num_objects,
                                        visit_length=visit_length, sparse_reward=True)

        self._config_string = f"s{seed}w{width}h{height}n{num_objects}v{visit_length}"

        self._legal_positions = []
        for x, col in enumerate(self._grid):
            for y, obj in enumerate(col):
                if obj is None or obj.type != "Wall":
                    self._legal_positions.append((x, y))
        self._position2idx = {pos: idx for idx, pos in enumerate(self._legal_positions)}

        self._taken_objects_list = []
        self._taken_objects_dict = {}
        goal_indices = set()
        idx = 0
        for taken_objects in permutations(range(-1, len(self._object_types))):
            taken_objects = taken_objects[:taken_objects.index(-1)]
            if taken_objects not in self._taken_objects_dict:
                self._taken_objects_list.append(taken_objects)
                self._taken_objects_dict[taken_objects] = idx
                if len(taken_objects) == len(self._items_to_visit):
                    finished_taking_objects = True
                    for obj, item_to_visit in zip(taken_objects, self._items_to_visit):
                        if self._object_types[obj] != item_to_visit:
                            finished_taking_objects = False
                            break
                    if finished_taking_objects:
                        goal_indices.add(idx * len(self._legal_positions)
                                + self._position2idx[self._object_positions[taken_objects[-1]]])
                idx += 1

        self._idx_conversion = {goal_index: 0 for goal_index in goal_indices}
        self._rev_idx_conversion = {}
        new_idx = 1
        for idx in range(len(self._taken_objects_dict) * len(self._legal_positions)):
            if idx not in goal_indices:
                self._idx_conversion[idx] = new_idx
                self._rev_idx_conversion[new_idx] = idx
                new_idx += 1
        self._obs_space_size = new_idx

        self.goal_state_index = 0

        self._taken_objects = []

        super().__init__()

    @property
    def observation_space(self):
        return gym.spaces.Discrete(self._obs_space_size)

    @property
    def config_string(self):
        return self._config_string

    def reset(self, *, seed=None, options=None):
        self._taken_objects = []
        _, info = super().reset(seed=seed, options=options)
        return self.state_index, info

    def step(self, action):
        if action == 4 and self.get(self._agent_pos) is not None:
            obj_idx = self._object_positions.index(tuple(self._agent_pos))
            self._taken_objects.append(obj_idx)
        _, reward, terminated, truncated, info = super().step(action)
        return self.state_index, reward, terminated, truncated, info

    @property
    def state_index(self):
        idx = self._taken_objects_dict[tuple(self._taken_objects)] * len(self._legal_positions) + self._position2idx[tuple(self._agent_pos)]
        return self._idx_conversion[idx]

    def get_array_obs(self) -> tuple[NDArray[np.float64], int]:
        """ Return obs array as multihot vector at every cell. """
        return self._gen_obs()

    def get_state_embedder(self, embed_dim=128, hidden_dim=32, kernel_size=3, padding=0):
        width = self._width
        height = self._height

        class StateEmbedder(nn.Module):
            def __init__(self, embed_dim, hidden_dim, kernel_size, padding):
                super().__init__()

                self.embed_dim = embed_dim
                input_dim = 12
                obj_embed_dim = 16
                self._grid_encoder = nn.Sequential(
                    nn.Conv2d(in_channels=input_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Flatten(start_dim=-3, end_dim=-1),
                    nn.Linear(in_features=(width + 4 * padding - 2 * (kernel_size - 1)) * (height + 4 * padding - 2 * (kernel_size - 1)) * hidden_dim,
                              out_features=embed_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=embed_dim,
                              out_features=embed_dim),
                    nn.ReLU(),
                )
                self._object_embedder = nn.Embedding(10, obj_embed_dim)
                self._final_layer = nn.Linear(embed_dim + obj_embed_dim, embed_dim)

            @flexible_batch_dims((3, 0))
            def forward(self, obs):
                grids, objs = obs
                grids = grids.swapaxes(-1, -2).swapaxes(-2, -3)
                grid_embed = self._grid_encoder(grids)
                obj_embed = self._object_embedder(objs)
                return self._final_layer(torch.cat((grid_embed, obj_embed), -1))

        return StateEmbedder(embed_dim, hidden_dim, kernel_size, padding)

    @property
    def terminal_states(self):
        return np.array([self.goal_state_index], dtype=np.int64)

    def _state_info_from_index(self, index):
        taken_objects_idx = index // len(self._legal_positions)
        position_idx = index % len(self._legal_positions)

        taken_objects = self._taken_objects_list[taken_objects_idx]
        for visit_index in range(len(self._items_to_visit) + 1):
            if visit_index >= len(self._items_to_visit) or visit_index >= len(taken_objects):
                break
            if self._items_to_visit[visit_index] != self._object_types[taken_objects[visit_index]]:
                break
        grid = copy.deepcopy(self._grid)
        for obj, obj_position in enumerate(self._object_positions):
            if obj in taken_objects:
                grid[obj_position[0]][obj_position[1]] = None
            else:
                grid[obj_position[0]][obj_position[1]] = self._initial_objects[obj]
        return grid, np.array(self._legal_positions[position_idx]), visit_index, list(taken_objects)

    @classmethod
    def _get_transition_table(cls, env):
        """ Get deterministic transition table T(s, a) """
        if env.action_space.n == env.unwrapped.action_space.n:
            # `env` is base environment (without abstractions)
            transitions_path = os.path.join(CACHE_DIR, f"compile_{env.unwrapped._config_string}_transitions.npy")
            if os.path.isfile(transitions_path):
                return np.load(transitions_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            env = copy.deepcopy(env.unwrapped)
            env.reset()
            transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
            print("Computing base environment transition table...")
            transition_table[env.goal_state_index] = 0
            for state in tqdm(range(env.observation_space.n)):
                if state != env.goal_state_index:
                    for action in range(env.action_space.n):
                        index = env._rev_idx_conversion[state]
                        (env._grid,
                         env._agent_pos,
                         env._visited_index,
                         env._taken_objects) = env._state_info_from_index(index)
                        env.step(action)
                        transition_table[state, action] = env.state_index
            np.save(transitions_path, transition_table)
            return transition_table
        # `env` is environment with abstractions
        transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
        transition_table[:,:env.unwrapped.action_space.n] = env.unwrapped.transition_table
        for state in range(env.observation_space.n):
            for ab, action in zip(env.abstractions, range(env.unwrapped.action_space.n, env.action_space.n)):
                next_state = state
                for base_action in ab:
                    next_state = transition_table[next_state, base_action.name]
                transition_table[state, action] = next_state
        return transition_table

    @classmethod
    def _get_rewards_table(cls, env):
        """ Get deterministic rewards table R(s, a) """
        return (env.transition_table == env.unwrapped.goal_state_index).astype(float)

    def _get_state_dist(self):
        state_dist = np.zeros(self.observation_space.n)
        taken_objects_prefix = self._taken_objects_dict[()] * len(self._legal_positions)
        for position_idx in range(len(self._legal_positions)):
            idx = self._idx_conversion[taken_objects_prefix + position_idx]
            state_dist[idx] = 1
        state_dist /= np.sum(state_dist)
        return state_dist

    def _get_state_dist_entropy(self):
        masked_state_dist = self.state_dist[self.state_dist != 0]
        return -np.sum(masked_state_dist * np.log(masked_state_dist)).item()

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, iterate=True, rtol: float=1e-3):
        """ `rtol` only applies to `iterate` being True (i.e., iteratively solve for q) """
        prob_table = get_prob_table(env.transition_table,
                                    env.unwrapped.goal_state_index,
                                    eps,
                                    iterate=iterate,
                                    rtol=rtol,
                                    use_torch=True)
        state_dist = torch.tensor(env.state_dist)
        assert torch.all((state_dist == 0) | (prob_table > 0)), "Nonzero importance weights in unsolvable states!"
        prob_table[prob_table == 0] = 1
        transform = lambda x: -torch.log(x) if take_log else 1 / x
        return torch.sum(state_dist * transform(prob_table)).item()


class RLDiffNPuzzleEnv(RLDiffEnv, NPuzzleEnv):
    def __init__(self, size: int):
        super(RLDiffEnv, self).__init__(size)

        self._config_string = f"N{size ** 2 - 1}"
        self.goal_state_index = self._grid_to_index(self._solved_grid)

        super().__init__()

    @property
    def observation_space(self):
        return gym.spaces.Discrete(factorial(self._size ** 2))

    @property
    def config_string(self):
        return self._config_string

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        while self._grid == self._solved_grid:
            _, info = super().reset(options=options)
        return self.state_index, info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self.state_index, reward, terminated, truncated, info

    @property
    def state_index(self):
        return self._grid_to_index(self._grid)

    @classmethod
    def _grid_to_index(cls, grid):
        total_index = 0
        for i in range(len(grid) - 1):
            index = grid[i] - sum(grid[j] < grid[i] for j in range(i))
            total_index = total_index * (len(grid) - i) + index
        assert cls._grid_and_pos_from_index(int(np.sqrt(len(grid))), total_index)[0] == grid
        return total_index

    def get_array_obs(self) -> NDArray[np.int64]:
        """ Return obs array as tile index at every cell. """
        return np.array(self._grid, dtype=np.int64).reshape(self._size, self._size)

    def get_state_embedder(self, embed_dim=32, hidden_dim=32, kernel_size=3, padding=1):
        size = self._size

        class StateEmbedder(nn.Module):
            def __init__(self, embed_dim, hidden_dim, kernel_size, padding):
                super().__init__()

                self.embed_dim = embed_dim
                self._tile_embedding = nn.Embedding(size ** 2, hidden_dim)
                self._grid_encoder = nn.Sequential(
                    nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ReLU(),
                    nn.Flatten(start_dim=-3, end_dim=-1),
                    nn.Linear(in_features=(size + 4 * padding - 2 * (kernel_size - 1)) * (size + 4 * padding - 2 * (kernel_size - 1)) * hidden_dim,
                              out_features=embed_dim),
                    nn.ReLU(),
                    nn.Linear(in_features=embed_dim,
                              out_features=embed_dim),
                )

            @flexible_batch_dims(2)
            def forward(self, obs):
                embeddings = self._tile_embedding(obs)
                return self._grid_encoder(embeddings.swapaxes(-1, -3))

        return StateEmbedder(embed_dim, hidden_dim, kernel_size, padding)

    @property
    def terminal_states(self):
        return np.array([self.goal_state_index], dtype=np.int64)

    @classmethod
    def _grid_and_pos_from_index(cls, size, total_index):
        indices = [0]
        for i in range(2, size ** 2 + 1):
            total_index, index = total_index // i, total_index % i
            indices.append(index)
        assert total_index == 0, "There is a bug in this function"
        # sign = sum(indices) % 2
        grid = []
        grid_set = set()
        for index in reversed(indices):
            tile = 0
            while tile in grid_set:
                tile += 1
            for _ in range(index):
                tile += 1
                while tile in grid_set:
                    tile += 1
            grid.append(tile)
            grid_set.add(tile)
        # inversions = 0
        # for i in range(1, len(grid)):
        #     for j in range(i):
        #         if grid[i] < grid[j]:
        #             inversions += 1
        # if inversions % 2 == 1:
        #     grid[-2], grid[-1] = grid[-1], grid[-2]
        # pos = grid.index(0)
        # pos_parity = ((pos // size) + (pos % size)) % 2
        # goal_state_sign = not (size % 2)
        # if (sign != goal_state_sign) != pos_parity:
        #     grid[-2], grid[-1] = grid[-1], grid[-2]
        #     pos = grid.index(0)
        pos = grid.index(0)
        return grid, pos

    @classmethod
    def _get_transition_table(cls, env):
        """ Get deterministic transition table T(s, a) """
        if env.action_space.n == env.unwrapped.action_space.n:
            # `env` is base environment (without abstractions)
            transitions_path = os.path.join(CACHE_DIR, f"npuzzle_{env.unwrapped._config_string}_transitions.npy")
            if os.path.isfile(transitions_path):
                return np.load(transitions_path)
            os.makedirs(CACHE_DIR, exist_ok=True)
            env = copy.deepcopy(env.unwrapped)
            env.reset()
            transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
            print("Computing base environment transition table...")
            transition_table[env.goal_state_index] = 0
            for state in tqdm(range(env.observation_space.n)):
                for action in range(env.action_space.n):
                    env._grid, env._pos = env._grid_and_pos_from_index(env._size, state)
                    env.step(action)
                    transition_table[state, action] = env.state_index
            np.save(transitions_path, transition_table)
            return transition_table
        # `env` is environment with abstractions
        transition_table = np.empty((env.observation_space.n, env.action_space.n), dtype=np.int64)
        transition_table[:,:env.unwrapped.action_space.n] = env.unwrapped.transition_table
        for state in range(env.observation_space.n):
            for ab, action in zip(env.abstractions, range(env.unwrapped.action_space.n, env.action_space.n)):
                next_state = state
                for base_action in ab:
                    next_state = transition_table[next_state, base_action.name]
                transition_table[state, action] = next_state
        return transition_table

    @classmethod
    def _get_rewards_table(cls, env):
        """ Get deterministic rewards table R(s, a) """
        return (env.transition_table == env.unwrapped.goal_state_index).astype(float)

    def _get_state_dist(self):
        max_scramble_len = self.gods_numbers[self._size]
        prob_table = np.zeros((max_scramble_len + 1, self.observation_space.n))
        prob_table[0, self.goal_state_index] = 1
        actions_that_move = self.transition_table != np.arange(self.observation_space.n).reshape(-1, 1)
        branching_factors = np.sum(actions_that_move, axis=1)
        prev_state_branching_factors = branching_factors[self.transition_table]
        for scramble_len in range(1, max_scramble_len + 1):
            # for old_state in range(self.observation_space.n):
            #     new_states = self.transition_table[old_state]
            #     new_states = new_states[new_states != old_state]  # don't take actions that don't change state during scramble
            #     prob_table[scramble_len, new_states] += prob_table[scramble_len - 1, old_state] / len(new_states)
            prev_state_probs = prob_table[scramble_len - 1][self.transition_table] * actions_that_move
            prob_table[scramble_len] = np.sum(prev_state_probs / prev_state_branching_factors, axis=1)
        state_dist = np.sum(prob_table, axis=0)
        state_dist[self.goal_state_index] -= 1
        state_dist /= max_scramble_len
        assert abs(np.sum(state_dist) - 1) < 1e-7
        state_dist[self.goal_state_index] = 0
        state_dist /= np.sum(state_dist)
        return state_dist

    def _get_state_dist_entropy(self):
        masked_state_dist = self.state_dist[self.state_dist != 0]
        return -np.sum(masked_state_dist * np.log(masked_state_dist)).item()

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, iterate=True, rtol: float=1e-3):
        """ `rtol` only applies to `iterate` being True (i.e., iteratively solve for q) """
        prob_table = get_prob_table(env.transition_table,
                                    env.unwrapped.goal_state_index,
                                    eps,
                                    iterate=iterate,
                                    rtol=rtol,
                                    use_torch=True)
        state_dist = torch.tensor(env.state_dist)
        assert torch.all((state_dist == 0) | (prob_table > 0)), "Nonzero importance weights in unsolvable states!"
        prob_table[prob_table == 0] = 1
        transform = lambda x: -torch.log(x) if take_log else 1 / x
        return torch.sum(state_dist * transform(prob_table)).item()


T = TypeVar('T')
def bfs(
        start_state: Union[np.int64, int],
        goal_state: Union[np.int64, int],
        transition_table: Union[NDArray[np.int64], list[list[list[int]]]],  # (S, A) or S x A -> P(S)
        start_value: T,
        value_update: Callable[[T, Union[np.int64, int], int, Union[np.int64, int]], T],
        default_value: Optional[T]
    ) -> Optional[T]:
    num_actions = len(transition_table[0])
    to_visit: Queue[tuple[Union[np.int64, int], T]] = Queue()
    to_visit.put((start_state, start_value))
    visited_states = {start_state}
    while True:
        if to_visit.empty():
            return default_value
        state, value = to_visit.get()
        for action in range(num_actions):
            if isinstance(transition_table, np.ndarray):
                next_states = [transition_table[state, action]]
            else:
                next_states = transition_table[state][action]
            for next_state in next_states:
                if next_state not in visited_states:
                    next_value = value_update(value, state, action, next_state)
                    if next_state == goal_state:
                        return next_value
                    to_visit.put((next_state, next_value))
                    visited_states.add(next_state)


def get_distances(goal_state_index, transition_table, inverted_transitions=False):
    obs_space_size = transition_table.shape[0]
    action_space_size = transition_table.shape[1]
    if inverted_transitions:
        inv_transition_table = transition_table
    else:
        inv_transition_table = [[[] for _ in range(action_space_size)] for _ in range(obs_space_size)]
        for state in range(obs_space_size):
            for action in range(action_space_size):
                inv_transition_table[transition_table[state, action]][action].append(state)
    distances = np.empty(obs_space_size)
    distances[:] = np.inf
    distances[goal_state_index] = 0
    def update_dists(dist, state, action, next_state):
        distances[next_state] = dist + 1
        return dist + 1
    bfs(goal_state_index,
        -1,
        inv_transition_table,
        start_value=0,
        value_update=update_dists,
        default_value=None)
    return distances


def get_prob_table(
        transition_table: NDArray[np.int64],
        goal_state_index: int,
        eps: float,
        iterate: bool,
        rtol: float,
        use_torch: bool
    ) -> Union[NDArray, torch.Tensor]:

    np_or_torch = torch if use_torch else np
    obs_space_size, action_space_size = transition_table.shape

    if use_torch:
        transition_table = torch.tensor(transition_table)

    if iterate:

        def iterate_prob_table(prob_table):
            new_prob_table = (1 - eps) * np_or_torch.mean(prob_table[transition_table], axis=1)
            new_prob_table[goal_state_index] = 1
            return new_prob_table

        prob_table = np_or_torch.zeros(obs_space_size)
        prob_table[goal_state_index] = 1

        while True:
            new_prob_table = iterate_prob_table(prob_table)
            errors = np_or_torch.where((new_prob_table == 0) & (prob_table == 0), 0,
                                       np_or_torch.log(new_prob_table / prob_table))
            err = np_or_torch.max(errors)
            if err < rtol:
                break
            prob_table = new_prob_table

        return new_prob_table

    # SPARSE SOLVE WITH NUMPY SPARSE ARRAYS
    # data = np.ones(transition_table.size, dtype=dtype) * (1 - eps) / action_space_size
    # indices = transition_table.reshape(-1)
    # indptr = np.arange(0, transition_table.size + 1, action_space_size)
    # A = sparse.csr_array((data, indices, indptr), shape=(obs_space_size, obs_space_size))
    # A -= sparse.identity(obs_space_size, dtype=dtype, format="csr")
    # A[[goal_state_index]] = sparse.csr_array((np.ones((1,), dtype=dtype), [goal_state_index], [0, 1]), shape=(1, obs_space_size))
    # b = np.zeros((obs_space_size,), dtype=dtype)
    # b[goal_state_index] = 1
    # res = sparse.linalg.spsolve(A, b)

    A = np_or_torch.zeros((obs_space_size, obs_space_size))
    for row_A, row_transition_table in zip(A, transition_table):
        row_A[row_transition_table] = (1 - eps) / action_space_size
    A -= np_or_torch.eye(obs_space_size)
    A[goal_state_index] = 0
    A[goal_state_index, goal_state_index] = 1

    b = np_or_torch.zeros(obs_space_size)
    b[goal_state_index] = 1

    return np_or_torch.linalg.solve(A, b)


gym.register(
    id="CliffWalking-v0-rldiff",
    entry_point="rldiff_envs:RLDiffCliffWalkingEnv",
)

gym.register(
    id="RubiksCube222-v0-rldiff",
    entry_point="rldiff_envs:RLDiffRubiksCube222Env",
)

gym.register(
    id="CompILE-v0-rldiff",
    entry_point="rldiff_envs:RLDiffCompILEEnv",
)

gym.register(
    id="NPuzzle-v0-rldiff",
    entry_point="rldiff_envs:RLDiffNPuzzleEnv",
)
