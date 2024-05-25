from typing import Optional
import tqdm

import numpy as np
from numpy.typing import NDArray
import torch

import gymnasium as gym
from gymnasium import spaces

from abstractions.abstractions import Abstraction

from rldiff_envs import RLDiffEnv


class RLDiffEnvAdapter(RLDiffEnv, gym.Wrapper):
    def __init__(self, env: gym.Env):
        """ `env.unwrapped` must be instance of RLDiffEnv but `env` need not be """
        super(RLDiffEnv, self).__init__(env)
        super().__init__()

        self._cache = env.unwrapped._cache

    @property
    def config_string(self) -> Optional[str]:
        return self.unwrapped.config_string

    @property
    def state_index(self) -> int:
        return self.unwrapped.state_index

    def get_array_obs(self) -> NDArray:
        return self.unwrapped.get_array_obs()

    def get_state_embedder(self) -> torch.nn.Module:
        return self.unwrapped.get_state_embedder()

    @property
    def terminal_states(self) -> NDArray[np.uint]:
        return self.unwrapped.terminal_states

    @classmethod
    def _get_transition_table(cls, env) -> NDArray[np.uint]:
        return env.unwrapped.transition_table

    @classmethod
    def _get_rewards_table(cls, env) -> NDArray:
        return env.unwrapped.rewards_table

    @classmethod
    def _get_relevant_inds(cls, env) -> Optional[tuple[NDArray[np.uint], NDArray[np.uint]]]:
        return env.unwrapped.relevant_inds

    def _get_state_dist(self) -> NDArray:
        return self.unwrapped.state_dist

    def _get_state_dist_entropy(self) -> float:
        return self.unwrapped.state_dist_entropy

    @classmethod
    def _get_avg_sol_len(cls, env) -> float:
        return env.unwrapped.avg_sol_len
    
    def get_incompress(self) -> float:
        return self.unwrapped.get_incompress()

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, **kwargs) -> float:
        return env.unwrapped.get_expldiff(eps, take_log, **kwargs)


class RLDiffEnvWrapper(RLDiffEnv, gym.Wrapper):
    def __init__(self, env: RLDiffEnv):
        super(RLDiffEnv, self).__init__(env)
        super().__init__()
        self._cache = env._cache

    @property
    def config_string(self) -> Optional[str]:
        return self.env.config_string

    @property
    def state_index(self) -> int:
        return self.env.state_index

    def get_array_obs(self) -> NDArray:
        return self.env.get_array_obs()

    def get_state_embedder(self) -> torch.nn.Module:
        return self.env.get_state_embedder()

    @property
    def terminal_states(self) -> NDArray[np.uint]:
        return self.env.terminal_states

    @classmethod
    def _get_transition_table(cls, env) -> NDArray[np.uint]:
        return env.env.transition_table

    @classmethod
    def _get_rewards_table(cls, env) -> NDArray:
        return env.env.rewards_table

    @classmethod
    def _get_relevant_inds(cls, env) -> Optional[tuple[NDArray[np.uint], NDArray[np.uint]]]:
        return env.env.relevant_inds

    def _get_state_dist(self) -> NDArray:
        return self.env.state_dist

    def _get_state_dist_entropy(self) -> float:
        return self.env.state_dist_entropy

    @classmethod
    def _get_avg_sol_len(cls, env) -> float:
        return env.env.avg_sol_len
    
    def get_incompress(self) -> float:
        return self.env.get_incompress()

    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, **kwargs) -> float:
        return env.env.get_expldiff(eps, take_log, **kwargs)


class RLDiffEnvAbsWrapper(RLDiffEnvWrapper):
    """Augments the actions of the environment with macroactions.

    The wrapped environment must have a discrete action space.
    """

    def __init__(self, env, abstractions: list[Abstraction],
                       truncate_steps: Optional[int]=None, truncate_base_steps: Optional[int]=None):
        super().__init__(env)

        self.abstractions = abstractions
        # Action space is default low-level actions + options
        self.action_space = spaces.Discrete(
                env.action_space.n + len(self.abstractions))
        self.num_base_actions = env.action_space.n.item() + sum(len(ab) for ab in abstractions)

        self.truncate_steps = truncate_steps
        self.truncate_base_steps = truncate_base_steps

        self._cache = {}

    def step(self, action):
        # Default low-level actions
        cur_base_steps = 0
        if action < self.env.action_space.n:
            next_state, cum_reward, terminated, truncated, info = self.env.step(action)
            cur_base_steps += 1
        else:
            ab = self.abstractions[action-self.env.action_space.n]
            cum_reward = 0.
            for ax in ab:
                next_state, reward, terminated, truncated, info = self.env.step(ax.name)
                cum_reward += reward
                cur_base_steps += 1
                if terminated:
                    break
        self.num_steps += 1
        self.num_base_steps += cur_base_steps
        truncated = (truncated
                or (self.truncate_steps is not None and self.num_steps >= self.truncate_steps)
                or (self.truncate_base_steps is not None and self.num_base_steps >= self.truncate_base_steps))
        info["steps"] = cur_base_steps
        return next_state, cum_reward, terminated, truncated, info
            
    def reset(self, **kwargs):
        self.num_steps = 0
        self.num_base_steps = 0
        return self.env.reset(**kwargs)

    @classmethod
    def _get_transition_table(cls, env) -> NDArray[np.uint]:
        return env.unwrapped._get_transition_table(env)

    @classmethod
    def _get_rewards_table(cls, env) -> NDArray:
        return env.unwrapped._get_rewards_table(env)

    @classmethod
    def _get_relevant_inds(cls, env) -> Optional[tuple[NDArray[np.uint], NDArray[np.uint]]]:
        return env.unwrapped._get_relevant_inds(env)

    @classmethod
    def _get_avg_sol_len(cls, env) -> float:
        return env.unwrapped._get_avg_sol_len(env)
    
    @classmethod
    def _get_expldiff(cls, env, eps: float, take_log: bool=True, **kwargs) -> float:
        return env.unwrapped._get_expldiff(env, eps, take_log, **kwargs)


# Adapted from yidingjiang/love/option_wrapper.py
class RLDiffEnvLoveWrapper(RLDiffEnvAbsWrapper):
    """Augments the actions of the environment with options from the HSSM.

    The wrapped environment must have a discrete action space.
    """

    def __init__(self, env, hssm, train_loader, init_size,
                 threshold=0.05, recurrent=False,
                 truncate_steps: Optional[int]=None, truncate_base_steps: Optional[int]=None):

        super(RLDiffEnvAbsWrapper, self).__init__(env)

        self._hssm = hssm

        # Compute z's on which HSSM has support over threshold:
        # i.e., {z | p(z) > threshold}
        num_options = hssm.post_abs_state.latent_n
        # p(z): of shape (num_options,)
        marginal = np.zeros(num_options)

        for train_obs_list, train_action_list, train_mask_list in tqdm.tqdm(train_loader):
            # Mean of the marginals in the batch, where each batch is
            # weighted by batch_size / total dataset size
            train_obs_list_part = train_obs_list[0] if isinstance(train_obs_list, tuple) else train_obs_list
            weight = train_obs_list_part.shape[0] / len(train_loader.dataset)
            marginal += hssm.abs_marginal(
                    train_obs_list, train_action_list, train_mask_list,
                    init_size=init_size)[0].cpu().data.numpy() * weight

            del train_obs_list
            del train_action_list
            del train_mask_list

        self._permitted_zs = [z for z in range(num_options)
                              if marginal[z] >= threshold]

        # Action space is default low-level actions + options
        self.action_space = spaces.Discrete(
                env.action_space.n + len(self._permitted_zs))

        self._current_state = None
        self._boundary_state = None
        self._recurrent = recurrent

        self.truncate_steps = truncate_steps
        self.truncate_base_steps = truncate_base_steps

    def step(self, action):
        # Default low-level actions
        if action < self.env.action_space.n:
            state, reward, terminated, truncated, info = self.env.step(action)
            self._current_state = state
            self.num_steps += 1
            self.num_base_steps += 1
            truncated = (truncated
                    or (self.truncate_steps is not None and self.num_steps >= self.truncate_steps)
                    or (self.truncate_base_steps is not None and self.num_base_steps >= self.truncate_base_steps))
            info["steps"] = 1
            return state, reward, terminated, truncated, info

        # Taking an option as an action
        # Follows the option until the option terminates
        z = self._permitted_zs[action - self.env.action_space.n]
        state = self._current_state
        total_reward = 0  # accumulate all rewards during option
        low_level_actions = []
        self._boundary_state = self._hssm.initial_boundary_state(state)
        hidden_state = None
        while True:
            action, next_hidden_state = self._hssm.play_z(
                    z, state, hidden_state,
                    recurrent=self._recurrent)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            low_level_actions.append(action)
            self.num_base_steps += 1
            total_reward += reward
            terminate_skill, self._boundary_state = self._hssm.z_terminates(
                    next_state, action, boundary_state=self._boundary_state)
            state = next_state
            hidden_state = next_hidden_state
            truncated = (truncated
                    or (self.truncate_base_steps is not None and self.num_base_steps >= self.truncate_base_steps))
            if terminated or truncated or terminate_skill:
                break

        self._current_state = state
        truncated = (truncated
                or (self.truncate_steps is not None and self.num_steps >= self.truncate_steps))
        self.num_steps += 1
        info["low_level_actions"] = low_level_actions
        info["steps"] = len(low_level_actions)
        return state, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.num_steps = 0
        self.num_base_steps = 0
        self._current_state, info = self.env.reset(**kwargs)
        return self._current_state, info


class RLDiffEnvArrayObsWrapper(RLDiffEnvWrapper, gym.ObservationWrapper):
    def observation(self, obs):
        return self.env.get_array_obs()


class RLDiffEnvTensorWrapper(RLDiffEnvWrapper, gym.ObservationWrapper):
    def observation(self, obs):
        if isinstance(obs, tuple):
            return tuple(map(torch.tensor, obs))
        return torch.tensor(obs)

    @property
    def terminal_states(self) -> torch.LongTensor:
        return torch.tensor(self.env.terminal_states)

    @classmethod
    def _get_transition_table(cls, env) -> torch.LongTensor:
        return torch.tensor(env.env.transition_table)

    @classmethod
    def _get_rewards_table(cls, env) -> torch.Tensor:
        return torch.tensor(env.env.rewards_table)

    def _get_state_dist(self) -> torch.Tensor:
        return torch.tensor(self.env.state_dist)
