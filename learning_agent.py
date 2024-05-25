from __future__ import annotations

from argparse import Namespace
import os
from abc import ABC, abstractmethod
from typing import Optional, Union
from numbers import Number
import json
import pickle as pkl
import warnings

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

from gymnasium import spaces

from utils import Episode, ReplayBuffer, RewardTarget, MovingAverage, batch_obs, quantile_sorted, diff_args
from wrappers import RLDiffEnvTensorWrapper


class Agent(ABC):
    @abstractmethod
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper) -> None:
        self.args = args
        self.env = env

    @abstractmethod
    def get_action(self, obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]], greedy=False) -> int:
        """
        Returns the action of the greedy policy (if `greedy`) or the exploration policy (if not `greedy`).
        """
        pass

    @abstractmethod
    def setup_training(self, run_id):
        """ Set up training. """
        pass

    @abstractmethod
    def on_test_end(
        self,
        test_episodes: list[Episode],
        test_mean_reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> tuple[dict[str, Optional[float]], Optional[bool]]:
        """
        Updates agent state (and `progress_bar_dict` if applicable)
        at the end of testing.
        Returns (extra_log_dict, extra_early_stop)
        """
        pass

    @abstractmethod
    def on_training_end(self, run_id) -> None:
        """
        Does anything required at the end of the entire training loop,
        such as saving the model weights.
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict) -> None:
        """
        Load agent state from `state_dict` into `self`
        """
        if self.args != state_dict["args"]:
            warnings.warn("Loaded args differ from args of currently-running script! "
                    "Current args will override loaded args.\n"
                    "Loaded args contain these extra key-value pairs:\n"
                    + json.dumps(diff_args(self.args, state_dict["args"]), indent=2) +
                    "Current args contain these extra key-value pairs:\n"
                    + json.dumps(diff_args(state_dict["args"], self.args), indent=2)
                    )
        # Note: Loading random state of environment is done in `run_RL_train_loop`

    @abstractmethod
    def get_state_dict(self) -> dict:
        """
        Dump agent state into state dict and return it
        """
        state_dict = {"args": self.args}
        # Note: Saving random state of environment is done in `run_RL_train_loop`
        return state_dict


class LearningAgent(Agent, ABC):
    subtypes: dict[str, type[LearningAgent]] = {}

    @staticmethod
    def new(args, env):
        return LearningAgent.subtypes[args.rl_algo](args, env)

    @staticmethod
    def register(name: str):
        def register_subtype(cls: type[LearningAgent]):
            LearningAgent.subtypes[name] = cls
            return cls
        return register_subtype

    @abstractmethod
    def on_new_experience(
        self,
        obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        """ Updates agent state upon seeing a new experience """
        pass

    @abstractmethod
    def on_episode_end(
        self,
        episode: Episode,
        reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> None:
        """
        Updates agent state (and `progress_bar_dict` if applicable)
        at the end of a training episode.
        """
        pass


class PlanningAgent(ABC):
    subtypes: dict[str, type[PlanningAgent]] = {}

    @staticmethod
    def new(args, env):
        return PlanningAgent.subtypes[args.rl_algo](args, env)

    @staticmethod
    def register(name: str):
        def register_subtype(cls: type[PlanningAgent]):
            PlanningAgent.subtypes[name] = cls
            return cls
        return register_subtype

    @abstractmethod
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper) -> None:
        pass

    @abstractmethod
    def update_all(self):
        """ Updates the entire table of values given rewards. """
        pass

    @abstractmethod
    def on_iteration_end(self, progress_bar_dict: Optional[dict[str, Optional[Number]]]=None) -> None:
        """
        Updates agent state (and `progress_bar_dict` if applicable)
        at the end of a training episode.
        """
        pass


class EpsGreedyLearningAgent(LearningAgent, ABC):
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper) -> None:
        """
        Initialization for ABC for an off-policy learning agent that uses epsilon greedy exploration
        and an optional replay buffer for collected experience

        args:
            start_eps (float): initial epsilon value
            eps_decay (float): decay for epsilon
            final_eps (float): final epsilon value
            adaptive_eps_greedy (bool): whether to use adaptive epsilon greedy (epsilon decay based on)
            reward_start_target (float): initial reward target
            reward_target_step (float): increment in reward target every time it is reached during testing
            reward_final_target (float): final reward target
            use_replay_buffer (bool): whether to use a replay buffer
            replay_buffer_size (int): max number of experiences the replay buffer can store
        """
        super().__init__(args, env)
        self.epsilon: float = args.start_eps
        self.reward_target: Optional[RewardTarget] = None
        if args.adaptive_eps_greedy:
            self.reward_target = RewardTarget(args.reward_start_target, args.reward_target_step, args.reward_final_target)
        self.replay_buffer: Optional[ReplayBuffer] = ReplayBuffer(args.replay_buffer_size) if args.use_replay_buffer else None
        self.episode_idx: int = 0

    def on_new_experience(
        self,
        obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        if self.replay_buffer is not None:
            self.replay_buffer.add(obs, action, reward, terminated, next_obs)
        else:
            self.update(obs, action, reward, terminated, next_obs)

    @abstractmethod
    def update(
        self,
        obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        """ Updates agent according to single experience """
        pass

    def on_episode_end(
        self,
        episode: Episode,
        reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> None:
        # decay epsilon if applicable
        if self.reward_target is None:
            self.decay_epsilon()
            if progress_bar_dict is not None:
                progress_bar_dict["eps"] = self.epsilon
        # update the agent if applicable
        if self.replay_buffer is not None and (self.episode_idx + 1) % self.args.update_every == 0:
            obs_list, action_list, reward_list, terminated_list, next_obs_list = self.replay_buffer.sample(self.args.batch_size)
            self.update_batch(obs_list, action_list, reward_list, terminated_list, next_obs_list)
        self.episode_idx += 1

    @abstractmethod
    def update_batch(
        self,
        obses: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        actions: torch.IntTensor,
        rewards: torch.Tensor,
        terminateds: torch.BoolTensor,
        next_obses: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        """ Updates agent according to batch of experiences """
        pass

    def on_test_end(
        self,
        test_episodes: list[Episode],
        test_mean_reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> tuple[dict[str, Optional[float]], Optional[bool]]:
        # decay epsilon
        if self.reward_target is not None and test_mean_reward >= self.reward_target.get():
            self.decay_epsilon()
            if progress_bar_dict is not None:
                progress_bar_dict["eps"] = self.epsilon
            self.reward_target.update()
        return {"epsilon": self.epsilon}, None

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.args.final_eps, self.epsilon - self.args.eps_decay)

    @abstractmethod
    def load_state_dict(self, state_dict) -> dict:
        super().load_state_dict(state_dict)
        self.epsilon: float = state_dict["epsilon"]
        self.reward_target: Optional[RewardTarget] = state_dict["reward_target"]
        self.replay_buffer: Optional[ReplayBuffer] = state_dict["replay_buffer"]
        self.episode_idx: int = state_dict["episode_idx"]
        return state_dict

    @abstractmethod
    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        state_dict.update({
            "epsilon": self.epsilon,
            "reward_target": self.reward_target,
            "replay_buffer": self.replay_buffer,
            "episode_idx": self.episode_idx,
        })
        return state_dict


@LearningAgent.register("QLearning")
@PlanningAgent.register("QLearning")
class QLearningAgent(EpsGreedyLearningAgent, PlanningAgent):
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper):
        """
        Initialize a Q-learning agent

        args:
            deep (bool): whether to use a neural network for state embedding
            lr (float): learning rate
            discount_factor (float): discount factor for computing the Q-value
            td_moving_avg_alpha (float): alpha for the moving average of td error
            ... (see docstring for EpsGreedyLearningAgent)
        """
        super().__init__(args, env)

        self.q_network = None
        if args.deep:
            state_embedder = env.get_state_embedder()
            self.q_network = nn.Sequential(state_embedder,
                                           nn.Linear(state_embedder.embed_dim, env.action_space.n)
                                          )
            self.optimizer = optim.AdamW(self.q_network.parameters(), lr=args.lr)
        else:
            self.q_values = torch.zeros((env.observation_space.n, env.action_space.n))

        self.td_moving_avg = MovingAverage(args.td_moving_avg_alpha)

    def _get_q_values(self, obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]], keep_grad=False) -> torch.Tensor:
        if self.q_network is None:
            return self.q_values[obs]
        if keep_grad:
            return self.q_network(obs)
        with torch.no_grad():
            return self.q_network(obs)

    def get_action(self, obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]], greedy=False) -> int:
        """
        Returns the best action with probability (1 - epsilon) or if `greedy`
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # if `greedy` or with probability (1 - epsilon) act greedily (exploit)
        if greedy or np.random.random() >= self.epsilon:
            return self._get_q_values(obs).argmax().item()
        # if not `greedy` then with probability epsilon return a random action to explore the environment
        return self.env.action_space.sample()

    def update(
        self,
        obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        """ Updates the Q-value of an action. """
        if self.q_network is not None:
            self.optimizer.zero_grad()

        future_q_value = (not terminated) * self._get_q_values(next_obs).max()
        target = reward + self.args.discount_factor * future_q_value
        assert not target.requires_grad

        current_q_value = self._get_q_values(obs, keep_grad=True)[action]
        temporal_difference = target - current_q_value

        if self.q_network is None:
            self.q_values[obs,action] += self.args.lr * temporal_difference
        else:
            loss = temporal_difference ** 2
            loss.backward()
            self.optimizer.step()

        self.td_moving_avg.update(torch.abs(temporal_difference).item())

    def update_batch(
        self,
        obses: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        actions: torch.IntTensor,
        rewards: torch.Tensor,
        terminateds: torch.BoolTensor,
        next_obses: Union[torch.Tensor, tuple[torch.Tensor, ...]],
    ):
        """ Updates the Q-values of a batch of actions. """
        if self.q_network is not None:
            self.optimizer.zero_grad()

        future_q_values = self._get_q_values(next_obses).max(axis=1).values
        future_q_values = (~terminateds) * future_q_values
        targets = rewards + self.args.discount_factor * future_q_values
        assert not targets.requires_grad

        current_q_values = self._get_q_values(obses, keep_grad=True)[range(len(actions)), actions]
        temporal_differences = targets - current_q_values

        if self.q_network is None:
            self.q_values[obses,actions] += self.args.lr * temporal_differences
        else:
            loss = (temporal_differences ** 2).mean()
            loss.backward()
            self.optimizer.step()

        self.td_moving_avg.update(torch.sqrt((temporal_differences ** 2).mean()).item())

    def update_all(self):
        """ Updates the entire Q-table given rewards. """
        assert self.q_network is None, "Cannot update entire Q-table with DQN"

        future_q_values = self.q_values.max(axis=1).values
        future_q_values[self.env.terminal_states] = 0
        future_q_values = future_q_values[self.env.transition_table]
        targets = self.env.rewards_table + self.args.discount_factor * future_q_values
        temporal_differences = targets - self.q_values

        self.q_values += self.args.lr * temporal_differences

        self.td_moving_avg.update(torch.sqrt((temporal_differences ** 2).mean()).item())

    def decay_epsilon(self):
        self.epsilon = max(self.args.final_eps, self.epsilon - self.args.eps_decay)

    def setup_training(self, run_id):
        if not self.args.deep:
            self.true_q_values = None
            if self.args.early_stop_model_err is not None:
                assert self.args.true_model_path is not None, "Must specify path to true Q-values if using Q-values to early-stop"
            if self.args.true_model_path is not None:
                self.true_q_values = torch.tensor(np.load(os.path.join(self.args.true_model_path, f"run{run_id}_Q_values.npy")))

    def on_episode_end(
        self,
        episode: Episode,
        reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> None:
        super().on_episode_end(episode, reward, progress_bar_dict)
        self.on_iteration_end(progress_bar_dict)

    def on_iteration_end(self, progress_bar_dict: Optional[dict[str, Optional[Number]]]=None) -> None:
        if progress_bar_dict is not None and self.td_moving_avg.get() is not None:
            progress_bar_dict["td"] = self.td_moving_avg.get()

    def on_test_end(
        self,
        test_episodes: list[Episode],
        test_mean_reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> tuple[dict[str, Optional[float]], Optional[bool]]:
        log_dict, early_stop = super().on_test_end(test_episodes, test_mean_reward, progress_bar_dict)
        log_dict.update({
            "td_moving_avg": self.td_moving_avg.get()
        })
        if not self.args.deep and self.true_q_values is not None:
            if self.args.mask_model_errs:
                visited_s_a_pairs = {s_a for test_episode in test_episodes for s_a in test_episode}
                q_comparison_mask = tuple(torch.tensor(inds) for inds in zip(*visited_s_a_pairs))
                masked_weights = torch.ones(len(q_comparison_mask[0]))
            else:
                q_comparison_mask = torch.all(
                        torch.arange(self.env.observation_space.n, dtype=torch.int64).reshape(-1, 1)
                        != self.env.terminal_states.reshape(1, -1),
                        axis=1)
                masked_weights = self.env.state_dist[q_comparison_mask].unsqueeze(1).expand(-1, self.q_values.shape[1]).flatten()
            masked_weights /= torch.sum(masked_weights)
            masked_q_values = self.q_values[q_comparison_mask].flatten()
            masked_true_q_values = self.true_q_values[q_comparison_mask].flatten()
            q_values_rel_err = torch.abs(torch.log(masked_q_values / masked_true_q_values))
            q_values_rel_err[torch.isnan(q_values_rel_err)] = 0
            q_values_rel_err, rel_err_sort_inds = q_values_rel_err.sort()
            q_values_abs_err = torch.abs(masked_q_values - masked_true_q_values)
            q_values_abs_err, abs_err_sort_inds = q_values_abs_err.sort()
            log_dict.update({
                "q_values_wmean_rel_err": torch.sum(q_values_rel_err * masked_weights[rel_err_sort_inds]).item(),
                "q_values_max_rel_err": torch.max(q_values_rel_err).item(),
                "q_values_99prctl_rel_err": quantile_sorted(q_values_rel_err, 0.99).item(),
                "q_values_95prctl_rel_err": quantile_sorted(q_values_rel_err, 0.95).item(),
                "q_values_90prctl_rel_err": quantile_sorted(q_values_rel_err, 0.90).item(),
                "q_values_75prctl_rel_err": quantile_sorted(q_values_rel_err, 0.75).item(),
                "q_values_median_rel_err": quantile_sorted(q_values_rel_err, 0.50).item(),
                "q_values_rms_rel_err": torch.sqrt(torch.mean(q_values_rel_err ** 2)).item(),
                "q_values_wmean_abs_err": torch.sum(q_values_abs_err * masked_weights[abs_err_sort_inds]).item(),
                "q_values_max_abs_err": torch.max(q_values_abs_err).item(),
                "q_values_99prctl_abs_err": quantile_sorted(q_values_abs_err, 0.99).item(),
                "q_values_95prctl_abs_err": quantile_sorted(q_values_abs_err, 0.95).item(),
                "q_values_90prctl_abs_err": quantile_sorted(q_values_abs_err, 0.90).item(),
                "q_values_75prctl_abs_err": quantile_sorted(q_values_abs_err, 0.75).item(),
                "q_values_median_abs_err": quantile_sorted(q_values_abs_err, 0.50).item(),
                "q_values_rms_abs_err": torch.sqrt(torch.mean(q_values_abs_err ** 2)).item(),
            })
            if progress_bar_dict is not None and self.args.model_err_metric is not None:
                progress_bar_dict[self.args.model_err_metric] = log_dict[f"q_values_{self.args.model_err_metric}"]
        if (early_stop is None or early_stop) and (self.args.early_stop_td is not None or self.args.early_stop_model_err is not None):
            td_passed = (self.args.early_stop_td is None
                    or log_dict["td_moving_avg"] is not None and log_dict["td_moving_avg"] <= self.args.early_stop_td)
            q_err = log_dict[f"q_values_{self.args.model_err_metric}"]
            q_err_passed = (self.args.early_stop_model_err is None
                    or q_err <= self.args.early_stop_model_err)
            early_stop = td_passed and q_err_passed
        return log_dict, early_stop

    def on_training_end(self, run_id):
        # save final Q-values if applicable
        if not self.args.deep:
            np.save(os.path.join(self.args.expt_dir, f"run{run_id}_Q_values.npy"), self.q_values.to("cpu").numpy())

    def load_state_dict(self, state_dict) -> dict:
        super().load_state_dict(state_dict)
        if self.args.deep:
            self.q_network.load_state_dict(state_dict["q_network"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            self.q_values = torch.tensor(state_dict["q_values"])
        self.td_moving_avg = state_dict["td_moving_avg"]
        return state_dict

    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        if self.args.deep:
            state_dict["q_network"] = self.q_network.state_dict()
            state_dict["optimizer"] = self.optimizer.state_dict()
        else:
            state_dict["q_values"] = self.q_values.cpu().numpy()
        state_dict["td_moving_avg"] = self.td_moving_avg
        return state_dict


@LearningAgent.register("ValueIteration")
@PlanningAgent.register("ValueIteration")
class ValueIterationAgent(EpsGreedyLearningAgent):
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state values and a learning rate.

        args:
            deep (bool): whether to use a neural network for state embedding
            lr (float): learning rate
            discount_factor (float): discount factor for computing the state value
            td_moving_avg_alpha (float): alpha for the moving average of td error
            ... (see docstring for EpsGreedyLearningAgent)
        """
        super().__init__(args, env)

        assert not args.deep, "Deep RL not supported for value iteration!"
        self.values = torch.zeros(self.env.observation_space.n)
        self.td_moving_avg = MovingAverage(args.td_moving_avg_alpha)

    def _get_value(self, obs: Union[int, torch.LongTensor]) -> torch.Tensor:
        return self.values[obs]

    def get_action(self, obs: int, greedy=False) -> int:
        """
        Returns the best action with probability (1 - epsilon) or if `greedy`
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # if `greedy` or with probability (1 - epsilon) act greedily (exploit)
        if greedy or np.random.random() >= self.epsilon:
            q_values = self.env.rewards_table[obs] + self.args.discount_factor * self._get_value(self.env.transition_table[obs])
            return q_values.argmax().item()
        # if not `greedy` then with probability epsilon return a random action to explore the environment
        return self.env.action_space.sample()

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ) -> None:
        """ Updates the value of a state. """

        next_states = self.env.transition_table[obs]
        not_terminal = torch.all(next_states.reshape(-1, 1) != self.env.terminal_states.reshape(1, -1), axis=1)
        future_values = not_terminal * self._get_value(next_states)
        target = (self.env.rewards_table[obs] + self.args.discount_factor * future_values).max()

        current_value = self._get_value(obs)
        temporal_difference = target - current_value

        self.values[obs] += self.args.lr * temporal_difference

        self.td_moving_avg.update(torch.abs(temporal_difference).item())

    def update_batch(
        self,
        obses: torch.LongTensor,
        actions: torch.IntTensor,
        rewards: torch.Tensor,
        terminateds: torch.BoolTensor,
        next_obses: torch.LongTensor,
    ):
        """ Updates the Q-values of a batch of actions. """

        next_states = self.env.transition_table[obses]
        not_terminal = torch.all(next_states[..., None] != self.env.terminal_states.reshape(1, 1, -1), axis=2)
        future_values = not_terminal * self._get_value(next_states)
        targets = (self.env.rewards_table[obses] + self.args.discount_factor * future_values).max(axis=1).values

        current_values = self._get_value(obses)
        temporal_differences = targets - current_values

        self.values[obses] += self.args.lr * temporal_differences

        self.td_moving_avg.update(torch.sqrt((temporal_differences ** 2).mean()).item())

    def update_all(self):
        """ Updates the entire value table given rewards. """

        future_values = self.values[self.env.transition_table]
        future_values[self.env.transition_table == self.env.terminal_states] = 0
        targets = (self.env.rewards_table + self.args.discount_factor * future_values).max(axis=1).values
        temporal_differences = targets - self.values

        self.values += self.args.lr * temporal_differences

        self.td_moving_avg.update(torch.sqrt((temporal_differences ** 2).mean()).item())

    def setup_training(self, run_id):
        if not self.args.deep:
            self.true_values = None
            if self.args.early_stop_model_err is not None:
                assert self.args.true_model_path is not None, "Must specify path to true values if using values to early-stop"
            if self.args.true_model_path is not None:
                self.true_values = torch.tensor(np.load(os.path.join(self.args.true_model_path, f"run{run_id}_V_values.npy")))

    def on_episode_end(
        self,
        episode: Episode,
        reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> None:
        super().on_episode_end(episode, reward, progress_bar_dict)
        self.on_iteration_end(progress_bar_dict)

    def on_iteration_end(self, progress_bar_dict: Optional[dict[str, Optional[Number]]]=None) -> None:
        if progress_bar_dict is not None and self.td_moving_avg.get() is not None:
            progress_bar_dict["td"] = self.td_moving_avg.get()

    def on_test_end(
        self,
        test_episodes: list[Episode],
        test_mean_reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> tuple[dict[str, Optional[float]], Optional[bool]]:
        log_dict, early_stop = super().on_test_end(test_episodes, test_mean_reward, progress_bar_dict)
        log_dict.update({
            "td_moving_avg": self.td_moving_avg.get()
        })
        if not self.args.deep and self.true_values is not None:
            if self.args.mask_model_errs:
                visited_states = {s_a[0] for test_episode in test_episodes for s_a in test_episode}
                v_comparison_mask = torch.tensor(list(visited_states))
                masked_weights = torch.ones(len(v_comparison_mask))
            else:
                v_comparison_mask = torch.all(
                        torch.arange(self.env.observation_space.n, dtype=torch.int64).reshape(-1, 1)
                        != self.env.terminal_states.reshape(1, -1),
                        axis=1)
                masked_weights = self.env.state_dist[v_comparison_mask]
            masked_weights /= torch.sum(masked_weights)
            masked_values = self.values[v_comparison_mask]
            masked_true_values = self.true_values[v_comparison_mask]
            values_rel_err = torch.abs(torch.log(masked_values / masked_true_values))
            values_rel_err[torch.isnan(values_rel_err)] = 0
            values_rel_err, rel_err_sort_inds = values_rel_err.sort()
            values_abs_err = torch.abs(masked_values - masked_true_values)
            values_abs_err, abs_err_sort_inds = values_abs_err.sort()
            log_dict.update({
                "values_wmean_rel_err": torch.sum(values_rel_err * masked_weights[rel_err_sort_inds]).item(),
                "values_max_rel_err": torch.max(values_rel_err).item(),
                "values_99prctl_rel_err": quantile_sorted(values_rel_err, 0.99).item(),
                "values_95prctl_rel_err": quantile_sorted(values_rel_err, 0.95).item(),
                "values_90prctl_rel_err": quantile_sorted(values_rel_err, 0.90).item(),
                "values_75prctl_rel_err": quantile_sorted(values_rel_err, 0.75).item(),
                "values_median_rel_err": quantile_sorted(values_rel_err, 0.50).item(),
                "values_rms_rel_err": torch.sqrt(torch.mean(values_rel_err ** 2)).item(),
                "values_wmean_abs_err": torch.sum(values_abs_err * masked_weights[abs_err_sort_inds]).item(),
                "values_max_abs_err": torch.max(values_abs_err).item(),
                "values_99prctl_abs_err": quantile_sorted(values_abs_err, 0.99).item(),
                "values_95prctl_abs_err": quantile_sorted(values_abs_err, 0.95).item(),
                "values_90prctl_abs_err": quantile_sorted(values_abs_err, 0.90).item(),
                "values_75prctl_abs_err": quantile_sorted(values_abs_err, 0.75).item(),
                "values_median_abs_err": quantile_sorted(values_abs_err, 0.50).item(),
                "values_rms_abs_err": torch.sqrt(torch.mean(values_abs_err ** 2)).item(),
            })
            if progress_bar_dict is not None and self.args.model_err_metric is not None:
                progress_bar_dict[self.args.model_err_metric] = log_dict[f"values_{self.args.model_err_metric}"]
        if (early_stop is None or early_stop) and (self.args.early_stop_td is not None or self.args.early_stop_model_err is not None):
            td_passed = (self.args.early_stop_td is None
                    or log_dict["td_moving_avg"] is not None and log_dict["td_moving_avg"] <= self.args.early_stop_td)
            v_err = log_dict[f"values_{self.args.model_err_metric}"]
            v_err_passed = (self.args.early_stop_model_err is None
                    or v_err <= self.args.early_stop_model_err)
            early_stop = td_passed and v_err_passed
        return log_dict, early_stop

    def on_training_end(self, run_id):
        # save final values if applicable
        if not self.args.deep:
            np.save(os.path.join(self.args.expt_dir, f"run{run_id}_V_values.npy"), self.values.to("cpu").numpy())

    def load_state_dict(self, state_dict) -> dict:
        super().load_state_dict(state_dict)
        self.values = torch.tensor(state_dict["values"])
        self.td_moving_avg = state_dict["td_moving_avg"]
        return state_dict

    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        state_dict["values"] = self.values.cpu().numpy()
        state_dict["td_moving_avg"] = self.td_moving_avg
        return state_dict


@LearningAgent.register("REINFORCE")
class REINFORCEAgent(LearningAgent):
    def __init__(self, args: Namespace, env: RLDiffEnvTensorWrapper):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state values and a learning rate.

        args:
            deep: whether to use a neural network for state embedding
            lr: The learning rate
            discount_factor: The discount factor for computing the Q-value
            J_moving_avg_alpha: alpha for the moving average of gradient ascent objective J
        """
        super().__init__(args, env)

        self.pi_network = None
        if args.deep:
            state_embedder = env.get_state_embedder()
            self.pi_network = nn.Sequential(state_embedder,
                                            nn.Linear(state_embedder.embed_dim, env.action_space.n)
                                           )
            self.optimizer = optim.AdamW(self.pi_network.parameters(), lr=args.lr)
        else:
            self.pi_logits = torch.zeros((env.observation_space.n, env.action_space.n))

        self.epsilon = args.start_eps

        self.J_moving_avg = MovingAverage(args.J_moving_avg_alpha)

    def _get_logits(self, obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]], keep_grad=False) -> torch.Tensor:
        if self.pi_network is None:
            return self.pi_logits[obs]
        if keep_grad:
            return self.pi_network(obs)
        with torch.no_grad():
            return self.pi_network(obs)

    def get_action(self, obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]], greedy=False) -> int:
        """
        Returns the best action if `greedy`
        otherwise a random action according to the current policy
        """
        logits = self._get_logits(obs)
        if greedy:
            return logits.argmax().item()
        distribution = Categorical(logits=logits)
        return distribution.sample().item()

    def setup_training(self, run_id):
        if not self.args.deep:
            self.true_pi_actions = None
            if self.args.early_stop_model_err is not None:
                assert self.args.true_model_path is not None, "Must specify path to true policy if using policy error to early-stop"
            if self.args.true_model_path is not None:
                self.true_pi_actions = torch.tensor(np.load(os.path.join(self.args.true_model_path, f"run{run_id}_pi_actions.npy")))

    def on_new_experience(
        self,
        obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Union[int, torch.Tensor, tuple[torch.Tensor, ...]],
    ) -> None:
        pass

    def on_episode_end(
        self,
        episode: Episode,
        reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> None:
        """ Updates the policy at the end of an episode. """
        assert episode is not None and reward is not None

        if self.pi_network is not None:
            self.optimizer.zero_grad()

        reward_to_go = reward
        rev_rewards_to_go = []
        for i in range(episode.num_steps):
            rev_rewards_to_go.append(reward_to_go)
            reward_to_go *= self.args.discount_factor
        rev_rewards_to_go = torch.tensor(rev_rewards_to_go)
        rev_obses = batch_obs([obs for obs, _ in reversed(episode)])
        rev_actions = torch.tensor([action for _, action in reversed(episode)])
        rev_logits = self._get_logits(rev_obses, keep_grad=True)
        rev_log_probs = torch.gather(rev_logits, dim=1, index=rev_actions.unsqueeze(1)).squeeze(1) \
                        - torch.logsumexp(rev_logits, dim=1)
        J = torch.sum(rev_rewards_to_go * rev_log_probs)

        self.J_moving_avg.update(J.item())

        if self.pi_network is None:
            # manual gradient update if policy is logit table
            self.pi_logits[rev_obses, rev_actions] += self.args.lr * rev_rewards_to_go
            probs = torch.softmax(self.pi_logits[rev_obses], 1)
            self.pi_logits[rev_obses] -= self.args.lr * probs * rev_rewards_to_go.unsqueeze(1)  # TODO what if there are repeated states?
        else:
            loss = -J
            loss.backward()
            self.optimizer.step()

    def on_test_end(
        self,
        test_episodes: list[Episode],
        test_mean_reward: float,
        progress_bar_dict: Optional[dict[str, Optional[Number]]]=None
    ) -> tuple[dict[str, Optional[float]], Optional[bool]]:
        log_dict = {
            "J_moving_avg": self.J_moving_avg.get()
        }
        if not self.args.deep and self.true_pi_actions is not None:
            if self.args.mask_model_errs:
                visited_states = {s_a[0] for test_episode in test_episodes for s_a in test_episode}
                pi_comparison_mask = torch.tensor(list(visited_states))
                masked_weights = torch.ones(len(pi_comparison_mask))
            else:
                pi_comparison_mask = torch.all(
                        torch.arange(self.env.observation_space.n, dtype=torch.int64).reshape(-1, 1)
                        != self.env.terminal_states.reshape(1, -1),
                        axis=1)
                masked_weights = self.env.state_dist[pi_comparison_mask]
            masked_weights /= torch.sum(masked_weights)
            masked_pi_logits = self.pi_logits[pi_comparison_mask]
            masked_true_pi_actions = self.true_pi_actions[pi_comparison_mask]
            masked_true_action_logits = torch.where(masked_true_pi_actions, masked_pi_logits, -torch.inf)
            logsumexp_masked_pi_logits = torch.logsumexp(masked_pi_logits, dim=-1)
            pi_kl = logsumexp_masked_pi_logits - torch.logsumexp(masked_true_action_logits, dim=-1)
            pi_kl, kl_sort_inds = pi_kl.sort()
            pi_nlc = logsumexp_masked_pi_logits - torch.max(masked_pi_logits, dim=-1).values  # NLC: Negative Log Confidence
            pi_nlc, nlc_sort_inds = pi_nlc.sort()
            log_dict.update({
                "pi_wmean_kl": torch.sum(pi_kl * masked_weights[kl_sort_inds]).item(),
                "pi_max_kl": torch.max(pi_kl).item(),
                "pi_99prctl_kl": quantile_sorted(pi_kl, 0.99).item(),
                "pi_95prctl_kl": quantile_sorted(pi_kl, 0.95).item(),
                "pi_90prctl_kl": quantile_sorted(pi_kl, 0.90).item(),
                "pi_75prctl_kl": quantile_sorted(pi_kl, 0.75).item(),
                "pi_median_kl": quantile_sorted(pi_kl, 0.50).item(),
                "pi_rms_kl": torch.sqrt(torch.mean(pi_kl ** 2)).item(),
                "pi_wmean_nlc": torch.sum(pi_nlc * masked_weights[nlc_sort_inds]).item(),
                "pi_max_nlc": torch.max(pi_nlc).item(),
                "pi_99prctl_nlc": quantile_sorted(pi_nlc, 0.99).item(),
                "pi_95prctl_nlc": quantile_sorted(pi_nlc, 0.95).item(),
                "pi_90prctl_nlc": quantile_sorted(pi_nlc, 0.90).item(),
                "pi_75prctl_nlc": quantile_sorted(pi_nlc, 0.75).item(),
                "pi_median_nlc": quantile_sorted(pi_nlc, 0.50).item(),
                "pi_rms_nlc": torch.sqrt(torch.mean(pi_nlc ** 2)).item(),
            })
            if progress_bar_dict is not None and self.args.model_err_metric is not None:
                progress_bar_dict[self.args.model_err_metric] = log_dict[f"pi_{self.args.model_err_metric}"]
        early_stop: Optional[bool] = None
        if self.args.early_stop_model_err is not None:
            pi_err = log_dict[f"pi_{self.args.model_err_metric}"]
            early_stop = (self.args.early_stop_model_err is None
                    or pi_err <= self.args.early_stop_model_err)
        return log_dict, early_stop

    def on_training_end(self, run_id):
        # save final pi logits if applicable
        if not self.args.deep:
            np.save(os.path.join(self.args.expt_dir, f"run{run_id}_pi_logits.npy"), self.pi_logits.to("cpu").numpy())

    def load_state_dict(self, state_dict) -> dict:
        super().load_state_dict(state_dict)
        if self.args.deep:
            self.pi_network.load_state_dict(state_dict["pi_network"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            self.pi_logits = torch.tensor(state_dict["pi_logits"])
        self.J_moving_avg = state_dict["J_moving_avg"]
        return state_dict

    def get_state_dict(self) -> dict:
        state_dict = super().get_state_dict()
        if self.args.deep:
            state_dict["pi_network"] = self.pi_network.state_dict()
            state_dict["optimizer"] = self.optimizer.state_dict()
        else:
            state_dict["pi_logits"] = self.pi_logits.cpu().numpy()
        state_dict["J_moving_avg"] = self.J_moving_avg
        return state_dict
