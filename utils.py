import random
import time
from types import FrameType
from typing import Optional, Union, Sequence
from numbers import Number
import signal
import argparse
from datetime import datetime

import numpy as np
import torch

from custom_types import NumpyArrayObs


def format_stats_dict(stats: dict[str, Optional[Number]]):
    def format_number(number: Number):
        if isinstance(number, int):
            return f"{number:7d}"
        return f"{number:7.4f}"
    stats_str = ', '.join([key + ':' + (' ' * 7 if value is None else format_number(value))
                           for key, value in stats.items()])
    return stats_str


def quantile_sorted(array, q: float):
    idx = q * (len(array) - 1)
    int_idx = int(idx)
    if idx == int_idx:
        return array[int_idx]
    frac_idx = idx - int_idx
    return (1 - frac_idx) * array[int_idx] + frac_idx * array[int_idx + 1]


def batch_obs(obs_list: Sequence[Union[int, torch.Tensor, tuple[torch.Tensor, ...]]]) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    assert obs_list, "Empty batch"
    if isinstance(obs_list[0], int):
        return torch.tensor(obs_list)
    if isinstance(obs_list[0], tuple):
        return tuple(map(torch.stack, zip(*obs_list)))
    return torch.stack(obs_list)


def batch_obs_numpy(obs_list: Sequence[NumpyArrayObs]) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    assert obs_list, "Empty batch"
    if isinstance(obs_list[0], tuple):
        return tuple(map(np.stack, zip(*obs_list)))
    return np.stack(obs_list)


def array_obs_float64_to_float32(obs: NumpyArrayObs):
    if isinstance(obs, tuple):
        return tuple(map(lambda obs_part: obs_part.astype(np.float32) if obs_part.dtype == np.float64 else obs_part, obs))
    return obs.astype(np.float32) if obs.dtype == np.float64 else obs


def flexible_batch_dims(num_nonbatch_dims):
    def decorator(func):
        def decorated_func(self, obs):
            if isinstance(num_nonbatch_dims, tuple):
                batch_dims_list = [obs_part.shape[:len(obs_part.shape)-num_nonbatch_dims_part]
                                   for obs_part, num_nonbatch_dims_part in zip(obs, num_nonbatch_dims)]
                batch_dims = batch_dims_list[0]
                assert all(batch_dims_part == batch_dims for batch_dims_part in batch_dims_list)
                new_obs = tuple(obs_part.reshape((-1, *obs_part.shape[len(obs_part.shape)-num_nonbatch_dims_part:]))
                                for obs_part, num_nonbatch_dims_part in zip(obs, num_nonbatch_dims))
            else:
                batch_dims = obs.shape[:-num_nonbatch_dims]
                new_obs = obs.reshape((-1, *obs.shape[-num_nonbatch_dims:]))
            output = func(self, new_obs)
            return output.reshape((*batch_dims, *output.shape[1:]))
        return decorated_func
    return decorator


class Episode(object):
    def __init__(self, s_a_pairs, final_state):
        self._s_a_pairs = s_a_pairs
        self._final_state = final_state

    def __iter__(self):
        return iter(self._s_a_pairs)

    def __reversed__(self):
        return reversed(self._s_a_pairs)

    @property
    def final_state(self):
        return self._final_state

    @property
    def num_steps(self):
        return len(self._s_a_pairs)


def diff_args(base_args, new_args):
    diff = {}
    for k, v in vars(new_args).items():
        if v != vars(base_args).get(k):
            diff[k] = v
    return diff


""" Adapted from OpenAI Baselines https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py """
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, action, reward, terminated, next_obs):
        data = (obs, action, reward, terminated, next_obs)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, actions, rewards, terminateds, next_obses = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, reward, terminated, next_obs = data
            obses.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminateds.append(terminated)
            next_obses.append(next_obs)
        obses_tensor = batch_obs(obses)
        next_obses_tensor = batch_obs(next_obses)
        return obses_tensor, torch.tensor(actions), torch.tensor(rewards), torch.tensor(terminateds), next_obses_tensor

    def sample(self, batch_size) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], torch.IntTensor, torch.Tensor, torch.BoolTensor, Union[torch.Tensor, tuple[torch.Tensor]]]:
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: torch.Tensor
            batch of observations
        act_batch: torch.Tensor
            batch of actions executed given obs_batch
        rew_batch: torch.Tensor
            rewards received as results of executing act_batch
        terminated_mask: torch.Tensor
            terminated_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        next_obs_batch: torch.Tensor
            next set of observations seen after executing act_batch
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class MovingAverage:
    def __init__(self, alpha: float):
        self._value: Optional[float] = None
        self._alpha = alpha

    def get(self) -> Optional[float]:
        return self._value

    def update(self, new_value: float) -> float:
        if self._value is None:
            self._value = new_value
        else:
            self._value = self._alpha * new_value + (1 - self._alpha) * self._value
        return self._value


class RewardTarget:
    def __init__(self, start_target: float, step: float, final_target: float):
        self._target = start_target
        self._step = step
        self._final_target = final_target

    def get(self) -> float:
        return self._target

    def update(self) -> float:
        self._target = min(self._target + self._step, self._final_target)
        return self._target


class EarlyStopper:
    def __init__(self, early_stop_reward: Optional[float], early_stop_reward_delay: int):
        self.early_stop_reward = early_stop_reward
        self.early_stop_reward_delay = early_stop_reward_delay
        self.last_bad_reward_episode = -1

    def update(self, test_mean_reward, episode_idx, extra_early_stop):
        early_stop = False
        if self.early_stop_reward is not None and test_mean_reward < self.early_stop_reward:
            self.last_bad_reward_episode = episode_idx
        if self.early_stop_reward is not None or extra_early_stop is not None:
            reward_passed = (self.early_stop_reward is None
                    or episode_idx > self.last_bad_reward_episode + self.early_stop_reward_delay)
            extra_passed = extra_early_stop is None or extra_early_stop
            if reward_passed and extra_passed:
                early_stop = True
        return early_stop


class StopWatch:
    def __init__(self):
        self._running = False
        self._total_time = 0.0

    @property
    def total_time(self):
        return self._total_time

    def start(self):
        if self._running:
            raise Exception("Stopwatch already started")
        self._last_start_time = time.time()
        self._running = True

    def stop(self):
        if not self._running:
            raise Exception("Stopwatch already stopped")
        time_interval = time.time() - self._last_start_time
        self._total_time += time_interval
        self._running = False
        return time_interval


class RLState:
    def __init__(self, args):
        self.start_time = time.time()
        self.training_stopwatch = StopWatch()
        self.learning_stopwatch = StopWatch()
        self.exploring_stopwatch = StopWatch()
        self.env_steps = 0
        self.env_base_steps = 0
        self.train_reward_moving_avg = MovingAverage(args.train_reward_moving_avg_alpha)
        self.early_stopper = EarlyStopper(args.early_stop_reward, args.early_stop_reward_delay)
        self.next_test_it = args.test_every


# Adapted from https://stackoverflow.com/a/21919644
class DelayedInterrupt:
    def __enter__(self):
        self.sig_received: Optional[signal.Signals] = None
        self.frame_received: Optional[FrameType] = None
        self.old_handlers = {
            signal.SIGINT: signal.signal(signal.SIGINT, self.handler),
            signal.SIGTERM: signal.signal(signal.SIGTERM, self.handler),
        }

    def handler(self, sig, frame):
        self.sig_received = sig
        self.frame_received = frame

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self.old_handlers[signal.SIGTERM])
        if self.sig_received is not None:
            self.old_handlers[self.sig_received](self.sig_received, self.frame_received)


def get_arg_parser():
    model_err_metrics = [f"{prefix}_{suffix}"
            for suffix in ["rel_err", "abs_err", "kl", "nlc"]
            for prefix in ["wmean", "max", "99prctl", "95prctl", "90prctl", "75prctl", "median", "rms"]
        ]

    parser = argparse.ArgumentParser(description="Run several hRL training runs on the same base environment in parallel")
    parser.add_argument("--output_path", type=str, default="output_main/",
            help="Base directory of training output")
    parser.add_argument("--expt_name", type=str, default=datetime.now().strftime("expt_20%y-%m-%d_%Hh%Mm%Ss"),
            help="Name of experiment")
    parser.add_argument("--env", choices=["CliffWalking-v0", "RubiksCube222-v0", "CompILE-v0", "NPuzzle-v0"], required=True,
            help="Name of (base) environment")
    parser.add_argument("--env_config", default="{}",
            help="Environment config as a string containing a Python dictionary")
    parser.add_argument("--rl_algo", choices=["QLearning", "ValueIteration", "REINFORCE"], required=True,
            help="RL/planning algorithm to use")
    parser.add_argument("--no_explore", action="store_true",
            help="Apply the planning algorithm rather than RL algorithm (only available for QLearning, ValueIteration)")
    parser.add_argument("--deep", action="store_true",
            help="Use a neural network to embed observations (only available for QLearning, REINFORCE)")
    parser.add_argument("--seed", type=int, default=0,
            help="Random seed")
    parser.add_argument("--no_parallel", action="store_true",
            help="Run hRL training runs of the same base environment sequentially rather than in parallel")
    parser.add_argument("--num_workers", type=int, default=16,
            help="Number of multiprocessing workers")
    parser.add_argument("--use_gpu", action="store_true",
            help="Use GPU(s)")
    parser.add_argument("--ids_to_run", type=str,
            help="String defining Python container object of ids to run")
    parser.add_argument("--overwrite", "-f", action="store_true",
            help="Force overwrite of output directory")
    # Macroactions
    parser.add_argument("--abs_path",
            help="Path to macroactions; if None, then use the next 4 options to generate macroactions")
    parser.add_argument("--n_abs_spaces_per_size", type=int, nargs='+', default=[1],
            help="Number of sets of macroactions to randomly generate for each size of the set; "
                 'e.g., "1 4 6 3" means "1 set of 0 abstractions, 4 sets of 1 abstraction each, '
                 '6 sets of 2 abstractions each, and 3 sets of 3 abstractions each"')
    parser.add_argument("--avg_abs_len", type=float, default=2,
            help="Average macroaction length to randomly generate")
    parser.add_argument("--abs_base_action_weights", type=str,
            help="Frequencies of different actions in randomly generated macroactions; "
                 'as a list, e.g., "[0.25, 0.25, 0.25, 0.25]", '
                 'or as a dict, e.g. {(0, 1): (0.3, [0.5, 0.5]), (2, 3): (0.7, [0.5, 0.5])}')
    parser.add_argument("--extra_abs_path", type=str,
            help="Path to extra macroactions, e.g., optimal ones learnt by LEMMA and those based on them")
    # LOVE
    parser.add_argument("--love", action="store_true",
            help="Use LOVE options rather than macroactions")
    parser.add_argument("--love_ckpt_path",
            help="Path to LOVE options")
    parser.add_argument("--love_model_config_path",
            help="Path to LOVE options config")
    parser.add_argument("--love_traj_path",
            help="Path to trajectories used to learn LOVE options")
    parser.add_argument("--love_threshold", type=float, default=0.0,
            help="Frequency threshold for keeping LOVE options")
    # Environment
    parser.add_argument("--truncate_steps", type=int,
            help="Truncate episode after this number of steps")
    parser.add_argument("--truncate_base_steps", type=int,
            help="Truncate episode after this number of base environment steps")
    parser.add_argument("--max_test_steps", type=int, default=100,
            help="Maximum number of environment steps during test")
    # RL hyperparameters
    parser.add_argument("--lr", type=float, required=True,
            help="RL learning rate")
    parser.add_argument("--n_episodes", type=int,
            help="Max number of training episodes")
    parser.add_argument("--n_env_steps", type=int,
            help="Max number of training environment steps")
    parser.add_argument("--n_env_base_steps", type=int,
            help="Max number of training base environment steps")
    parser.add_argument("--early_stop_reward", type=float,
            help="Reward threshold for early stopping")
    parser.add_argument("--early_stop_reward_delay", type=int, default=0,
            help="Delay early stopping due to reward by this many episodes")
    parser.add_argument("--early_stop_td", type=float,
            help="Temporal difference error threshold for early stopping (only applies to QLearning, ValueIteration)")
    parser.add_argument("--early_stop_model_err", type=float,
            help="Model error threshold for early stopping (see --model_err_metric option below)")
    parser.add_argument("--test_every", type=int, default=1,
            help="Test once every (at least) this many episodes; see also --test_every_ratio below")
    parser.add_argument("--test_every_ratio", type=float, default=1.0,
            help="Minimum episode number ratio between consecutive tests; see also --test_every above")
    parser.add_argument("--test_episodes", type=int, default=1,
            help="Number of episodes used during test")
    parser.add_argument("--test_no_greedy", action="store_true",
            help="Don't use greedy policy during test")
    parser.add_argument("--use_replay_buffer", action="store_true",
            help="Use replay buffer (only applies to QLearning, ValueIteration)")
    parser.add_argument("--replay_buffer_size", type=int, default=1000,
            help="Size of replay buffer, if --use_replay_buffer is set")
    parser.add_argument("--batch_size", type=int, default=32,
            help="Size of batch sampled from replay buffer, if --use_replay_buffer is set")
    parser.add_argument("--update_every", type=int, default=4,
            help="Frequency of agent updates, if --use_replay_buffer is set")
    parser.add_argument("--adaptive_eps_greedy", action="store_true",
            help="Use an adaptive epsilon-greedy schedule based on test reward")
    parser.add_argument("--start_eps", type=float, default=1.0,
            help="Initial epsilon of epsilon-greedy schedule")
    parser.add_argument("--eps_decay", type=float, default=0.002,
            help="Step size of epsilon-greedy schedule")
    parser.add_argument("--final_eps", type=float, default=0.1,
            help="Final epsilon of epsilon-greedy schedule")
    parser.add_argument("--reward_start_target", type=float, default=0.002,
            help="Start reward target for adaptive epsilon greedy")
    parser.add_argument("--reward_target_step", type=float, default=0.002,
            help="Step size of reward target for adaptive epsilon greedy")
    parser.add_argument("--reward_final_target", type=float, default=1.0,
            help="Final reward target for adaptive epsilon greedy")
    parser.add_argument("--discount_factor", type=float, default=0.95,
            help="MDP discount factor gamma")
    # Logging parameters
    parser.add_argument("--train_reward_moving_avg_alpha", type=float, default=0.01,
            help="Weight parameter of train reward moving average")
    parser.add_argument("--td_moving_avg_alpha", type=float, default=0.01,
            help="Weight parameter of td error moving average (for QLearning and ValueIteration)")
    parser.add_argument("--J_moving_avg_alpha", type=float, default=0.01,
            help="Weight parameter of J objective moving average (for REINFORCE)")
    parser.add_argument("--log_test_episodes", action="store_true",
            help="Include test episodes in history")
    # Compare values or policy with ground truth
    parser.add_argument("--model_err_metric", choices=model_err_metrics,
            help="Metric for computing error in values or policy compared to ground truth values or optimal policy")
    parser.add_argument("--true_model_path",
            help="Path to ground truth values or optimal policy")
    parser.add_argument("--mask_model_errs", action="store_true",
            help="Restrict comparison to test trajectories")
    # Save checkpoints
    parser.add_argument("--save_every", type=int,
            help="Frequency of saving model checkpoints and training history")

    return parser
