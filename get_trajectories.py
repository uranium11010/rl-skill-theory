import os
import argparse
import random
import pickle as pkl
from tqdm import tqdm

import gymnasium as gym

from rldiff_envs import bfs
from wrappers import RLDiffEnvAdapter, RLDiffEnvArrayObsWrapper


def get_solution(start_state, goal_state, transition_table):
    return bfs(
        start_state,
        goal_state,
        transition_table,
        start_value=[],
        value_update=lambda trajectory, state, action, next_state: trajectory + [action],
        default_value=None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="trajectories/")
    parser.add_argument("--env", choices=["CliffWalking-v0", "RubiksCube222-v0", "CompILE-v0", "NPuzzle-v0"], required=True)
    parser.add_argument("--env_config", default="{}", help="Dictionary as string")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trajectories", type=int, required=True)
    args = parser.parse_args()
    args.env_config = eval(args.env_config)

    random.seed(args.seed)

    trajectories = []
    env = RLDiffEnvArrayObsWrapper(RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config)))
    for i in tqdm(range(args.num_trajectories)):
        obs_sequence = [env.reset(seed=random.randint(0, int(1e19)))[0]]
        action_sequence = get_solution(env.state_index, env.unwrapped.goal_state_index, env.transition_table)
        assert action_sequence is not None, "No solution to state!"
        for action in action_sequence:
            obs_sequence.append(env.step(action)[0])
        trajectories.append(list(zip(obs_sequence, action_sequence)))

    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, args.env
                                            + (f"_{env.config_string}" if env.config_string is not None else "")
                                            + "_traj.pkl")
    print(f"Saving trajectories to {file_path}...")
    with open(file_path, 'wb') as f:
        pkl.dump(trajectories, f)
