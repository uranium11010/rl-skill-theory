import os, shutil
from multiprocessing import pool
import time
from datetime import timedelta
import itertools
import pickle as pkl
import json
import random

from tqdm import tqdm
import numpy as np
import torch

import gymnasium as gym

from abstractions.abstractions import Axiom, AxiomSeq
from love import HierarchicalStateSpaceModel
from compute_envinfo import generate_abstractions
from abstract_trajectories_love import ActionEncoder, GridDecoder

from wrappers import RLDiffEnvAdapter, RLDiffEnvAbsWrapper, RLDiffEnvLoveWrapper, RLDiffEnvArrayObsWrapper, RLDiffEnvTensorWrapper
from learning_agent import LearningAgent
from utils import DelayedInterrupt, EarlyStopper, Episode, RLState, StopWatch, format_stats_dict, get_arg_parser
from love_utils import get_traj_data_loader


def run_episode(args, env, agent, test=False, seed=None,
        learning_stopwatch=None, exploring_stopwatch=None,
        episode_return_state_index=False):
    if exploring_stopwatch is not None:
        exploring_stopwatch.start()
    if seed is None:
        seed = random.randint(0, int(1e10))
    obs, info = env.reset(seed=seed)
    total_reward = 0
    s_a_pairs = []
    num_base_actions = 0
    while True:
        state_index = env.state_index
        action = agent.get_action(obs, greedy=test and args.test_greedy)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if episode_return_state_index:
            s_a_pairs.append((state_index, action))
        else:
            s_a_pairs.append((obs, action))
        total_reward += reward
        num_base_actions += info["steps"]
        if not test:
            if exploring_stopwatch is not None:
                exploring_stopwatch.stop()
            if learning_stopwatch is not None:
                learning_stopwatch.start()
            agent.on_new_experience(obs, action, reward, terminated, next_obs)
            if learning_stopwatch is not None:
                learning_stopwatch.stop()
            if exploring_stopwatch is not None:
                exploring_stopwatch.start()
        if terminated or truncated or (test and len(s_a_pairs) >= args.max_test_steps):
            if exploring_stopwatch is not None:
                exploring_stopwatch.stop()
            final_state = env.state_index if episode_return_state_index else next_obs
            episode = Episode(s_a_pairs, final_state)
            return total_reward, episode, num_base_actions
        obs = next_obs


def run_test_loop(args, env, agent):
    test_rewards = []
    test_episodes = []
    test_env_steps = []
    test_env_base_steps = []
    test_rng = np.random.default_rng(args.seed)
    for _ in range(args.test_episodes):
        test_reward, episode, num_base_actions = run_episode(
                args, env, agent,
                test=True, seed=test_rng.choice(int(1e10)),
                episode_return_state_index=True)
        test_rewards.append(test_reward)
        test_episodes.append(episode)
        test_env_steps.append(episode.num_steps)
        test_env_base_steps.append(num_base_actions)
    test_mean_reward = np.mean(test_rewards)
    # log results
    log_dict = {
        "test_mean_reward": test_mean_reward,
        "test_mean_env_steps": np.mean(test_env_steps),
        "test_mean_env_base_steps": np.mean(test_env_base_steps),
        "test_rewards": test_rewards,
    }
    return log_dict, test_episodes


def run_RL_train_loop(args, env, agent, run_id=0):
    # initialize training
    agent.setup_training(run_id)
    start_i = 0
    ckpt_paths = [os.path.join(args.expt_dir, f"run{run_id}_{ckpt_type}.pkl")
                  for ckpt_type in ["train_history", "training_state"]]
    temp_ckpt_paths = [ckpt_path + ".temp" for ckpt_path in ckpt_paths]
    if all(os.path.exists(ckpt_path) for ckpt_path in ckpt_paths):
        with open(ckpt_paths[0], 'rb') as f:
            history = pkl.load(f)
        with open(ckpt_paths[1], 'rb') as f:
            start_i, rl_state, agent_state, random_state = pkl.load(f)
        agent.load_state_dict(agent_state)
        random.setstate(random_state[0])
        np.random.set_state(random_state[1])
        torch.set_rng_state(random_state[2])
        env.np_random = random_state[3]
        env.action_space = random_state[4]
    else:
        history = []
        rl_state = RLState(args)
    # begin training
    with tqdm(initial=start_i, total=args.n_episodes, desc=f"#{run_id}", position=run_id % args.num_workers) as progress_bar:
        progress_bar_dict = {}
        episode_iterator = itertools.count(start_i) if args.n_episodes is None else range(start_i, args.n_episodes)
        for i in episode_iterator:
            # run one iteration of training
            early_stop = False
            rl_state.training_stopwatch.start()
            train_reward, episode, num_base_actions = run_episode(
                    args, env, agent,
                    learning_stopwatch=rl_state.learning_stopwatch, exploring_stopwatch=rl_state.exploring_stopwatch)
            rl_state.train_reward_moving_avg.update(train_reward)
            rl_state.env_steps += episode.num_steps
            rl_state.env_base_steps += num_base_actions
            progress_bar_dict["train_r"] = rl_state.train_reward_moving_avg.get()
            progress_bar_dict["N"] = rl_state.env_steps
            progress_bar_dict["N(base)"] = rl_state.env_base_steps
            rl_state.learning_stopwatch.start()
            agent.on_episode_end(episode=episode, reward=train_reward, progress_bar_dict=progress_bar_dict)
            rl_state.learning_stopwatch.stop()
            rl_state.training_stopwatch.stop()
            # test and log
            if i + 1 == round(rl_state.next_test_it):
                rl_state.next_test_it = max(rl_state.next_test_it + args.test_every, rl_state.next_test_it * args.test_every_ratio)
                log_dict, test_episodes = run_test_loop(args, env, agent)
                extra_log_dict, extra_early_stop = agent.on_test_end(test_episodes, log_dict["test_mean_reward"], progress_bar_dict)
                progress_bar_dict["test_r"] = log_dict["test_mean_reward"]
                log_dict.update({
                    "episode": i + 1,
                    "elapsed_time": time.time() - rl_state.start_time,
                    "training_time": rl_state.training_stopwatch.total_time,
                    "train_reward_moving_avg": rl_state.train_reward_moving_avg.get(),
                    "env_steps": rl_state.env_steps,
                    "env_base_steps": rl_state.env_base_steps,
                    "learning_time": rl_state.learning_stopwatch.total_time,
                    "exploring_time": rl_state.exploring_stopwatch.total_time,
                    **extra_log_dict
                })
                if args.log_test_episodes:
                    log_dict["test_episodes"] = list(test_episodes)
                history.append(log_dict)
                # early stopping
                early_stop = rl_state.early_stopper.update(log_dict["test_mean_reward"], i, extra_early_stop)
            # update tqdm bar
            progress_bar.postfix = format_stats_dict(progress_bar_dict)
            progress_bar.update()
            # end training if early stop or env steps budget reached
            if (early_stop
                    or (args.n_env_steps is not None and rl_state.env_steps >= args.n_env_steps)
                    or (args.n_env_base_steps is not None and rl_state.env_base_steps >= args.n_env_base_steps)):
                break
            # save history and training state
            if args.save_every is not None and (i + 1) % args.save_every == 0:
                with open(temp_ckpt_paths[0], 'wb') as f:
                    pkl.dump(history, f)
                random_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), env.np_random, env.action_space)
                with open(temp_ckpt_paths[1], 'wb') as f:
                    pkl.dump((i + 1, rl_state, agent.get_state_dict(), random_state), f)
                with DelayedInterrupt():
                    for temp_ckpt_path, ckpt_path in zip(temp_ckpt_paths, ckpt_paths):
                        os.rename(temp_ckpt_path, ckpt_path)
    agent.on_training_end(run_id)
    # save history
    history[-1]["finished"] = True
    with open(temp_ckpt_paths[0], 'wb') as f:
        pkl.dump(history, f)
    with DelayedInterrupt():
        os.rename(temp_ckpt_paths[0], ckpt_paths[0])


def run_planning_train_loop(args, env, agent, run_id=0):
    # initialize training
    start_time = time.time()
    training_stopwatch = StopWatch()
    agent.setup_training(run_id)
    history = []
    early_stopper = EarlyStopper(args.early_stop_reward, args.early_stop_reward_delay)
    next_test_it = args.test_every
    # begin training
    with tqdm(total=args.n_episodes, desc=f"#{run_id}", position=run_id % args.num_workers) as progress_bar:
        progress_bar_dict = {}
        episode_iterator = itertools.count() if args.n_episodes is None else range(args.n_episodes)
        for i in episode_iterator:
            # run one iteration of training
            early_stop = False
            training_stopwatch.start()
            agent.update_all()
            agent.on_iteration_end(progress_bar_dict=progress_bar_dict)
            training_stopwatch.stop()
            # test and log
            if i + 1 == round(next_test_it):
                next_test_it = max(next_test_it + args.test_every, next_test_it * args.test_every_ratio)
                log_dict, test_episodes = run_test_loop(args, env, agent)
                progress_bar_dict["test_r"] = log_dict["test_mean_reward"]
                # log results
                log_dict.update({
                    "num_it": i + 1,
                    "elapsed_time": time.time() - start_time,
                    "training_time": training_stopwatch.total_time,
                })
                if args.log_test_episodes:
                    log_dict["test_episodes"] = list(test_episodes)
                extra_log_dict, extra_early_stop = agent.on_test_end(test_episodes, progress_bar_dict)
                log_dict.update(extra_log_dict)
                history.append(log_dict)
                # early stopping
                early_stop = early_stopper.update(log_dict["test_mean_reward"], i, extra_early_stop)
            # save history so far
            if args.save_every is not None and (i + 1) % args.save_every == 0:
                with open(os.path.join(args.expt_dir, f"run{run_id}_train_history.pkl"), 'wb') as f:
                    pkl.dump(history, f)
            # update tqdm bar
            progress_bar.postfix = format_stats_dict(progress_bar_dict)
            progress_bar.update()
            if early_stop:
                break
    agent.on_training_end(run_id)
    # save history
    with open(os.path.join(args.expt_dir, f"run{run_id}_train_history.pkl"), 'wb') as f:
        pkl.dump(history, f)


def train_different_abstractions(args, base_env, run_id, abstraction_int_lists):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu:
        torch.set_default_device(run_id % torch.cuda.device_count())
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    abstractions = [AxiomSeq([Axiom(action) for action in abstraction_int_list])
                            for abstraction_int_list in abstraction_int_lists]
    env = RLDiffEnvTensorWrapper(RLDiffEnvAbsWrapper(base_env, abstractions, args.truncate_steps, args.truncate_base_steps))
    env.action_space.seed(args.seed)
    agent = LearningAgent.new(args, env)
    if args.no_explore:
        run_planning_train_loop(args, env, agent, run_id=run_id)
    else:
        run_RL_train_loop(args, env, agent, run_id=run_id)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    args.test_greedy = not args.test_no_greedy
    args.env_config = eval(args.env_config)
    args.ids_to_run = list(eval(args.ids_to_run)) if args.ids_to_run is not None else None
    if args.abs_base_action_weights is not None:
        args.abs_base_action_weights = eval(args.abs_base_action_weights)
    has_training_end_condition = (args.n_episodes is not None
            or args.early_stop_reward is not None or args.early_stop_td is not None or args.early_stop_q_rel_err is not None)
    assert has_training_end_condition, "Must specify condition for ending training"

    torch.set_default_dtype(torch.float64)
    np.random.seed(args.seed)

    # Set up base environment
    base_env = RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config))
    if args.deep:
        base_env = RLDiffEnvArrayObsWrapper(base_env)
    if args.no_explore or args.rl_algo == "ValueIteration":
        base_env.transition_table  # this caches the transition table

    # Set up experiment directory
    env_dir_string = args.env + (f"_{base_env.config_string}" if base_env.config_string is not None else "")
    args.expt_dir = os.path.join(args.output_path,
                                 env_dir_string
                                 + f"_{args.rl_algo}"
                                 + ("_no-expl" if args.no_explore else "")
                                 + ("_deep" if args.deep else ""),
                                 args.expt_name)
    if os.path.exists(args.expt_dir) and os.listdir(args.expt_dir):
        if args.overwrite:
            print(f"Experiment directory {args.expt_dir} not empty! Removing contents...")
            shutil.rmtree(args.expt_dir, ignore_errors=True)  # remove directory if exists
            time.sleep(5)
        elif args.no_explore or args.save_every is None:
            raise Exception(f"Experiment directory {args.expt_dir} not empty!")
        else:
            print(f"Experiment directory {args.expt_dir} not empty! Resuming training run...")
    print(f"Starting experiment in directory {args.expt_dir}")
    os.makedirs(args.expt_dir, exist_ok=True)

    if args.love:  # hRL on LOVE options
        assert args.deep and not args.no_explore and args.ids_to_run is None

        # Load LOVE options
        with open(args.love_traj_path, 'rb') as f:
            trajectories = pkl.load(f)
        train_loader, test_loader = get_traj_data_loader(trajectories, 64, shuffle_train=False)
        dummy_env = RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config))
        encoder = dummy_env.get_state_embedder()
        action_encoder = ActionEncoder(
            action_size=dummy_env.action_space.n,
            embedding_size=encoder.embed_dim,
        )
        decoder = GridDecoder(
            input_size=encoder.embed_dim,
            action_size=dummy_env.action_space.n,
            feat_size=encoder.embed_dim,
        )
        with open(args.love_model_config_path, 'r') as f:
            model_config = json.load(f)
        hssm = HierarchicalStateSpaceModel(
            action_encoder=action_encoder,
            encoder=encoder,
            decoder=decoder,
            belief_size=encoder.embed_dim,
            state_size=model_config["state_size"],
            num_layers=model_config["num_layers"],
            max_seg_len=model_config["max_seg_len"],
            max_seg_num=model_config["max_seg_num"],
            latent_n=model_config["latent_n"],
            use_min_length_boundary_mask=True,
            ddo=model_config["ddo"],
            output_normal=True
        )
        hssm.load_state_dict(torch.load(args.love_ckpt_path))
        hssm = hssm.cpu()
        hssm.eval()
        hssm = hssm.cuda()

        # Set seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_default_device("cuda")
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        # Create environment wrapper for LOVE options
        env = RLDiffEnvLoveWrapper(RLDiffEnvTensorWrapper(base_env),
                hssm, train_loader, init_size=1, threshold=args.love_threshold,
                truncate_steps=args.truncate_steps, truncate_base_steps=args.truncate_base_steps)
        env.action_space.seed(args.seed)

        # Train hRL agent
        agent = LearningAgent.new(args, env)
        run_RL_train_loop(args, env, agent)

    else:  # hRL on LEMMA macroactions (base environment is specified as empty set of macroactions)

        # Load different sets of LEMMA macroactions
        if args.abs_path is not None:
            with open(args.abs_path, 'r') as f:
                abstractions_list = json.load(f)
        else:
            abstractions_list = generate_abstractions(
                                    base_env.action_space.n,
                                    args.extra_abs_path,
                                    args.n_abs_spaces_per_size,
                                    args.avg_abs_len,
                                    args.abs_base_action_weights
                                )
        print("ABSTRACTIONS:")
        for abstractions in abstractions_list:
            print(abstractions)
        with open(os.path.join(args.expt_dir, f"all_abstractions.json"), 'w') as f:
            json.dump(abstractions_list, f)

        # Run hRL training on different macroaction augmentations
        ids_to_run = list(range(len(abstractions_list))) if args.ids_to_run is None else args.ids_to_run
        start_time = time.time()
        if args.no_parallel:
            for run_id in ids_to_run:
                train_different_abstractions(args, base_env, run_id, abstractions_list[run_id])
        else:
            # Check run histories to skip runs that have already completed
            unfinished_ids_to_run = []
            for run_id in ids_to_run:
                history_path = os.path.join(args.expt_dir, f"run{run_id}_train_history.pkl")
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        history = pkl.load(f)
                    if history and history[-1].get("finished"):
                        continue
                unfinished_ids_to_run.append(run_id)
            print("RUN IDS:", unfinished_ids_to_run)
            # Run hRL training runs in parallel
            with pool.Pool(args.num_workers, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock) as p:
                p.starmap(train_different_abstractions,
                          [(args, base_env, run_id, abstractions_list[run_id]) for run_id in unfinished_ids_to_run])
        print("TOTAL TIME:", timedelta(seconds=time.time() - start_time))
