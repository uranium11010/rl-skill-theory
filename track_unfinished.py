import os
import json
import pickle as pkl
import shlex
from colorama import Fore

import gymnasium as gym

from wrappers import RLDiffEnvAdapter
from utils import get_arg_parser


parser = get_arg_parser()
output_path = "output_main"
envinfo_output_path = "envinfo_output_main"
scripts_dir = "scripts_main"
results_path = "run_statuses.json"
deep_algo_combinations = [(False, "QLearning"), (False, "ValueIteration"), (False, "REINFORCE"), (True, "QLearning")]
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results_dict = json.load(f)
else:
    results_dict = {}
num_scripts = 0
for scripts_subdir in sorted(os.listdir(scripts_dir)):
    subdir_path = os.path.join(scripts_dir, scripts_subdir)
    for script_file in sorted(os.listdir(subdir_path)):
        if (script_file.find("deep") >= 0, scripts_subdir) not in deep_algo_combinations:
            continue
        if script_file.find("true") >= 0:
            continue
        full_script_path = os.path.join(scripts_dir, scripts_subdir, script_file)
        with open(full_script_path, 'r') as f:
            script = f.read()
        _, program_name, *arg_list = shlex.split(script)
        if program_name != "train_rl.py":
            continue
        arg_list = [arg for arg in arg_list if arg != '\n']
        args = parser.parse_args(arg_list)

        num_scripts += 1
        print(f"{Fore.BLUE}{num_scripts}. {Fore.RESET}{full_script_path}")

        if results_dict.get(full_script_path, {}).get("status") == "COMPLETED":
            print(Fore.GREEN + "COMPLETED" + Fore.RESET)
            continue

        args.test_greedy = not args.test_no_greedy
        args.env_config = eval(args.env_config)
        args.ids_to_run = list(eval(args.ids_to_run)) if args.ids_to_run is not None else None
        if args.abs_base_action_weights is not None:
            args.abs_base_action_weights = eval(args.abs_base_action_weights)
        has_training_end_condition = (args.n_episodes is not None
                or args.early_stop_reward is not None or args.early_stop_td is not None or args.early_stop_q_rel_err is not None)
        assert has_training_end_condition, "Must specify condition for ending training"
        base_env = RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config))
        env_dir_string = args.env + (f"_{base_env.config_string}" if base_env.config_string is not None else "")
        expt_dir = os.path.join(output_path,
                                env_dir_string
                                + f"_{args.rl_algo}"
                                + ("_no-expl" if args.no_explore else "")
                                + ("_deep" if args.deep else ""),
                                args.expt_name)
        envinfo_expt_name = "few_abs_extra"
        if args.seed != 0:
            envinfo_expt_name += f"_s{args.seed}"
        envinfo_expt_dir = os.path.join(envinfo_output_path, env_dir_string, envinfo_expt_name)

        if not os.path.exists(expt_dir):
            print(Fore.RED + "NOT STARTED" + Fore.RESET)
            status = "NOT STARTED"
        else:
            not_done_ids = []
            for run_id in range(1 if args.love else 32):
                if args.no_explore:
                    if not (os.path.exists(os.path.join(expt_dir, f"run{run_id}_Q_values.npy"))
                            or os.path.exists(os.path.join(expt_dir, f"run{run_id}_V_values.npy"))):
                        not_done_ids.append(run_id)
                else:
                    history_file = os.path.join(expt_dir, f"run{run_id}_train_history.pkl")
                    if not os.path.exists(history_file):
                        not_done_ids.append(run_id)
                    else:
                        with open(history_file, 'rb') as f:
                            history = pkl.load(f)
                        if not history[-1].get("finished"):
                            not_done_ids.append(run_id)
            if not_done_ids:
                if args.love:
                    assert len(not_done_ids) == 1
                    print(Fore.YELLOW + "UNFINISHED" + Fore.RESET)
                else:
                    print(Fore.YELLOW + "UNFINISHED IDS", sorted(not_done_ids), Fore.RESET)
                status = "UNFINISHED"
            else:
                print(Fore.GREEN + "COMPLETED" + Fore.RESET)
                status = "COMPLETED"

        results_dict[full_script_path] = {
                "expt_dir": expt_dir,
                "envinfo_expt_dir": envinfo_expt_dir,
                "status": status,
                "num_runs": 1 if args.love else 32,
            }

with open(results_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
