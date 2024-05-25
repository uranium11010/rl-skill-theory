""" Based on yidingjiang/love/train_rl.py """

import argparse
import sys
import os
import shutil
import logging
import time
from datetime import datetime
import json
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb

import gymnasium as gym
from wrappers import RLDiffEnvAdapter

from love_utils import get_traj_data_loader, log_train, log_test
from love import EnvModel

class GridDecoder(nn.Module):
    """Decoder for actions from a latent vector."""
    def __init__(self, input_size, action_size, feat_size=64):
        super().__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(
                input_size=input_size, output_size=feat_size, nonlinear=nn.Identity()
            )
        self.network = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, action_size),
        )

    def forward(self, input_data):
        return self.network(self.linear(input_data))

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, nonlinear=nn.ELU(inplace=True)):
        super(LinearLayer, self).__init__()
        # linear
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.linear(input_data))

LOGGER = logging.getLogger(__name__)

def date_str():
    s = str(datetime.now())
    d, t = s.split(" ")
    t = "-".join(t.split(":")[:-1])
    return d + "-" + t

def main(args):
    # experiment directory
    exp_dir = os.path.join(args.output_path, args.name)
    if os.path.exists(exp_dir):
        if args.overwrite:
            print(f"Experiment directory {exp_dir} not empty! Removing contents...")
            shutil.rmtree(exp_dir, ignore_errors=True)  # remove directory if exists
            time.sleep(5)
        else:
            raise Exception(f"Experiment directory {exp_dir} not empty!")
    print(f"Starting experiment in directory {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    init_size = 1

    # set device as gpu
    device = torch.device("cuda", 0)

    # set writer
    exp_name = args.name + "_" + date_str()

    wandb.init(
        project="love_CompILE_hssm",
        entity="conpole2",
        name=exp_name,
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )

    LOGGER.info("EXP NAME: " + exp_name)
    LOGGER.info(">" * 80)
    LOGGER.info(args)
    LOGGER.info(">" * 80)

    # load dataset
    with open(args.traj_path, 'rb') as f:
        trajectories = pkl.load(f)
    train_loader, test_loader = get_traj_data_loader(
            trajectories, args.batch_size, shuffle_train=not args.no_shuffle, float64_to_float32=True)
    env = RLDiffEnvAdapter(gym.make(f"{args.env}-rldiff", **args.env_config))
    encoder = env.get_state_embedder()
    action_encoder = ActionEncoder(
        action_size=env.action_space.n,
        embedding_size=encoder.embed_dim,
    )
    decoder = GridDecoder(
        input_size=encoder.embed_dim,
        action_size=env.action_space.n,
        feat_size=encoder.embed_dim,
    )
    output_normal = True

    # init models
    with open(args.model_config_path, 'r') as f:
        model_config = json.load(f)
    model = EnvModel(
        action_encoder=action_encoder,
        encoder=encoder,
        decoder=decoder,
        belief_size=encoder.embed_dim,
        state_size=model_config["state_size"],
        num_layers=model_config["num_layers"],
        max_seg_len=model_config["max_seg_len"],
        max_seg_num=model_config["max_seg_num"],
        latent_n=model_config["latent_n"],
        kl_coeff=model_config["kl_coeff"],
        rec_coeff=model_config["rec_coeff"],
        coding_len_coeff=model_config["coding_len_coeff"],
        use_min_length_boundary_mask=model_config["use_min_length_boundary_mask"],
        ddo=model_config["ddo"],
        output_normal=output_normal
    ).to(device)
    LOGGER.info("Model initialized")

    # init optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

    # test data
    pre_test_full_state_list, pre_test_full_action_list, pre_test_full_mask_list = next(iter(test_loader))
    if isinstance(pre_test_full_state_list, tuple):
        pre_test_full_state_list = tuple(map(lambda t: t.to(device), pre_test_full_state_list))
    else:
        pre_test_full_state_list = pre_test_full_state_list.to(device)
    pre_test_full_action_list = pre_test_full_action_list.to(device)
    pre_test_full_mask_list = pre_test_full_mask_list.to(device)

    # for each iter
    torch.autograd.set_detect_anomaly(False)
    b_idx = 0
    while b_idx <= args.max_iters:
        # for each batch
        for train_obs_list, train_action_list, train_mask_list in train_loader:
            b_idx += 1
            # mask temp annealing
            if args.beta_anneal:
                model.state_model.mask_beta = (
                    args.max_beta - args.min_beta
                ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
            else:
                model.state_model.mask_beta = args.max_beta

            ##############
            # train time #
            ##############
            if isinstance(pre_test_full_state_list, tuple):
                train_obs_list = tuple(map(lambda t: t.to(device), train_obs_list))
            else:
                train_obs_list = train_obs_list.to(device)
            train_action_list = train_action_list.to(device)
            train_mask_list = train_mask_list.to(device)

            # run model with train mode
            model.train()
            optimizer.zero_grad()
            results = model(
                train_obs_list, train_action_list, train_mask_list, init_size, args.obs_std
            )

            if args.coding_len_coeff > 0:
                if results["obs_cost"].mean() < 0.02:
                    model.coding_len_coeff += 0.00002
                elif b_idx > 0:
                    model.coding_len_coeff -= 0.00002

                model.coding_len_coeff = min(0.05, model.coding_len_coeff)
                model.coding_len_coeff = max(0.000000, model.coding_len_coeff)
                results["coding_len_coeff"] = model.coding_len_coeff

            # get train loss and backward update
            train_total_loss = results["train_loss"]
            train_total_loss.backward()
            if args.grad_clip > 0.0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()

            # log
            if b_idx % 5 == 0:
                results["grad_norm"] = grad_norm
                train_stats, log_str, log_data = log_train(results, None, b_idx)
                # Boundaries for grid world

                LOGGER.info(log_str, *log_data)
                wandb.log(train_stats, step=b_idx)

            np.set_printoptions(threshold=100000)
            torch.set_printoptions(threshold=100000)
            # if b_idx % 200 == 0:
            #     for batch_idx in range(min(train_obs_list.shape[0], 10)):  # TODO accomodate CompILE2 obs
            #         states = train_obs_list[batch_idx][init_size:-init_size]
            #         actions = train_action_list[batch_idx][init_size:-init_size]
            #         reconstructed_actions = torch.argmax(results["rec_data"], -1)[
            #             batch_idx
            #         ]
            #         options = results["option_list"][batch_idx]
            #         boundaries = results["mask_data"][batch_idx]
            #         frames = []
            #         curr_option = options[0]

            #         for seq_idx in range(states.shape[0]):
            #             # read new option if boundary is 1
            #             if boundaries[seq_idx].item() == 1:
            #                 curr_option = options[seq_idx]

            #             # TODO write what the actions and options look like?
            #             # frame.write_text(f"Option: {curr_option}")
            #             # frame.write_text(f"Boundary: {boundaries[seq_idx].item()}")
            #             # frame.write_text(f"Obs NLL: {results['obs_cost'].mean()}")
            #             # frame.write_text(
            #             #     f"Coding length: {results['encoding_length'].item()}"
            #             # )
            #             # frame.write_text(
            #             #     f"Num reads: {results['mask_data'].sum(1).mean().item()}"
            #             # )
            #             # frames.append(frame.image())

            #         # save_path = os.path.join(exp_dir, f"{batch_idx}.gif")
            #         # frames[0].save(
            #         #     save_path,
            #         #     save_all=True,
            #         #     append_images=frames[1:],
            #         #     duration=750,
            #         #     loop=0,
            #         #     optimize=True,
            #         #     quality=20,
            #         # )

            if b_idx % 100 == 0:
                LOGGER.info("#" * 80)
                LOGGER.info(">>> option list")
                LOGGER.info("\n" + repr(results["option_list"][:10]))
                LOGGER.info(">>> boundary mask list")
                LOGGER.info("\n" + repr(results["mask_data"][:10].squeeze(-1)))
                LOGGER.info(">>> train_action_list")
                LOGGER.info("\n" + repr(train_action_list[:10]))
                LOGGER.info(">>> argmax reconstruction")
                LOGGER.info("\n" + repr(torch.argmax(results["rec_data"], -1)[:10]))
                LOGGER.info(">>> diff")
                LOGGER.info(
                    "\n"
                    + repr(
                        train_action_list[:10, 1:-1]
                        - torch.argmax(results["rec_data"][:10], -1)
                    )
                )
                LOGGER.info(">>> marginal")
                LOGGER.info("\n" + repr(results["marginal"]))
                LOGGER.info("#" * 80)

            if b_idx % 2000 == 0:
                torch.save(
                    model.state_model.state_dict(), os.path.join(exp_dir, f"model-{b_idx}.ckpt")
                )

            #############
            # test time #
            #############
            if b_idx % 100 == 0:
                with torch.no_grad():
                    ##################
                    # test data elbo #
                    ##################
                    model.eval()
                    results = model(
                        pre_test_full_state_list,
                        pre_test_full_action_list,
                        pre_test_full_mask_list,
                        init_size,
                        args.obs_std,
                    )

                    # log
                    test_stats, log_str, log_data = log_test(results, None, b_idx)
                    # Actions for grid world
                    reconstructed_actions = torch.argmax(results["rec_data"], -1)
                    true_actions = pre_test_full_action_list[:, init_size:-init_size]
                    test_stats["valid/actions_acc"] = (reconstructed_actions == true_actions).to(float).mean()
                    LOGGER.info(log_str, *log_data)
                    wandb.log(test_stats, step=b_idx)


class ActionEncoder(nn.Module):
    """Embedder for discrete actions."""

    def __init__(self, action_size, embedding_size):
        super().__init__()
        self.action_size = action_size
        self.embedding_size = embedding_size
        self._embedder = nn.Embedding(action_size, embedding_size)

    def forward(self, x):
        return self._embedder(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["CliffWalking-v0", "RubiksCube222-v0", "CompILE-v0", "NPuzzle-v0"], required=True)
    parser.add_argument("--env_config", default="{}", help="Dictionary as string")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="st")
    parser.add_argument("--log_examples", action="store_true")
    parser.add_argument("--print_examples", type=int, default=0)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--overwrite", "-f", action="store_true")

    # model config
    parser.add_argument("--model_config_path", required=True)

    # data size
    parser.add_argument("--batch_size", type=int, default=128)

    # observation distribution
    parser.add_argument("--obs_std", type=float, default=1.0)
    parser.add_argument("--obs_bit", type=int, default=5)

    # optimization
    parser.add_argument("--learn_rate", type=float, default=0.0005)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--max_iters", type=int, default=100000)

    # gumbel params
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--min_beta", type=float, default=0.1)
    parser.add_argument("--beta_anneal", type=float, default=100)

    # log dir
    parser.add_argument("--log_dir", type=str, default="./asset/log/")

    # coding length params
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--rec_coeff", type=float, default=1.0)
    parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    parser.add_argument("--use_min_length_boundary_mask", action="store_true")

    # baselines
    parser.add_argument("--ddo", action="store_true")

    args = parser.parse_args()
    args.env_config = eval(args.env_config)

    main(args)
