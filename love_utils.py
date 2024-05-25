from typing import Union, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import array_obs_float64_to_float32, batch_obs_numpy
from custom_types import NumpyArrayObs

# from yidingjiang/love/utils.py
def concat(*data_list):
    return torch.cat(data_list, 1)

# from yidingjiang/love/utils.py
def gumbel_sampling(log_alpha, temp, margin=1e-4):
    noise = log_alpha.new_empty(log_alpha.size()).uniform_(margin, 1 - margin)
    gumbel_sample = -torch.log(-torch.log(noise))
    return torch.div(log_alpha + gumbel_sample, temp)

# from yidingjiang/love/utils.py
def log_density_concrete(log_alpha, log_sample, temp):
    exp_term = log_alpha - temp * log_sample
    log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
    return log_prob

# based on `compile_loader()` of yidingjiang/love/utils.py
def get_traj_data_loader(trajectories, batch_size, shuffle_train=True, float64_to_float32=False):
    # TODO add padding before and after each trajectory
    train_dataset = TrajectoryDataset(trajectories, partition="train", float64_to_float32=float64_to_float32)
    test_dataset = TrajectoryDataset(trajectories, partition="test", float64_to_float32=float64_to_float32)
    # TODO collate_batch to pad uneven sequence lengths (output additional mask tensor, and allow array obs to be tuple)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=shuffle_train, drop_last=False, collate_fn=traj_collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset),
                             shuffle=False, collate_fn=traj_collate_fn)
    return train_loader, test_loader

def traj_collate_fn(trajectories):
    batch_size = len(trajectories)
    obs_list = []
    action_list = []
    for obses, actions in trajectories:
        obs_list.append(tuple(map(torch.tensor, obses)) if isinstance(obses, tuple) else torch.tensor(obses))
        action_list.append(torch.tensor(actions))
    if isinstance(obs_list[0], tuple):
        obs_tensor = []
        for obs_list_part in zip(*obs_list):
            obs_tensor_part = pad_sequence(obs_list_part, batch_first=True)
            dummy_obs_part = torch.zeros((batch_size, 1, *obs_tensor_part.shape[2:]), dtype=obs_tensor_part.dtype)
            obs_tensor_part = torch.cat([dummy_obs_part, obs_tensor_part, dummy_obs_part], dim=1)
            obs_tensor.append(obs_tensor_part)
        obs_tensor = tuple(obs_tensor)
    else:
        obs_tensor = pad_sequence(obs_list, batch_first=True)
        dummy_obs = torch.zeros((batch_size, 1, *obs_tensor.shape[2:]), dtype=obs_tensor.dtype)
        obs_tensor = torch.cat([dummy_obs, obs_tensor, dummy_obs], dim=1)
    action_tensor = pad_sequence(action_list, batch_first=True)
    dummy_action = torch.zeros((batch_size, 1), dtype=torch.int32)
    action_tensor = torch.cat([dummy_action, action_tensor, dummy_action], dim=1)
    mask_tensor = pad_sequence([torch.ones_like(actions, dtype=torch.bool) for actions in action_list], batch_first=True)
    # dummy_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
    # mask_tensor = torch.cat([dummy_mask, mask_tensor, dummy_mask], dim=1)
    return obs_tensor, action_tensor, mask_tensor


# based on `ComPILEDataset` of yidingjiang/love/utils.py
class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: list[list[tuple[NumpyArrayObs, int, NumpyArrayObs]]],
        partition: Union[Literal["train"], Literal["test"]],
        float64_to_float32: bool=False
    ):
        self.partition = partition
        self.float64_to_float32 = float64_to_float32
        num_heldout = 100

        if self.partition == "train":
            self.state: list[list[tuple[NumpyArrayObs, int, NumpyArrayObs]]] = trajectories[:-num_heldout]  
        else:
            self.state: list[list[tuple[NumpyArrayObs, int, NumpyArrayObs]]] = trajectories[-num_heldout:]
        # note: self.state is N x L x (s, a, s_tp1)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index: int):
        traj = self.state[index]
        # s, a, _ = zip(*traj)
        s, a = zip(*traj)
        if self.float64_to_float32:
            s = list(map(lambda obs: array_obs_float64_to_float32(obs), s))
        return batch_obs_numpy(s), np.array(a, dtype=np.int64)
        # s, a, sp, z, m = zip(*traj)
        # s = np.stack(s).astype(np.float32)
        # a = np.stack(a)
        # sp = np.stack(sp).astype(np.float32)
        # z = np.stack(z).astype(np.int64)
        # z_mask = z >= 0
        # m = np.stack(m).astype(np.int64)
        # return s, a, sp, z, z_mask, m


def log_train(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    train_obs_cost = results["obs_cost"].mean()
    train_kl_abs_cost = results["kl_abs_state"].mean()
    train_kl_obs_cost = results["kl_obs_state"].mean()
    train_kl_mask_cost = results["kl_mask"].mean()

    stats = {}
    stats["train/full_cost"] = (
        train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost
    )
    stats["train/obs_cost"] = train_obs_cost
    stats["train/kl_full_cost"] = (
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost
    )
    stats["train/kl_abs_cost"] = train_kl_abs_cost
    stats["train/kl_obs_cost"] = train_kl_obs_cost
    stats["train/kl_mask_cost"] = train_kl_mask_cost
    stats["train/q_ent"] = results["p_ent"].mean()
    stats["train/p_ent"] = results["q_ent"].mean()
    stats["train/read_ratio"] = results["mask_data"].sum(1).mean()
    stats["train/beta"] = results["beta"]

    log_str = (
        "[%08d] train=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f, beta: %3.3f, "
        "p_ent: %3.2f, q_ent: %3.2f"
    )
    log_data = [
        b_idx,
        -(train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost),
        train_obs_cost,
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
        train_kl_abs_cost,
        train_kl_obs_cost,
        train_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
        results["beta"],
        results["p_ent"].mean(),
        results["q_ent"].mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        stats["train/coding_length"] = coding_length
    if "coding_len_coeff" in results:
        coding_len_coeff = results["coding_len_coeff"]
        log_str += ", coding_len_coeff: %3.6f"
        log_data.append(coding_len_coeff)
        stats["train/coding_len_coeff"] = coding_len_coeff
    if "marginal" in results:
        for i, p in enumerate(results["marginal"]):
            stats["option_{}/option_{}".format("train", i)] = p
    if "grad_norm" in results:
        grad_norm = results["grad_norm"]
        log_str += ", grad_norm: %3.2f"
        log_data.append(grad_norm)
        stats["train/grad_norm"] = grad_norm
    if "vq_loss_list" in results:
        vq_loss = results["vq_loss_list"]
        log_str += ", vq_loss: %3.2f"
        log_data.append(vq_loss)
        stats["train/vq_loss"] = vq_loss
    if "precision" in results:
        log_str += ", precision: %3.2f"
        log_data.append(results["precision"])
        stats["train/precision"] = results["precision"]
        log_str += ", recall: %3.2f"
        log_data.append(results["recall"])
        stats["train/recall"] = results["recall"]
        log_str += ", f1: %3.2f"
        log_data.append(results["f1"])
        stats["train/f1"] = results["f1"]
    return stats, log_str, log_data


def log_test(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    test_obs_cost = results["obs_cost"].mean()
    test_kl_abs_cost = results["kl_abs_state"].mean()
    test_kl_obs_cost = results["kl_obs_state"].mean()
    test_kl_mask_cost = results["kl_mask"].mean()

    stats = {}
    stats["valid/full_cost"] = (
        test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost
    )
    stats["valid/obs_cost"] = test_obs_cost
    stats["valid/kl_full_cost"] = (
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost
    )
    stats["valid/kl_abs_cost"] = test_kl_abs_cost
    stats["valid/kl_obs_cost"] = test_kl_obs_cost
    stats["valid/kl_mask_cost"] = test_kl_mask_cost
    stats["valid/q_ent"] = results["p_ent"].mean()
    stats["valid/p_ent"] = results["q_ent"].mean()
    stats["valid/read_ratio"] = results["mask_data"].sum(1).mean()
    stats["valid/beta"] = results["beta"]

    log_str = (
        "[%08d] valid=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f"
    )
    log_data = [
        b_idx,
        -(test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost),
        test_obs_cost,
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
        test_kl_abs_cost,
        test_kl_obs_cost,
        test_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        # writer.add_scalar('valid/coding_length', coding_length, global_step=b_idx)
        stats["valid/coding_length"] = coding_length
    if "marginal" in results:
        for i, p in enumerate(results["marginal"]):
            stats["option_{}/option_{}".format("valid", i)] = p
    if "precision" in results:
        log_str += ", precision: %3.2f"
        log_data.append(results["precision"])
        stats["train/precision"] = results["precision"]
        log_str += ", recall: %3.2f"
        log_data.append(results["recall"])
        stats["train/recall"] = results["recall"]
        log_str += ", f1: %3.2f"
        log_data.append(results["f1"])
        stats["train/f1"] = results["f1"]
    return stats, log_str, log_data


