# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import os
import pathlib
import warnings
from collections.abc import Callable
from datetime import timedelta
from string import Template
from typing import Any, TypeVar

import git
import numpy as np
import ot
import torch
from tensordict import TensorDict

T = TypeVar("T", TensorDict, torch.Tensor)


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolves the activation function from the name.

    Args:
        act_name: The name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolves the optimizer from the name.

    Args:
        optimizer_name: The name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype_name = dtype_name.lower()
    if dtype_name in dtype_dict:
        return dtype_dict[dtype_name]
    else:
        raise ValueError(f"Invalid dtype '{dtype_name}'. Valid optimizers are: {list(dtype_dict.keys())}")


def split_and_pad_trajectories(tensor: T, dones: torch.Tensor) -> tuple[T, torch.Tensor]:
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length of the longest
    trajectory. Returns masks corresponding to valid parts of the trajectories.

    Example:
        Input: [[a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """

    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # add at least one full length trajectory
            trajectories = trajectories + (torch.zeros(v.shape[0], *v.shape[2:], device=v.device),)
            # pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)
            # remove the added tensor
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)]
        )
    else:
        # split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # add at least one full length trajectory
        trajectories = trajectories + (torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),)
        # pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
        # remove the added tensor
        padded_trajectories = padded_trajectories[:, :-1]
    # create masks for the valid parts of the trajectories
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name: The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validates the observation configuration and defaults missing observation sets.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like:
        {
            "policy": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"]
        }

    This means that the 'policy' observation set will contain the observations "group_1" and "group_2" and the
    'critic' observation set will contain the observations "group_1" and "group_3". This function will check that all
    the observations in the 'policy' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Additionally, if one of the `default_sets`, e.g. "critic", is not present in the configuration dictionary,
    this function will:

    1. Check if a group with the same name exists in the observations and assign this group to the observation set.
    2. If 1. fails, it will assign the observations from the 'policy' observation set to the default observation set.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Observation sets configuration.
        default_sets: Reserved observation set names used by the algorithm (besides 'policy').
            If not provided in 'obs_groups', a default behavior gets triggered.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
    """
    # check if policy observation set exists
    if "policy" not in obs_groups.keys():
        if "policy" in obs:
            obs_groups["policy"] = ["policy"]
            warnings.warn(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                " As an observation group with the name 'policy' was found, this is assumed to be the observation set."
                " Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity."
                " This behavior will be removed in a future version."
            )
        else:
            raise ValueError(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                f" Found keys: {list(obs_groups.keys())}"
            )

    # check all observation sets for valid observation groups
    for set_name, groups in obs_groups.items():
        # check if the list is empty
        if len(groups) == 0:
            msg = f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list."
            if set_name in default_sets:
                if set_name not in obs:
                    msg += " Consider removing the key to default to the observations used for the 'policy' set."
                else:
                    msg += (
                        f" Consider removing the key to default to the observation '{set_name}' from the environment."
                    )
            raise ValueError(msg)
        # check groups exist inside the observations from the environment
        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                    f" environment. Available observations from the environment: {list(obs.keys())}"
                )

    # fill missing observation sets
    for default_set_name in default_sets:
        if default_set_name not in obs_groups.keys():
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As an observation group with the name '{default_set_name}' was found, this is assumed to be the"
                    f" observation set. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )
            else:
                obs_groups[default_set_name] = obs_groups["policy"].copy()
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As the configuration for '{default_set_name}' is missing, the observations from the 'policy' set"
                    f" are used. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )

    # print the final parsed observation sets
    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups


def get_obs_dimensions(obs: TensorDict, obs_groups: list[str]) -> int:
    num_obs: int = 0
    for obs_group in obs_groups:
        assert len(obs[obs_group].shape) == 2, "Only 1D observations are supported."
        num_obs += obs[obs_group].shape[-1]
    return num_obs


def dtype_numpytotorch(np_dtype: Any) -> torch.dtype:
    if isinstance(np_dtype, torch.dtype):
        return np_dtype
    if np_dtype == np.float16:
        return torch.float16
    elif np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == bool:  # noqa: E721
        return torch.bool
    elif np_dtype == np.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unknown type {np_dtype}")


@torch.jit.script
def compute_td_targets(data: torch.Tensor, lam: float) -> torch.Tensor:
    """Computes TD targets, i.e.

    .. math::

        mean(x) - \\frac{lam}{n^2 - n} \\sum_{i, j} |x_i - x_j|
    """
    N = data.shape[0]
    mean = data.mean(dim=0)
    uncertainty = torch.abs(data.unsqueeze(dim=0) - data.unsqueeze(dim=1)).sum(dim=(0, 1)) / (N**2 - N)
    return mean - lam * uncertainty


def reset_parameters(m: torch.nn.Module) -> None:
    if hasattr(m, "init_weights"):
        m.init_weights()  # type: ignore


def compute_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = torch.sum(x**2, dim=-1, keepdim=True)
    y_norm = torch.sum(y**2, dim=-1, keepdim=True).transpose(-2, -1)
    mat = x_norm + y_norm - 2 * (x @ y.transpose(-2, -1))
    # ensure no negative values from numerical imprecision
    return mat.clamp_(min=0.0)


def compute_emd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2 and len(y.shape) == 2, "Only 2D tensors are supported for EMD computation."
    cost_matrix = compute_distance_matrix(x, y).detach()
    x_pot = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    y_pot = torch.ones(y.shape[0], device=y.device) / y.shape[0]
    return ot.emd2(x_pot, y_pot, cost_matrix, numItermax=100000)


def pad_to_size(x: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    assert x.shape[dim] <= size
    shape = list(x.shape)
    shape[dim] = size - shape[dim]
    return torch.cat([x, torch.zeros(shape, device=x.device)], dim=dim)


def get_obs(obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
    obs_list = []
    for obs_group in obs_groups:
        obs_list.append(obs[obs_group])
    return torch.cat(obs_list, dim=-1)


def forward_sliding_mean(x: torch.Tensor, window_len: int, dim: int = 0) -> torch.Tensor:
    # Move target dim to position 0 for simplicity
    perm = [i for i in range(x.dim()) if i != dim]
    perm.insert(1, dim)
    x = x.permute(perm)

    cumsum = torch.cumsum(x, dim=0)
    pad = torch.zeros(1, *cumsum.shape[1:], device=cumsum.device)
    cumsum = torch.cat([pad, cumsum], dim=0)

    L = x.shape[0]
    start_idx = torch.arange(L, device=x.device)
    end_idx = torch.clamp(start_idx + window_len, max=L)
    lengths = (end_idx - start_idx).view(L, *([1] * (x.dim() - 1)))

    mean = (cumsum[end_idx] - cumsum[start_idx]) / lengths
    inv_perm = [perm.index(i) for i in range(len(perm))]
    return mean.permute(inv_perm)


class eval_mode:
    def __init__(self, *models: torch.nn.Module) -> None:
        self.models = models
        self.prev_states: list[bool] = []

    def __enter__(self) -> None:
        self.prev_states.clear()
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


class _TimeDeltaTemplate(Template):
    delimiter = "%"


def format_date(fmt: str, time_s: float) -> str:
    tdelta = timedelta(seconds=time_s)
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    # Zero-pad H, M, S
    d = {k: f"{v:02}" if k in ["H", "M", "S"] else str(v) for k, v in d.items()}
    t = _TimeDeltaTemplate(fmt)
    return t.substitute(**d)


def soft_update_params(
    params: tuple[torch.Tensor, ...] | None,
    target_params: tuple[torch.Tensor, ...] | None,
    tau: float,
) -> None:
    assert params is not None and target_params is not None
    torch._foreach_mul_(target_params, tau)  # type: ignore
    torch._foreach_add_(target_params, params, alpha=(1 - tau))  # type: ignore
