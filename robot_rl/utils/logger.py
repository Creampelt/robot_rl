# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import git
import os
import pathlib
import statistics
import time
import torch
from collections import deque

import robot_rl
from robot_rl.utils import format_date


class Logger:
    """Logger to save the learning metrics to different logging services."""

    def __init__(
        self,
        log_dir: str | None,
        cfg: dict,
        env_cfg: dict | object,
        num_envs: int,
        is_distributed: bool,
        gpu_world_size: int,
        gpu_global_rank: int,
        device: str,
    ) -> None:
        """Initialize buffers and logging state for a training run."""
        self.log_dir = log_dir
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.gpu_world_size = gpu_world_size
        self.device = device
        self.git_status_repos = [robot_rl.__file__]
        self.tot_timesteps = 0
        self.tot_time = 0

        # Create buffers
        self.ep_extras: list[dict] = []
        self.eval_extras: list[dict] = []
        self.loss_extras: list[dict] = []
        self.algo_extras: list[dict] = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Create RND buffers
        if self.cfg["algorithm"].get("rnd_cfg", None):
            self.erewbuffer = deque(maxlen=100)
            self.irewbuffer = deque(maxlen=100)
            self.cur_ereward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cur_ireward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = is_distributed and gpu_global_rank != 0

    def init_logging_writer(self) -> None:
        """Initialize the logging writer, which can be either Tensorboard, W&B or Neptune and save the code state.

        If the writer is either W&B or Neptune, the configuration and code state are uploaded as well.
        """
        if self.log_dir is not None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "neptune":
                from robot_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "wandb":
                from robot_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, num_envs=self.num_envs, cfg=self.cfg
                )
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'wandb', 'neptune', or 'tensorboard'.")
        else:
            self.writer = None

        # Save code state
        files_to_upload = self._store_code_state()

        # Upload configuration and code state to external logging service if applicable
        if self.writer is not None and self.logger_type in ["wandb", "neptune"]:
            self.writer.store_config(self.env_cfg, self.cfg)  # type: ignore
            for path in files_to_upload:
                self.writer.save_file(path)  # type: ignore

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
        eval_extras: list[dict] | None = None,
        loss_extras: list[dict] | None = None,
        algo_extras: list[dict] | None = None,
    ) -> None:
        """Add metrics from the environment step to the buffers."""
        if self.writer is not None:
            # Store episode extras
            if "episode" in extras:
                self.ep_extras.append(extras["episode"])
            elif "log" in extras:
                self.ep_extras.append(extras["log"])
            # Store evaluation extras
            if eval_extras:
                self.eval_extras.extend(eval_extras)
            # Store loss metrics
            if loss_extras:
                self.loss_extras.extend(loss_extras)
            # Store algo extras
            if algo_extras:
                self.algo_extras.extend(algo_extras)

            # Update rewards and episode length
            if intrinsic_rewards is not None:
                self.cur_ereward_sum += rewards
                self.cur_ireward_sum += intrinsic_rewards
                self.cur_reward_sum += rewards + intrinsic_rewards
            else:
                self.cur_reward_sum += rewards
            self.cur_episode_length += 1

            # Clear data for completed episodes
            new_ids = (dones > 0).nonzero(as_tuple=False)
            self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
            if intrinsic_rewards is not None:
                self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self.cur_ereward_sum[new_ids] = 0
                self.cur_ireward_sum[new_ids] = 0

    def reset_all_envs(self) -> None:
        """Save and clear all logging buffers."""
        # Skip reset if all episodes are at 0 steps
        if self.cur_episode_length.any():
            self.rewbuffer.extend(self.cur_reward_sum.cpu().numpy().tolist())
            self.lenbuffer.extend(self.cur_episode_length.cpu().numpy().tolist())
            self.cur_reward_sum[:] = 0
            self.cur_episode_length[:] = 0
            if self.cfg["algorithm"].get("rnd_cfg", None):
                self.erewbuffer.extend(self.cur_ereward_sum.cpu().numpy().tolist())
                self.irewbuffer.extend(self.cur_ireward_sum.cpu().numpy().tolist())
                self.cur_ereward_sum[:] = 0
                self.cur_ireward_sum[:] = 0

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict | None = None,
        learning_rate: float | None = None,
        action_std: torch.Tensor | None = None,
        rnd_weight: float | None = None,
        eval_time: float | None = None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log the training metrics to the logging service and print them to the console.

        If videos are available, they are uploaded to the logging service (W&B) as well.
        """
        if self.writer is not None:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            if eval_time:
                iteration_time += eval_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            # Log episode extras
            extras_string = self._log_extras(self.ep_extras, it, pad, "Episode", "Mean episode {}")
            extras_string += self._log_extras(self.algo_extras, it, pad, "Train", "Train/{}")
            eval_string = self._log_extras(self.eval_extras, it, pad, "Evaluation", "Mean evaluation {}")
            # Log losses
            if loss_dict:
                # Use loss dict directly for on-policy (one update per log)
                loss_string = ""
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"Loss/{key}", value, it)
                    loss_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
                if learning_rate is not None:
                    self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
            else:
                # Use loss extras buffer for off-policy (multiple updates per log)
                loss_string = self._log_extras(
                    self.loss_extras, it, pad, "Loss", "Mean {}", override_console_format=False
                )

            # Log std
            if action_std is not None:
                self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

            # Log performance
            fps = int(collection_size / iteration_time)
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)
            if eval_time:
                self.writer.add_scalar("Perf/eval_time", eval_time, it)

            # Log rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"].get("rnd_cfg", None):
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                if self.logger_type != "wandb":
                    self.writer.add_scalar(
                        "Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time)
                    )
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            # Print to console
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            # Print run name if provided
            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            # Print performance
            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )
            if eval_time:
                log_string += f"""{"Evaluation time:":>{pad}} {eval_time:.3f}s \n"""

            # Print losses
            log_string += loss_string

            # Print rewards and episode length
            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"].get("rnd_cfg", None):
                    log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            # Print std
            if action_std is not None:
                log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

            # Print extras
            log_string += eval_string
            if not print_minimal:
                log_string += extras_string

            # Print footer
            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {format_date("%D days %H:%M:%S", eta)}\n"""
            )
            print(log_string)

            # Upload available videos
            if self.logger_type == "wandb":
                for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                    self.writer.save_video(video, it)  # type: ignore

            # Clear extras buffers
            self.ep_extras.clear()
            self.eval_extras.clear()
            self.loss_extras.clear()
            self.algo_extras.clear()

    def save_model(self, path: str, it: int) -> None:
        """Save the model to external logging services if specified."""
        if self.writer is not None and self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, it)  # type: ignore

    def stop_logging_writer(self) -> None:
        """Stop the logging writer."""
        if self.writer is not None and self.logger_type in ["neptune", "wandb"]:
            self.writer.stop()  # type: ignore

    def _store_code_state(self) -> list[str]:
        """Store the current git diff of the code repositories involved in the experiment."""
        files_to_upload = []
        if self.log_dir is not None and not self.disable_logs:
            git_log_dir = os.path.join(self.log_dir, "git")
            os.makedirs(git_log_dir, exist_ok=True)
            # Iterate over all repositories to log
            for repository_file_path in self.git_status_repos:
                try:
                    repo = git.Repo(repository_file_path, search_parent_directories=True)
                    t = repo.head.commit.tree
                    commit_hash = repo.head.commit.hexsha
                except Exception:
                    print(f"Could not find git repository in {repository_file_path}. Skipping.")
                    continue
                # Get the name of the repository
                repo_name = pathlib.Path(repo.working_dir).name
                diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
                # Check if the diff file already exists
                if os.path.isfile(diff_file_name):
                    continue
                # Write the diff file
                print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
                with open(diff_file_name, "x", encoding="utf-8") as f:
                    content = (
                        f"--- git commit ---\n{commit_hash}\n\n\n"
                        f"--- git status ---\n{repo.git.status()} \n\n\n"
                        f"--- git diff ---\n{repo.git.diff(t)}"
                    )
                    f.write(content)
                # Add the file path to the list of files to be uploaded
                files_to_upload.append(diff_file_name)
        return files_to_upload

    def _log_extras(
        self,
        extras: list,
        it: int,
        pad: int,
        scalar_prefix: str,
        console_format: str,
        override_console_format: bool = True,
    ) -> str:
        extras_string = ""
        if extras:
            # Iterate over all keys in the episode info dictionary
            for key in extras[0]:
                infotensor = torch.tensor([], device=self.device)
                # Iterate over all steps
                for info in extras:
                    # Handle missing, scalar, and zero dimensional tensors
                    if key not in info:
                        continue
                    if not isinstance(info[key], torch.Tensor):
                        info[key] = torch.Tensor([info[key]])
                    if len(info[key].shape) == 0:
                        info[key] = info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, info[key].to(self.device)))
                value = torch.nan_to_num(torch.mean(infotensor), nan=0.0)
                if "/" in key:
                    self.writer.add_scalar(key, value, it)  # type: ignore
                    if override_console_format:
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        extras_string += f"""{f"{console_format.format(key)}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar(f"{scalar_prefix}/" + key, value, it)  # type: ignore
                    extras_string += f"""{f"{console_format.format(key)}:":>{pad}} {value:.4f}\n"""
        return extras_string
