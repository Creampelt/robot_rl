# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for W&B."""

    def __init__(self, log_dir: str, flush_secs: int, num_envs: int, cfg: dict) -> None:
        """Initialize a W&B run for logging."""
        super().__init__(log_dir, flush_secs=flush_secs)

        # Get the run name and group
        run_name = cfg.get("wandb_run_name") or os.path.split(log_dir)[-1]
        group = cfg.get("wandb_group") or None

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        self.shared = cfg.get("shared", False)
        self.num_envs = num_envs

        settings = wandb.Settings(start_method="thread")
        tags = []
        if self.shared:
            settings.x_label = "main"
            settings.mode = "shared"
            settings.x_primary = True
            tags.append("log_videos_async")

        # Initialize wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config={"log_dir": log_dir},
            settings=settings,
            tags=tags,
        )

        # Define custom metrics
        self.run.define_metric("*", step_metric="local_step")  # global step (custom defined for async video logging)
        self.run.define_metric("*", step_metric="env_step")  # env step (step * num_envs)

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B."""
        self.run.config.update({"train_cfg": train_cfg})
        try:
            self.run.config.update({"env_cfg": env_cfg.to_dict()})  # type: ignore
        except Exception:
            self.run.config.update({"env_cfg": asdict(env_cfg)})  # type: ignore

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.run.log(
            {tag: scalar_value, "local_step": global_step, "env_step": global_step * self.num_envs},
            step=global_step if not self.shared else None,
        )

    def stop(self) -> None:
        """Finish the active W&B run."""
        self.run.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        self.run.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        self.run.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            self.run.log(
                {"video": wandb.Video(str(video), format="mp4"), "local_step": it, "env_step": it * self.num_envs},
                step=it if not self.shared else None,
            )
            self.logged_videos.add(video.name)
