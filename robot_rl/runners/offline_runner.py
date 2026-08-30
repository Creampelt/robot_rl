from __future__ import annotations

import os
import time
import torch
from typing import Any

from robot_rl.algorithms import Fb
from robot_rl.env import URLVecEnv
from robot_rl.utils import demote_old_checkpoint, resolve_callable
from robot_rl.utils.logger import Logger


class OfflineRlRunner:
    """Runner for algorithms that learn from a stored dataset instead of environment interaction.

    Structurally the off-policy runner without a rollout: no env stepping, no seed phase, and no
    per-iteration observation flow. The env is still constructed, because it supplies the observation
    shapes and action count the models are built from, and because evaluation replays in it.
    """

    def __init__(
        self,
        env: URLVecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        inference: bool = False,
    ) -> None:
        """Construct the runner, algorithm, and logging stack.

        ``inference=True`` skips loading the dataset, so play/visualization can build the policy without it.
        """
        self.cfg = train_cfg
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        obs = self.env.get_observations()
        alg_class: type[Fb] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device, inference=inference)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, **kwargs: Any) -> None:
        """Run the learning loop: per iteration, run agent updates on stored data, then log and save."""
        del kwargs
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        num_updates = self.cfg["algorithm"].get("num_agent_updates", 1)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        learn_time = 0.0
        eval_time = 0.0
        for it in range(start_it + 1, total_it + 1):
            start = time.time()
            loss_extras: list[dict] = []
            algo_extras: list[dict] = []
            for _ in range(num_updates):
                loss_dict, algo_dict = self.alg.update()
                loss_extras.append(loss_dict)
                if algo_dict:
                    algo_extras.append(algo_dict)
            self.logger.process_update_extras(eval_extras=[], loss_extras=loss_extras, algo_extras=algo_extras)
            learn_time += time.time() - start
            self.current_learning_iteration = it

            if it % self.cfg.get("log_interval", 1) == 0:
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=0.0,
                    learn_time=learn_time,
                    eval_time=eval_time,
                )
                learn_time = 0.0
                eval_time = 0.0

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore
                demoted = demote_old_checkpoint(
                    self.alg,
                    self.logger.log_dir,
                    it,
                    self.cfg.get("keep_full_checkpoints"),
                    self.cfg["save_interval"],
                )
                if demoted is not None:  # re-upload so the live sync replaces the full remote copy
                    self.logger.save_model(os.path.join(self.logger.log_dir, f"model_{demoted}.pt"), demoted)

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Write a checkpoint containing the algorithm state and the current iteration.

        Args:
            path: Destination file.
            infos: Optional extra payload stored alongside the algorithm state.
        """
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.logger.writer is not None:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True) -> dict | None:
        """Restore a checkpoint, continuing the iteration count when optimizer state came with it.

        Args:
            path: Checkpoint to read.
            load_cfg: Forwarded to the algorithm's loader.
            strict: Whether module state dicts must match exactly.

        Returns:
            The checkpoint's ``infos`` payload.
        """
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        resumable = self.alg.load(loaded_dict, load_cfg, strict)
        if resumable:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None):  # noqa: ANN201
        """Return the policy in eval mode, optionally moved to another device."""
        self.alg.eval_mode()
        policy = self.alg.get_policy()
        if device is not None:
            policy.to(device)
        return policy

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Record a repository's state alongside the run."""
        self.logger.add_git_repo_to_log(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Read the distributed environment and set this rank's device."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu_cfg"] = None
            return
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu_cfg"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        torch.cuda.set_device(self.gpu_local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
