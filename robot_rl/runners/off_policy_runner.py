from __future__ import annotations

import contextlib
import os
import time
import torch
from typing import Any

from robot_rl.algorithms import FbCpr
from robot_rl.env import URLVecEnv
from robot_rl.models import MLPModel
from robot_rl.utils import check_nan, resolve_callable
from robot_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for reinforcement learning algorithms."""

    alg: FbCpr
    """The actor-critic algorithm."""

    def __init__(self, env: URLVecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.cfg = train_cfg
        self.device = device
        self.env = env

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[FbCpr] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
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
        """Run the learning loop for the specified number of iterations."""
        # Add expert buffer to environment, then re-reset so the initial state is RSI'd from the expert buffer rather
        # than the default-pose state produced by the env wrapper's first reset (which ran before attach).
        self.env.set_expert_buffer(self.alg.expert_buffer)

        # Start learning
        obs, _ = self.env.reset()
        obs = obs.to(self.device)
        # Switch models and environment to train mode (for dropout, env events, etc.)
        self.alg.train_mode()
        self.env.train_mode()

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        eval_time = 0.0
        collect_time = 0.0
        learn_time = 0.0
        z: torch.Tensor | None = None
        last_dones: torch.Tensor | None = None

        with self._get_profile_context() as prof:
            for it in range(start_it, total_it):
                with torch.inference_mode(), torch.profiler.record_function("rollout"):
                    # Run evaluation
                    eval_extras = None
                    if (it - start_it) % self.cfg["eval_interval"] == 0:
                        # Save and clear all logging buffers (environments will reset after eval)
                        self.logger.reset_all_envs()
                        # Run evaluation
                        start = time.time()
                        with torch.profiler.record_function("eval"):
                            eval_extras = self.alg.eval(self.env)
                        stop = time.time()
                        eval_time += stop - start

                        # reset env and training variables
                        obs, _ = self.env.reset()
                        obs = obs.to(self.device)
                        last_dones = None
                        cur_episode_length[:] = 0

                    # Rollout
                    start = time.time()
                    is_seed = it <= self.cfg["num_seed_steps_per_env"] + start_it
                    # Update latent z
                    z = self.alg.update_rollout_z(z, cur_episode_length, self.env.num_envs)
                    # Sample actions
                    actions = self.alg.act(
                        obs, z, last_dones, random_sample=is_seed, clip_actions=self.cfg["clip_actions"]
                    )
                    # Step the environment
                    with torch.profiler.record_function("env_step"):
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Update episode length and last dones
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    cur_episode_length[new_ids] = 0
                    last_dones = dones

                    # Update timer
                    stop = time.time()
                    collect_time += stop - start
                    start = stop

                loss_extras: list[dict] = []
                algo_extras: list[dict] = []
                if not is_seed and it % self.cfg["num_steps_per_env"] == 0:
                    with torch.inference_mode():
                        self.alg.compute_gammas()

                    # Update policy
                    with torch.profiler.record_function("update"):
                        for _ in range(self.cfg["num_agent_updates"]):
                            loss_dict, algo_dict = self.alg.update()
                            loss_extras.append(loss_dict)
                            algo_extras.append(algo_dict)
                    # gc.collect()

                # Book keeping
                self.logger.process_env_step(
                    rewards, dones, extras, eval_extras=eval_extras, loss_extras=loss_extras, algo_extras=algo_extras
                )

                stop = time.time()
                learn_time += stop - start
                self.current_learning_iteration = it

                # Log information
                if it % self.cfg["log_interval"] == 0:
                    self.logger.log(
                        it=it,
                        start_it=start_it,
                        total_it=total_it,
                        collect_time=collect_time,
                        learn_time=learn_time,
                        eval_time=eval_time,
                    )
                    eval_time = 0.0
                    collect_time = 0.0
                    learn_time = 0.0

                # Save model
                if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

                if prof is not None:
                    prof.step()

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _get_profile_context(self) -> contextlib.AbstractContextManager[torch.profiler.profile | None]:
        """Build a profiler context manager.

        Returns a :class:`contextlib.nullcontext` (yielding ``None``) if ``cfg["profile"]`` is False, otherwise a
        :class:`torch.profiler.profile` configured from the ``profile_*`` cfg entries.
        """
        if not self.cfg.get("profile", False):
            return contextlib.nullcontext()
        profile_wait = self.cfg.get("profile_wait", 15)
        profile_warmup = self.cfg.get("profile_warmup", 2)
        profile_active = self.cfg.get("profile_active", 3)
        profile_dir = os.path.join(self.logger.log_dir, "profile") if self.logger.log_dir else "profile"
        logger = self.logger

        def _on_trace_ready(prof: torch.profiler.profile) -> None:
            os.makedirs(profile_dir, exist_ok=True)
            trace_path = os.path.join(profile_dir, "trace.json")
            summary_path = os.path.join(profile_dir, "summary.txt")
            prof.export_chrome_trace(trace_path)
            summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            with open(summary_path, "w") as f:
                f.write(summary)
            print(f"[PROFILE] Trace exported to {trace_path}")
            print(summary)
            writer = getattr(logger, "writer", None)
            if writer is not None and hasattr(writer, "save_file"):
                writer.save_file(trace_path)
                writer.save_file(summary_path)
                print("[PROFILE] Trace and summary uploaded to logger.")

        print(
            f"[PROFILE] Will record {profile_active} iters after {profile_wait} wait + {profile_warmup} warmup."
            f" Output dir: {profile_dir}"
        )
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=profile_wait, warmup=profile_warmup, active=profile_active, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=False,
            with_stack=False,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
            self.env.unwrapped.common_step_counter = self.current_learning_iteration * self.cfg["num_steps_per_env"]  # type: ignore
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
