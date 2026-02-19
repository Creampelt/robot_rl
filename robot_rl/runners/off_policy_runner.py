from __future__ import annotations
from typing import Any

import os
import gc
import statistics
import time
import torch
from collections import deque
from tensordict import TensorDict

import robot_rl
from robot_rl.algorithms import FbCpr
from robot_rl.env import URLVecEnv
from robot_rl.modules import ForwardBackward
from robot_rl.utils import resolve_obs_groups, store_code_state, pad_to_size


class OffPolicyRunner:
    """Off-policy runner for training and evaluation of unsupervised/self-supervised methods."""

    def __init__(self, env: URLVecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.num_agent_updates: int = self.cfg["num_agent_updates"]
        self.num_seed_steps_per_env: int = self.cfg["num_seed_steps_per_env"]
        self.log_interval: int = self.cfg["log_interval"]
        self.eval_interval: int = self.cfg["eval_interval"]
        self.save_interval: int = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [robot_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # add expert buffer to environment
        self.env.set_expert_buffer(self.alg.expert_buffer)

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.prepare_for_training()
        self.train_mode()

        # Book keeping
        eval_infos: list[dict[str, torch.Tensor]] = []
        ep_infos: list[dict[str, torch.Tensor]] = []
        loss_infos: list[dict[str, torch.Tensor]] = []
        algo_infos: list[dict[str, torch.Tensor]] = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # initialize some variables
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        eval_time = 0.0
        collection_time = 0.0
        learn_time = 0.0
        z: torch.Tensor | None = None
        last_dones: torch.Tensor | None = None

        # Start training
        for it in range(start_iter, tot_iter):
            with torch.inference_mode():
                # eval to update priorities
                if (it - start_iter) % self.eval_interval == 0:
                    # save and clear rewbuffer and lenbuffer if in the middle of an episode
                    if cur_episode_length.any():
                        rewbuffer.extend(cur_reward_sum[:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[:, 0].cpu().numpy().tolist())
                        cur_reward_sum[:] = 0
                        cur_episode_length[:] = 0

                    # run evaluation
                    start = time.time()
                    eval_infos.extend(self.eval())
                    stop = time.time()
                    eval_time += stop - start

                    # reset env and training variables
                    obs, _ = self.env.reset()
                    obs = obs.to(self.device)
                    last_dones = None

                start = time.time()
                is_seed = it <= self.num_seed_steps_per_env + start_iter

                # Rollout
                z = self.alg.update_rollout_z(z, cur_episode_length, self.env.num_envs)
                # Sample actions
                actions = self.alg.act(obs, z, last_dones, random_sample=is_seed)
                # Step the environment
                obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                # Move to device
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                # process the step
                self.alg.process_env_step(obs, rewards, dones, extras)
                # book keeping
                if self.log_dir is not None:
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])
                    # Update rewards
                    cur_reward_sum += rewards
                    # Update episode length
                    cur_episode_length += 1
                    # Clear data for completed episodes
                    # -- common
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                last_dones = dones

                stop = time.time()
                collection_time += stop - start
                start = stop

            if not is_seed and it % self.num_steps_per_env == 0:
                with torch.inference_mode():
                    self.alg.compute_gammas()

                # update policy
                for _ in range(self.num_agent_updates):
                    loss_dict, extras = self.alg.update()
                    loss_infos.append(loss_dict)
                    algo_infos.append(extras["log"])
                gc.collect()

            stop = time.time()
            learn_time += stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs and it % self.log_interval == 0:
                # Log information
                self.log(locals())
                # Save model
                if (it - start_iter) % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                eval_time = 0.0
                collection_time = 0.0
                learn_time = 0.0

                # Clear episode infos
                eval_infos.clear()
                ep_infos.clear()
                algo_infos.clear()
                loss_infos.clear()

            # callback for video logging
            if self.logger_type in ["wandb"]:
                self.writer.callback(it)

            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def eval(self) -> list[dict[str, torch.Tensor]]:
        assert self.alg.expert_buffer is not None

        # switch to eval mode
        self.eval_mode()

        eval_infos: list[dict[str, torch.Tensor]] = []
        bucket_size = self.alg.expert_buffer.bucket_size

        idx = 0
        for eval_obs in self.alg.expert_buffer.get_batch_motions(self.env.num_envs, device=self.device):
            mini_batch_size = eval_obs.shape[0]
            eval_motions = self.alg.expert_buffer.get_expert_obs(eval_obs)
            eval_zs = self.alg.policy.goal_inference(eval_obs.view(-1)).view(mini_batch_size, bucket_size, -1)
            eval_zs = pad_to_size(eval_zs, self.env.num_envs, dim=0)

            first_motions = pad_to_size(eval_motions[:, 0, :], self.env.num_envs, dim=0)
            # assume that motions only contains root state and joint state
            num_joints = (first_motions.shape[1] - 13) // 2
            obs, _ = self.env.reset_to(
                {
                    "articulation": {
                        "robot": {
                            "root_pose": first_motions[:, :7],
                            "root_velocity": first_motions[:, 7:13],
                            "joint_position": first_motions[:, 13 : 13 + num_joints],
                            "joint_velocity": first_motions[:, 13 + num_joints :],
                        }
                    }
                },
                is_relative=True,
            )
            actual_qpos = torch.zeros((mini_batch_size, bucket_size, num_joints), device=self.device)
            for it in range(bucket_size):
                actions = self.alg.policy.act(obs, eval_zs[:, it, :])
                # pad out remaining envs with zeros
                actions = pad_to_size(actions, self.env.num_envs, dim=0)
                obs, _, _, _ = self.env.step(actions.to(self.env.device))
                obs_tensor = self.alg.expert_buffer.get_expert_obs(obs)
                actual_qpos[:, it, :] = obs_tensor[:mini_batch_size, 13 : 13 + num_joints].to(self.device)

            eval_qpos = eval_motions[:, :, 13 : 13 + num_joints]
            eval_info = self.alg.update_priorities(actual_qpos, eval_qpos, idx)
            eval_infos.append(eval_info)

            idx += mini_batch_size
        self.alg.normalize_priorities()

        # revert to train mode
        self.train_mode()
        return eval_infos

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        assert self.writer is not None

        # Compute the collection size
        collection_size = self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        iteration_time = locs["collection_time"] + locs["learn_time"]
        if locs["eval_time"] > 0.0:
            iteration_time += locs["eval_time"]
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        # -- Eval info
        eval_string = self._get_infos_str(
            locs["eval_infos"],
            locs["it"],
            pad=pad,
            writer_format="Evaluation/{}",
            console_format="Mean evaluation {}",
        )

        # -- Episode info
        ep_string = self._get_infos_str(
            locs["ep_infos"],
            locs["it"],
            pad=pad,
            writer_format="{}",  # episode infos have their own categories
            console_format="Mean {}",
        )

        mean_std = self.alg.policy.action_std.mean().item()
        fps = int(collection_size / iteration_time)

        # -- Losses
        loss_string = self._get_infos_str(
            locs["loss_infos"],
            locs["it"],
            pad=pad,
            writer_format="{}_loss",  # loss infos have their own categories
            console_format="Mean {}_loss",
        )

        # -- Algorithm info
        train_string = self._get_infos_str(
            locs["algo_infos"],
            locs["it"],
            pad=pad,
            writer_format="Train/{}",
            console_format="Train/{}",
        )

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        time_string = f"""collection: {locs["collection_time"]:.3f}s, learning: {locs["learn_time"]:.3f}s"""
        if locs["eval_time"] > 0.0:
            time_string += f", eval: {locs['eval_time']:.3f}s"

        log_string = (
            f"""{"#" * width}\n"""
            f"""{str.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s ({time_string})\n"""
            f"""{"Mean action noise std:":>{pad}} {mean_std:.2f}\n"""
        )
        # -- Evaluation
        log_string += eval_string
        # -- Losses
        log_string += loss_string
        # -- Training info
        log_string += train_string

        if len(locs["rewbuffer"]) > 0:
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # -- episode info
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "forward_optimizer_state_dict": self.alg.forward_optimizer.state_dict(),
            "backward_optimizer_state_dict": self.alg.backward_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # upload model to external logging service
        if not self.disable_logs and self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.forward_optimizer.load_state_dict(loaded_dict["forward_optimizer_state_dict"])
            self.alg.backward_optimizer.load_state_dict(loaded_dict["backward_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self) -> None:
        self.alg.train_mode()

    def eval_mode(self) -> None:
        self.alg.test_mode()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # TODO: multi-GPU training
        raise NotImplementedError

    def _construct_algorithm(self, obs: TensorDict) -> FbCpr:
        """Construct the forward-backward algorithm."""
        # initialize the policy
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ForwardBackward = policy_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: FbCpr = alg_class(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            self.env.num_envs,
            self.env.max_episode_length,
            obs,
            [self.env.num_actions],
            self.cfg["storage_scale"],
            self.cfg["z_buffer_capacity"],
            self.cfg["storage_device"],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from robot_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from robot_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir,
                    flush_secs=10,
                    num_envs=self.env.num_envs,
                    cfg=self.cfg,
                )
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _get_infos_str(
        self,
        infos: list[dict[str, Any]],
        it: int,
        pad: int,
        writer_format: str,
        console_format: str,
    ) -> str:
        assert self.writer is not None
        if not infos:
            return ""
        info_string = ""
        for key in infos[0]:
            infotensor = torch.tensor([], device=self.device)
            for info in infos:
                # handle scalar and zero dimensional tensor infos
                if key not in info:
                    continue
                if not isinstance(info[key], torch.Tensor):
                    info[key] = torch.Tensor([info[key]])
                if len(info[key].shape) == 0:
                    info[key] = info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, info[key].to(self.device)))
            value = torch.mean(infotensor).item()
            # log to logger and terminal
            self.writer.add_scalar(writer_format.format(key), value, it)
            info_string += f"""{f"{console_format.format(key)}:":>{pad}} {value:.4f}\n"""
        return info_string
