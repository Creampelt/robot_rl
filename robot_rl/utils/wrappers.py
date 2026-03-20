from dataclasses import MISSING
from typing import TypeVar

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

RunnerCfg = TypeVar("RunnerCfg", bound=RslRlOnPolicyRunnerCfg)


@configclass
class RobotRlActorCriticCfg(RslRlPpoActorCriticCfg):
    estimator_index: int = -1
    oracle: bool = False


@configclass
class ActorCriticMHACfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticMHA"
    n_heads: int = MISSING
    n_latent: int = MISSING
    n_channels: int = MISSING
    kernel_size: int = MISSING
    dropout: float = 0.0
    n_rows: int = MISSING
    n_cols: int = MISSING


@configclass
class SAECfg:
    """Configuration for the SAE network."""

    class_name: str = "SAE"
    """The policy class name. Default is SAE."""

    hidden_dim_scale: int = MISSING
    """The hidden dimensions of the SAE network."""

    sparsity_lambda: float = 1e-3
    """The loss coefficient encouraging sparsity in the feature dimension"""

    activation: str = "relu"
    """The activation function for the probe network."""


@configclass
class ProbeCfg:
    """Configuration for the Probe network."""

    class_name: str = "Probe"
    """The policy class name. Default is Probe."""

    probe_hidden_dims: list = []
    """The hidden dimensions of the probe network."""

    activation: str = "elu"
    """The activation function for the probe network."""


@configclass
class ProbeAlgorithmCfg:
    """Configuration for the Probe algorithm."""

    class_name: str = "Probe"
    """The algorithm class name. Default is Probe."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    learning_rate: float = MISSING
    """The learning rate for the probe."""

    estimate_obs: bool = False
    """Whether to estimate the observation (time t)."""

    estimate_next_obs: bool = False
    """Whether to estimate the next observation (time t+1)."""


@configclass
class RobotRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    storage_device: str | None = None
    """The device on which to store the rollout buffer. Defaults to None, in which case the runner device is used."""


@configclass
class RobotRlProbeRunner(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for probe algorithms."""

    algorithm: ProbeAlgorithmCfg = MISSING
    """The algorithm configuration."""

    probe: ProbeCfg | SAECfg = MISSING
    """The probe configuration."""

    layers: list = MISSING
    """The ActorCritic layers to probe."""

    policy_module: str = "actor"
    """ActorCritic attribute to probe.

    e.g. policy.policy_module[i]
    """

    probe_obs: bool = MISSING
    """Whether to probe the observation (time t)."""

    probe_next_obs: bool = MISSING
    """Whether to probe the next observation (time t+1)."""


@configclass
class EstimatorCfg:
    estimate_loss_coef: float = MISSING
    """Coef for estimator loss term."""

    estimate_loss_ramp: int = MISSING
    """Steps to linearly ramp recons coef."""

    estimate_obs: bool = MISSING
    """Whether to estimate the observation (time t)."""

    estimate_next_obs: bool = MISSING
    """Whether to estimate the next observation (time t+1)."""


@configclass
class MetaRlCfg:
    num_episodes_per_trial: int = MISSING
    """The number of episodes per trial (memory reset)."""


@configclass
class RobotRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_cfg: EstimatorCfg | None = None
    """The estimator configuration. Default is None, in which case it is not used."""

    meta_rl_cfg: MetaRlCfg | None = None
    """The meta RL configuration. Default is None, in which case it is not used."""


@configclass
class RobotRlForwardBackwardCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ForwardBackward"
    """The policy class name. Default is ForwardBackward."""

    z_dim: int = MISSING
    """The size of the latent task space."""

    fb_tau: float = MISSING
    """The polyak coefficient for updating forward and backward target networks."""

    critic_tau: float = MISSING
    """The polyak coefficient for updating critic target network."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_num_parallel: int = MISSING
    """The number of parallel networks for the actor."""

    actor_embedding_dims: list[int] = MISSING
    """The embedding dimensions of the actor network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_num_parallel: int = MISSING
    """The number of parallel networks for the critic network."""

    critic_embedding_dims: list[int] = MISSING
    """The embedding dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    backward_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the backward network."""

    discriminator_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""


@configclass
class RobotRlFbCprAlgorithmCfg:
    """Configuration for the FB-CPR algorithm."""

    class_name: str = "FbCpr"
    """The algorithm class name. Default is PPO."""

    motion_path: str = MISSING
    """Path to .pt file containing motion trajectories. Motions should be stored as TensorDict observations clipped to
    a fixed sequence length (see `play_trajectories` script)."""

    expert_sequence_length: int = MISSING
    """Sequence length when sampling motion trajectories."""

    steps_per_z_update: int = MISSING
    """The number of environment steps between z updates."""

    actor_learning_rate: float = MISSING
    """The learning rate for the actor."""

    forward_learning_rate: float = MISSING
    """The learning rate for the forward model."""

    backward_learning_rate: float = MISSING
    """The learning rate for the backward model."""

    discriminator_learning_rate: float = MISSING
    """The learning rate for the discriminator model."""

    disc_critic_learning_rate: float = MISSING
    """The learning rate for the discriminator critic model."""

    aux_critic_learning_rate: float = MISSING
    """The learning rate for the auxiliary critic model."""

    weight_decay: float = MISSING
    """The weight decay for the model optimizers."""

    max_grad_norm: float | None = MISSING
    """The maximum gradient norm, or None to disable gradient clipping."""

    clip_actor_std: float = MISSING
    """The clip to apply to the actor distribution stddev."""

    gamma: float = MISSING
    """The discount factor."""

    train_goal_ratio: float = MISSING
    """The ratio of z values to replace with state embeddings."""

    expert_asm_ratio: float = MISSING
    """The ratio of z values to replace with expert embeddings."""

    expert_rollout_ratio: float = MISSING
    """The ratio of z values to replace with expert embeddings during rollout inference."""

    expert_rollout_length: int = MISSING
    """The length of expert z trajectories during rollout inference."""

    z_relabel_ratio: float = MISSING
    """The ratio of training z values to relabel with sampled mixture."""

    forward_backward_pessimism: float = MISSING
    """The coefficient when computing forward and backward TD targets."""

    actor_pessimism: float = MISSING
    """The coefficient when computing actor TD targets."""

    disc_critic_pessimism: float = MISSING
    """The coefficient when computing discriminator critic TD targets."""

    aux_critic_pessimism: float = MISSING
    """The coefficient when computing auxiliary critic TD targets."""

    discriminator_reg_coef: float = MISSING
    """The coefficient for the discriminator regulation actor loss."""

    aux_reg_coef: float = MISSING
    """The coefficient for the auxiliary regulation actor loss."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    ortho_loss_coef: float = MISSING
    """The coefficient for the orthonormality (covariance) loss."""

    grad_loss_coef: float = MISSING
    """The coefficient for gradient loss."""

    batch_size: int = MISSING
    """The batch size for updating the policy."""

    dtype: str = "float32"
    """The dtype to use during training. Defaults to float32."""


@configclass
class RobotRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OffPolicyRunner"
    """The runner class name. Default is OffPolicyRunner."""

    num_steps_per_env = MISSING
    """The number of steps per environment per update."""

    num_agent_updates: int = MISSING
    """The number of policy updates per environment step."""

    num_seed_steps_per_env: int = MISSING
    """The number of seed steps per environment."""

    log_interval: int = MISSING
    """The number of iterations between logs."""

    eval_interval: int = MISSING
    """The number of iterations between evaluations."""

    storage_device: str = MISSING
    """The device to store the replay buffer on."""

    storage_scale: int = MISSING
    """The scale factor for determining storage capacity. Capacity is `storage_scale * num_envs *
    episode_length_steps`."""

    z_buffer_capacity: int = MISSING
    """Capacity for the z buffer."""

    policy: RobotRlForwardBackwardCfg = MISSING
    """The policy configuration."""

    algorithm: RobotRlFbCprAlgorithmCfg = MISSING
    """The algorithm configuration."""
