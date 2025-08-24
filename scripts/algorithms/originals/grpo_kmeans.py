import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.rename_wandb import generate_new_name


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 10
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "grpo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    algo_name: str = "grpo_kmeans"

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 20
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    k_groups: int = 3
    kl_beta: float = 0.04 

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size 
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    group_name, run_name = generate_new_name(vars(args))
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"group_{args.env_id}__{args.exp_name}", 
        )
    writer = SummaryWriter(f"kl_runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    reference_agent = Agent(envs).to(device)
    reference_agent.load_state_dict(agent.state_dict())
    reference_agent.eval()

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        initial_observations = np.zeros((args.num_envs,) + envs.single_observation_space.shape)
        
        for step in range(0, args.num_steps):
            if step == 0: 
                initial_observations = next_obs.cpu().numpy()
            
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = agent.get_action(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, terminations_np, truncations_np, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations_np, truncations_np)
            rewards[step] = torch.tensor(reward_np).to(device)
            next_obs, next_done = torch.Tensor(next_obs_np).to(device), torch.Tensor(next_done_np).to(device)

            if "final_info" in infos:
                for idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        returns_per_step = torch.zeros_like(rewards).to(device)
        for env_idx in range(args.num_envs):
            current_return_sum = 0.0
            for t in reversed(range(args.num_steps)):
                current_return_sum = rewards[t, env_idx] + args.gamma * current_return_sum * (1.0 - dones[t, env_idx])
                returns_per_step[t, env_idx] = current_return_sum

        scaler = StandardScaler()
        scaled_initial_observations = scaler.fit_transform(initial_observations.reshape(args.num_envs, -1))

        kmeans = KMeans(n_clusters=args.k_groups, random_state=args.seed, n_init=10) 
        cluster_labels = kmeans.fit_predict(scaled_initial_observations)
        
        advantages_per_step = torch.zeros_like(logprobs).to(device)
        
        for cluster_id in range(args.k_groups):
            env_indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
            
            if len(env_indices_in_cluster) == 0:
                continue 
            
            cluster_returns_per_step = returns_per_step[:, env_indices_in_cluster]
            
            cluster_returns_per_step_flat = cluster_returns_per_step.reshape(-1)

            if cluster_returns_per_step_flat.numel() < 2:
                for env_idx_in_cluster in env_indices_in_cluster:
                    advantages_per_step[:, env_idx_in_cluster] = 0.0
                continue
            
            cluster_mean_return_per_step = cluster_returns_per_step_flat.mean()
            cluster_std_return_per_step = cluster_returns_per_step_flat.std() + 1e-8 
            
            cluster_advantages_flat = (cluster_returns_per_step_flat - cluster_mean_return_per_step) / cluster_std_return_per_step
            
            cluster_advantages_reshaped = cluster_advantages_flat.reshape(args.num_steps, len(env_indices_in_cluster))

            for i, env_idx_in_cluster in enumerate(env_indices_in_cluster):
                advantages_per_step[:, env_idx_in_cluster] = cluster_advantages_reshaped[:, i]


        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages_per_step.reshape(-1) 
        
        b_inds = np.arange(args.batch_size) 
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions.long()[mb_inds])
                with torch.no_grad():
                    logits_ref = reference_agent.actor(b_obs[mb_inds])
                    probs_ref = Categorical(logits=logits_ref)
                    logprobs_ref = probs_ref.log_prob(b_actions.long()[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds] 
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                
                log_ratio_ref_vs_theta = logprobs_ref - newlogprob
                ratio_ref_vs_theta = log_ratio_ref_vs_theta.exp()
                kl_penalty_term = (ratio_ref_vs_theta - log_ratio_ref_vs_theta - 1).mean()
                
                loss = pg_loss - args.ent_coef * entropy_loss + args.kl_beta * kl_penalty_term 

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        reference_agent.load_state_dict(agent.state_dict())
        reference_agent.eval()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step) 
        writer.add_scalar("losses/kl_penalty", kl_penalty_term.item(), global_step) 
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()