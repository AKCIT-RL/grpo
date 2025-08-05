# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml 
from utils.rename_wandb import generate_new_name 
from functions import drop_critic_mc, drop_critic_gae, mc_critic, gae_critic


def add_args(parser):
    # GRPO flags
    parser.add_argument("--remove-value-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-critic", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-group", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-kmeans", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo-group-std", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--entropy", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-gae", type=bool, action=argparse.BooleanOptionalAction, default=False)

    #General settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="the name of this experiment")
    parser.add_argument("--algo-name", type=str, default="ppo",
                        help="The name of the algorithm to use")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Env parameters
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")


    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold") 
    parser.add_argument("--k-groups", type=float, default=3,
                        help="the target KL divergence threshold")
    parser.add_argument("--kl-beta-coef", type=float, default=0.04,
                        help="the target KL divergence threshold")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the YAML configuration file.")

    # to be filled in runtime
    parser.add_argument("--batch-size", type=int, default=0, help="the batch size (computed in runtime: num_envs * num_steps)")
    parser.add_argument("--minibatch-size", type=int, default=0, help="the mini-batch size (computed in runtime: batch_size // num_minibatches)")
    parser.add_argument("--num-iterations", type=int, default=0, help="the number of iterations (computed in runtime)")


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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser) 

    temp_args, remaining_argv = parser.parse_known_args()
    config_file_path = temp_args.config

    yaml_config = {} 
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config is None: 
            yaml_config = {}
    else:
        print(f"Aviso: Arquivo de configuração '{config_file_path}' não encontrado. Usando apenas padrões do argparse e argumentos de linha de comando.")

    for key, value in yaml_config.items():
        attr_name = key.replace('-', '_')
        if hasattr(temp_args, attr_name):
            setattr(temp_args, attr_name, value)

    args = parser.parse_args(remaining_argv, namespace=temp_args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

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
            group=group_name,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.grpo_group or args.grpo_kmeans:
        reference_agent = Agent(envs).to(device)
        logprobs_ref = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.grpo_group or args.grpo_kmeans:
            reference_agent.load_state_dict(agent.state_dict())
            reference_agent.eval()

        if args.grpo_kmeans:
            initial_observations = np.zeros((args.num_envs,) + envs.single_observation_space.shape)

        for step in range(0, args.num_steps):
            if args.grpo_kmeans and step == 0:
                initial_observations = next_obs.cpu().numpy()

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                if args.grpo_group or args.grpo_kmeans:
                    _, logprob_ref, _, _ = reference_agent.get_action_and_value(next_obs)
                    logprobs_ref[step] = logprob_ref
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob                

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            #MC Returns
            returns = torch.zeros_like(rewards).to(device)
            for env_idx in range(args.num_envs):
                current_return = 0
                for t in reversed(range(args.num_steps)):
                    current_return = rewards[t, env_idx] + args.gamma * current_return * (1.0 - dones[t, env_idx])
                    returns[t, env_idx] = current_return
            
            if args.drop_critic and not (args.grpo_kmeans):
                if args.use_gae:
                    advantages, flat_returns_for_normalization, returns = drop_critic_gae(rewards, device, args, dones, next_done, returns)

                else:
                    advantages, flat_returns_for_normalization = drop_critic_mc(returns)

            else:
                if args.use_gae:
                    with torch.no_grad():
                        next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages, flat_returns_for_normalization, returns = gae_critic(next_value, rewards, 
                                                                                     device, args, dones, values, next_done)
                else:
                    with torch.no_grad():
                        next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages, flat_returns_for_normalization, returns = mc_critic(next_value, rewards, device, args, 
                                                                                    dones, values, next_done, returns)

            if args.grpo_group_std:
                std_returns_batch = flat_returns_for_normalization.std() + 1e-8
                advantages = advantages / std_returns_batch

            if args.grpo_group or args.grpo_kmeans:
                b_logprobs_ref = logprobs_ref.reshape(-1)

        if args.grpo_kmeans:
            scaler = StandardScaler()
            scaled_initial_observations = scaler.fit_transform(initial_observations.reshape(args.num_envs, -1))

            kmeans = KMeans(n_clusters=args.k_groups, random_state=args.seed, n_init=10) 
            cluster_labels = kmeans.fit_predict(scaled_initial_observations)
                        
            for cluster_id in range(args.k_groups):
                env_indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
                
                if len(env_indices_in_cluster) == 0:
                    continue 
                
                cluster_returns_per_step = returns[:, env_indices_in_cluster]
                
                cluster_returns_per_step_flat = cluster_returns_per_step.reshape(-1)

                if cluster_returns_per_step_flat.numel() < 2:
                    for env_idx_in_cluster in env_indices_in_cluster:
                        advantages[:, env_idx_in_cluster] = 0.0
                    continue

                if args.drop_critic:
                    if args.use_gae:
                        cluster_advantages, _, _ = drop_critic_gae(rewards[:, env_indices_in_cluster], 
                                                                   device, args, dones[:, env_indices_in_cluster], next_done[env_indices_in_cluster], returns[:, env_indices_in_cluster])

                    else:
                        cluster_advantages = drop_critic_mc(returns[:, env_indices_in_cluster])

                else:
                    if args.use_gae:
                        with torch.no_grad():
                            next_value = agent.get_value(next_obs).reshape(1, -1)
                        next_value = next_value[:, env_indices_in_cluster]
                        cluster_advantages, _, _ = gae_critic(next_value, rewards[:, env_indices_in_cluster], 
                                                                                        device, args, dones[:, env_indices_in_cluster], 
                                                                                        values[:, env_indices_in_cluster], next_done[env_indices_in_cluster])
                    else:
                        with torch.no_grad():
                            next_value = agent.get_value(next_obs).reshape(1, -1)
                        next_value = next_value[:, env_indices_in_cluster]
                        cluster_advantages, _, _ = mc_critic(next_value, rewards[:, env_indices_in_cluster], device, args, 
                                                                                        dones[:, env_indices_in_cluster], values[:, env_indices_in_cluster], 
                                                                                        next_done[env_indices_in_cluster], returns[:, env_indices_in_cluster])

                if args.grpo_group_std:
                    cluster_std_returns_batch = cluster_returns_per_step_flat.std() + 1e-8
                    cluster_advantages = cluster_advantages / cluster_std_returns_batch
                
                for i, env_idx_in_cluster in enumerate(env_indices_in_cluster):
                    advantages[:, env_idx_in_cluster] = cluster_advantages[:, i]

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv: 
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                loss = 0
                if args.grpo_group or args.grpo_kmeans:
                    log_ratio_ref_to_new = b_logprobs_ref[mb_inds] - newlogprob
                    ratio_ref_to_new = log_ratio_ref_to_new.exp()
                    
                    kl_penalty_term = (ratio_ref_to_new - log_ratio_ref_to_new - 1).mean()
                    
                    loss += args.kl_beta_coef * kl_penalty_term

                if args.entropy:
                    entropy_loss = entropy.mean()
                    loss -= args.ent_coef * entropy_loss.mean()
                    
                if not args.remove_value_loss:
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    loss += v_loss * args.vf_coef

                loss += pg_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if not args.remove_value_loss:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if args.entropy:
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()