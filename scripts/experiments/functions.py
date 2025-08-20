import torch

def drop_critic_mc(returns):
    flat_returns_for_normalization = returns.view(-1)
    mean_returns_batch = flat_returns_for_normalization.mean()
    advantages = (returns - mean_returns_batch)
    return advantages, flat_returns_for_normalization

def drop_critic_gae(rewards, device, args, dones, next_done, returns):
    num_envs = returns.shape[-1]
    value_from_returns = torch.stack([torch.mean(returns, dim=-1)] * num_envs)
    value_from_returns = torch.transpose(value_from_returns, 0, 1)
    next_value = torch.stack([value_from_returns[-1].unsqueeze(0).mean()] * num_envs)
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = value_from_returns[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - value_from_returns[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #TODO: verificar se esta correto 
    returns = advantages + value_from_returns
    flat_returns_for_normalization = returns.view(-1)
    return advantages, flat_returns_for_normalization, returns

def mc_critic(next_value,rewards, device, args, dones, values, next_done):
    returns = torch.zeros_like(rewards).to(device)
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            returns[t] = rewards[t] + args.gamma * next_value * (1.0 - next_done)
        else:
            returns[t] = rewards[t] + args.gamma * returns[t + 1] * (1.0 - dones[t + 1])
    advantages = returns - values
    flat_returns_for_normalization = returns.view(-1)
    return advantages, flat_returns_for_normalization, returns

def gae_critic(next_value,rewards, device, args, dones, values, next_done):
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    flat_returns_for_normalization = returns.view(-1)
    return advantages, flat_returns_for_normalization, returns