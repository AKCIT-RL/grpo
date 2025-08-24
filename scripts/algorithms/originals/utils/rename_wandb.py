def generate_new_name(config):

    # Use config parameters to generate the group name
    exp_name = f"{config['algo_name']}"
    exp_name += f"_{config['env_id'].replace('/', '').replace('-', '').lower()}"
    exp_name += f"_gamma{config['gamma']}_envs{config['num_envs']}"

    if config['algo_name'] == ('grpo_kmeans' or 'grpo_kmeans_continuous' or 'grpo_kmeans_atari'):
        exp_name += f"kgroup{config['k_groups']}"
    
    run_name = f"{exp_name}_{config['seed']}"
    return exp_name, run_name