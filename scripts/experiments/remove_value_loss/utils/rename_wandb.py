def generate_new_name(config):

    # Use config parameters to generate the group name
    group_name = f"{config['algo_name']}"
    group_name += f"_{config['env_id'].replace('/', '').replace('-', '').lower()}"
    group_name += f"_gamma{config['gamma']}"

    
    run_name = f"{group_name}_{config['seed']}"
    return group_name, run_name