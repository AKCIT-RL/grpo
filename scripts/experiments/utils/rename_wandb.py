def generate_new_name(config):

    # Use config parameters to generate the group name
    exp_name = f"{config['algo_name']}"
    exp_name += f"_{config['env_id'].replace('/', '').replace('-', '').lower()}"
    exp_name += f"_gamma{config['gamma']}"

    
    run_name = f"{exp_name}_{config['seed']}"
    return exp_name, run_name