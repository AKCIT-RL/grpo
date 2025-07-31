import json
import yaml
import re
import wandb

def generate_new_name(config, old_name):
    # Extract date and time from the old name
    date_time_match = re.match(r'(\d{8}_\d{6})', old_name)
    if date_time_match:
        date_time = date_time_match.group(1)
    else:
        raise ValueError(f"Can't extract date and time from {old_name}")

    exp_name = f"{old_name.split('-')[1].split('_slidingpuzzle')[0]}"

    if type(config['env_configs']) != dict:
        config['env_configs'] = json.loads(config['env_configs']) if config['env_configs'] else {}

    if config['env_id'] != "SlidingPuzzle-v0":
        exp_name += "_" + config['env_id'].replace("/", "").replace("-", "").lower()
    
    if "w" in config['env_configs']:
        exp_name += f"_w{config['env_configs']['w']}"
    if "variation" in config['env_configs']:
        exp_name += f"_{config['env_configs']['variation']}"
    if "image_folder" in config['env_configs']:
        exp_name += f"_{config['env_configs']['image_folder'].replace('/', '').replace('-', '').lower()}"
    if "image_pool_size" in config['env_configs']:
        exp_name += f"_p{config['env_configs']['image_pool_size']}"
    
    run_name = f"{date_time}-{exp_name}_{config['seed']}"
    return exp_name, run_name

def rename_wandb_runs():
    api = wandb.Api()
    runs = api.runs("bryanoliveira/cleanrl")

    for run in runs:
        if "slidingpuzzle" in run.name.lower():
            config = run.config
            old_name = run.name
            
            try:
                exp_name, new_name = generate_new_name(config, old_name)
                
                if old_name != new_name:
                    run.name = new_name
                    run.group = exp_name
                    run.update()
                    print(f"Updated run name on wandb: {old_name} -> {new_name}")
                    
                    # Update exp_name in config
                    config['exp_name'] = exp_name
                    run.config.update(config)
                    print(f"Updated exp_name in config for {old_name}: {exp_name}")
                else:
                    print(f"Skipped: {old_name} (name already correct)")
            except Exception as e:
                print(f"Failed to update run {old_name}: {e}")
        else:
            print(f"Skipped: {run.name} (not a slidingpuzzle run)")

if __name__ == "__main__":
    rename_wandb_runs()