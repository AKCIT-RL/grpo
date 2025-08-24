import os
import subprocess

# 1. Lista de sementes para os experimentos
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 2. Lista de nomes dos projetos (algoritmos)
PROJECT_NAMES = [
    "ppo-grpo",
]

# 4. Lista de ambientes para os experimentos
ENVIRONMENTS = [
    "CartPole-v1",
    "Acrobot-v1"
]

# 5. Lista de número de ambientes para os experimentos
NUM_ENVS_LIST = [
    8,
    16,
    32,
    64
]

# --- Início dos Loops Aninhados ---

for seed in SEEDS:
    for project_name in PROJECT_NAMES:

        # 3. Lógica condicional para definir as flags do algoritmo
        if project_name == "ppo-grpo":
            FLAGS_STRING = ""
        elif project_name == "reinforce-clip-gae":
            FLAGS_STRING = "--no-use-entropy"
        
        for env_name in ENVIRONMENTS:
            # Define o total de timesteps com base no ambiente
            if env_name == "HalfCheetah-v4":
                TOTAL_TIMESTEPS = 4000000
            else:
                TOTAL_TIMESTEPS = 1000000

            if env_name == "HalfCheetah-v4":
                PYTHON_SCRIPT = "scripts/algorithms/no-baseline/ppo_continuous_action.py"
            else:
                PYTHON_SCRIPT = "scripts/algorithms/no-baseline/ppo.py"

            for num_envs in NUM_ENVS_LIST:
                command = [
                    "poetry", "run", "python", PYTHON_SCRIPT,
                    "--env-id", env_name,
                    "--num-envs", str(num_envs),
                    "--seed", str(seed),
                    "--total-timesteps", str(TOTAL_TIMESTEPS),
                    *FLAGS_STRING.split(),
                    "--track",
                    "--wandb-project-name", project_name
                ]
                
                print(f"Executando comando: {' '.join(command)}")

                try:
                    subprocess.run(command, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"O comando falhou com o erro: {e}")
                
                print("\n")

print("Todos os experimentos foram concluídos.")
