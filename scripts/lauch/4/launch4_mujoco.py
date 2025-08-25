import os
import subprocess

# 1. Lista de sementes para os experimentos
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 2. Lista de nomes dos projetos (algoritmos)
PROJECT_NAMES = [
    "grpo-gamma",
]

# 4. Lista de ambientes para os experimentos
ENVIRONMENTS = [
    "HalfCheetah-v4"
]

NUM_ENVS_LIST = [
    8
]

# Listas de horizonte para ambientes não-Mujoco e Mujoco
HORIZON_LIST = [
    32,
    64,
    128,
    256, 
    512
]

HORIZON_MUJOCO = [
    512,
    1024,
    2048,
    4096,
    8192
]
# --- Início dos Loops Aninhados ---

for seed in SEEDS:
    for project_name in PROJECT_NAMES:

        # 3. Lógica condicional para definir as flags do algoritmo
        if project_name == "grpo-gamma":
            FLAGS_STRING = "--no-use-baseline --no-use-entropy"
        elif project_name == "grpo-group-no-entropy":
            FLAGS_STRING = "--no-use-entropy"
        
        for env_name in ENVIRONMENTS:
            # Define o total de timesteps e a lista de horizontes com base no ambiente
            if env_name == "HalfCheetah-v4":
                TOTAL_TIMESTEPS = 4000000
                HORIZON_LIST_FOR_ENV = HORIZON_MUJOCO
                PYTHON_SCRIPT = "scripts/algorithms/originals/grpo_group_continuous_action.py"
            else:
                TOTAL_TIMESTEPS = 1000000
                HORIZON_LIST_FOR_ENV = HORIZON_LIST
                PYTHON_SCRIPT = "scripts/algorithms/originals/grpo_group.py"
            
            for num_envs in NUM_ENVS_LIST:
                for horizon in HORIZON_LIST_FOR_ENV:
                    command = [
                        "poetry", "run", "python", PYTHON_SCRIPT,
                        "--env-id", env_name,
                        "--num-envs", str(num_envs),
                        "--seed", str(seed),
                        "--total-timesteps", str(TOTAL_TIMESTEPS),
                        "--num-steps", str(horizon), # Adicionando a flag de horizonte
                        "--gamma", str(1),
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
