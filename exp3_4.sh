#!/bin/bash

# 1. Lista de sementes para os experimentos
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# 2. Lista de nomes dos projetos (algoritmos)
PROJECT_NAMES=(
    "reinforce-clip-batch"
    "reinforce-clip-gae"
)

# 4. Lista de ambientes para os experimentos
declare -a ENVIRONMENTS=(
    "CartPole-v1"
    "Acrobot-v1"
    "Humanoid-v4"
)

# 5. Lista de número de ambientes para os experimentos
declare -a NUM_ENVS_LIST=(
    "8"
    "16"
    "32"
    "64"
)

# --- Início dos Loops Aninhados ---

for SEED in "${SEEDS[@]}"; do
    for PROJECT_NAME in "${PROJECT_NAMES[@]}"; do

        # 3. Lógica condicional para definir as flags do algoritmo
        if [[ "$PROJECT_NAME" == "reinforce-clip-batch" ]]; then
            FLAGS_STRING="--no-use-value-function --no-use-gae --no-use-entropy --use-returns-mean-baseline"
        elif [[ "$PROJECT_NAME" == "reinforce-clip-gae" ]]; then
            FLAGS_STRING=" --no-use-entropy"
        fi

        for ENV_NAME in "${ENVIRONMENTS[@]}"; do
            # Define o total de timesteps com base no ambiente
            if [[ "$ENV_NAME" == "Humanoid-v4" ]]; then
                TOTAL_TIMESTEPS=4000000
            else
                TOTAL_TIMESTEPS=1000000
            fi

            if [[ "$ENV_NAME" == "Humanoid-v4" ]]; then
                PYTHON_SCRIPT="scripts/algorithms/no-baseline/ppo_continuous_action.py"
            else
                PYTHON_SCRIPT="scripts/algorithms/no-baseline/ppo.py"
            fi

            for NUM_ENVS in "${NUM_ENVS_LIST[@]}"; do
                poetry run python "$PYTHON_SCRIPT" \
                    --env-id "$ENV_NAME" \
                    --num-envs "$NUM_ENVS" \
                    --seed "$SEED" \
                    --total-timesteps "$TOTAL_TIMESTEPS" \
                    $FLAGS_STRING \
                    --track \
                    --update-epochs 1 \
                    --wandb-project-name "$PROJECT_NAME"
            done
        done
    done
done