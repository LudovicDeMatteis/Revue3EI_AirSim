import gym
import AirGym.air_gym as airgym
from AirGym.air_gym.envs.car_env import AirSimCarEnv

import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import os

# Utilisation d'un modèle déjà enregistré ou création d'un nouveau modèle
load = True
chemin_modele = "Saves/best_model_04_05_1650"

# Création de l'environnement gym pour AirSim
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "AirGym.air_gym:airsim-car-v0",
                ip_address="127.0.0.1",
                port_number = 41452,
                image_shape=(200, 1, 1),
            )
        )
    ]
)
env = VecTransposeImage(env, )

if (not(load)):
    # Initialisation du modèle d'apprentissage par renforcement
    # En l'occurence on utilise un MLP en changeant certains paramètres d'apprentissage
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.01,
        verbose=1,
        batch_size=64,
        train_freq=16,
        target_update_interval=2000,
        learning_starts=100,
        buffer_size=10000,
        tensorboard_log="./tb_logs/",
    )
else:
    model = DQN.load(chemin_modele)
    print("############# Loaded model ##############")
    model.set_env(env)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=1000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Entrainement pour 1e5 timesteps
model.learn(
    total_timesteps=100000, 
    tb_log_name="dqn_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("dqn_airsim_car_policy")