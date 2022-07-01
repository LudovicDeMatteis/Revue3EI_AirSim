#import setup_path
import gym
import AirGym.air_gym as airgym
#from car_env import AirSimCarEnv
from AirGym.air_gym.envs.car_env import AirSimCarEnv

# gym.envs.register(
#      id='AirGym:airsim-car-v0',
#      entry_point='AirSimCarEnv',
# )

import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import os

load = True
print(f"PID is {os.getpid()}")
# Create a DummyVecEnv for main airsim gym env
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

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env, )

# Initialize RL algorithm type and parameters
if (not(load)):
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.01,
        #gamma=0.5,
        verbose=1,
        batch_size=64,
        train_freq=16,
        target_update_interval=2000,
        learning_starts=100,
        buffer_size=10000,
        # max_grad_norm=10,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        device="cuda",
        tensorboard_log="./tb_logs/",
    )
else:
    print("Loaded model ##############")
    model = DQN.load("Saves/best_model_04_05_1650")
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


# for i in range (1,1000):
# Train for a certain number of timesteps
model.learn(
    total_timesteps=0.5e5, 
    tb_log_name="dqn_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("dqn_airsim_car_policy")

#client.enableApiControl(False)