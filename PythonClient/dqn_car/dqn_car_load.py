import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(40, 40, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

model = DQN.load("best_model")
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
    total_timesteps=1e5, tb_log_name="dqn_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("dqn_airsim_car_policy")

#client.enableApiControl(False)