import gym
import AirGym.air_gym as airgym
from AirGym.air_gym.envs.car_env import AirSimCarEnv


import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Définition de l'environnement
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

print("Loaded model ##############")
model = DQN.load("Saves/best_model_27_04_1049")
#model.set_env(env)


# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Les tests ont donnés les résultats suivants:\n\tRécompense moyenne : {mean_reward}\n\tEcart type : {std_reward}")

# Utilisation de l'agent entrainé et enregistrement des positions pour un affichage futur
all_pos = []
obs = env.reset()
done = False
while(not(done)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    all_pos.append([info[0]['pose'].position.x_val, info[0]['pose'].position.y_val])

# Tracé d'une trajectoire de la voiture
import matplotlib.pyplot as plt
X_pos = np.array(all_pos)[:-2,0]
Y_pos = np.array(all_pos)[:-2,1]
plt.figure()
plt.plot(Y_pos, -X_pos)
plt.xlim([-140, 40])
plt.ylim([-120, 60])
plt.show()