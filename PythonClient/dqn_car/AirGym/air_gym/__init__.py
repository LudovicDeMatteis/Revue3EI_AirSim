from gym.envs.registration import register

register(
     id='airsim-car-v0',
     entry_point='AirGym.air_gym.envs:AirSimCarEnv',
)