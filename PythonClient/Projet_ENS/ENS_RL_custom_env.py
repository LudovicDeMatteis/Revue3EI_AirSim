from airsim.types import Vector3r
import airsim

import sys
import math
import time
import argparse
import pprint
import numpy as np

import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpPolicy

NB_POINT = 36
CHECK_1_POSE = [51, 0]
CHECK_2_POSE = [51, -51]

class CarControl:

    def __init__(self, defaultPose):
        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.collision_info = self.client.simGetCollisionInfo()

        self.client.simPrintLogMessage("Client Connecte", "1", 2)
        self.reset_pose = defaultPose
        self.client.simSetVehiclePose(self.reset_pose, True, 'Car1')
        self.collision_points = []
        self.reward = {"collision" : -150,
                        "vitesse" : 0.5,
                        "distance" : 2}
        self.prev_dist = airsim.Vector3r()
        self.checkpoint = 0

    def change_control(self, steering, throttle):
        self.car_controls.steering = steering
        self.car_controls.throttle = throttle
        self.client.setCarControls(self.car_controls) 
    
    def distance_parcourue(self):
        pose = self.client.simGetObjectPose("Car1")
        d = pose.position

        dist_car = 0
        if(self.checkpoint==0):
            print("In checkpoint 0")
            dist_car = (d.x_val - self.reset_pose.position.x_val) ## On veut avancer
            if(d.x_val >= CHECK_1_POSE[0]):
                self.checkpoint += 1
                print("Checkpoint 0 done")

        if(self.checkpoint==1):
            print("In checkpoint 1")
            dist_car = -1*(d.y_val - CHECK_1_POSE[1])  ## On veut avancer
            if(d.y_val <= CHECK_2_POSE[1]):
                self.checkpoint += 1
                print("Checkpoint 1 done")
        print(self.checkpoint)
        dist_car += 50*self.checkpoint

        print("Récompense de distance : {}".format(dist_car*self.reward["distance"]))
        return(dist_car)

    def forward(self, T):
        state = self.client.getCarState()
        #print(f"State of the car is : {state}")
        control_over = False
        while(not(control_over)):
            self.change_control(0.5,1)
            
            # Get lidar data 
            lidarData = self.client.getLidarData()
            if (len(lidarData.point_cloud) < 3):
                    print("\tNo points received from Lidar data")
            else:
                points = self.parse_lidarData(lidarData)
                #print("\tReading : time_stamp: %d number_of_points: %d" % (lidarData.time_stamp, len(points)))
                #print("\t\tlidar position: %s" % (pprint.pformat(lidarData.pose.position)))
                #print("\t\tlidar orientation: %s" % (pprint.pformat(lidarData.pose.orientation)))
            
            control_over = self.detect_collisions()

        #self.stop()
            
    def detect_collisions(self):
        self.collision_info = self.client.simGetCollisionInfo()

        if(self.collision_info.has_collided):
            print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
                pprint.pformat(self.collision_info.position), 
                pprint.pformat(self.collision_info.normal), 
                pprint.pformat(self.collision_info.impact_point), 
                self.collision_info.penetration_depth, self.collision_info.object_name, self.collision_info.object_id))
            self.collision_points.append(self.collision_info.impact_point)
            self.client.simPlotPoints(self.collision_points, [1.0, 0.0, 0.0, 1.0], 10.0, -1.0, True)
            return(True)
        return(False)

    def parse_lidarData(self, data):
        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    def stop(self):

        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.handbrake = True
        self.client.setCarControls(self.car_controls)

        airsim.wait_key('Press any key to reset to original state')

        self.client.simSetVehiclePose(self.reset_pose, True, 'Car1')
        #print(self.reset_pose)
        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")


class CarRLEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, controller, N_POINTS):
    super(CarRLEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions :
    self.obs_size = N_POINTS
    self.action_space = spaces.Box(
                            np.array([-0.5,-1]), 
                            np.array([+0.5,+1,]),
                            dtype=np.float32,
                        )
    # Example for using image as input:
    self.observation_space = spaces.Box(
                            np.ones(N_POINTS)*(-500),
                            np.ones(N_POINTS)*500,
                            dtype=np.float32,
                        )
    self.control = controller

  def step(self, action):
    #print("Action measured : " + str(action))
    observation = None
    reward = 0
    done = False
    info = {}
    state = self.control.client.getCarState()

    self.control.car_controls.steering = float(action[0])
    self.control.car_controls.throttle = float(action[1])
    self.control.car_controls.handbrake = False
    self.control.client.setCarControls(self.control.car_controls)

    while(observation is None):
        lidarData = controller.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            observation = None
        else:
            observation = convert_lidar_data_to_polar(lidarData)
            observation = align_lidar(observation, NB_POINT)

    collision = controller.detect_collisions()
    if collision or (state.speed == 0.0 and action[1] == 0.0):
        reward = controller.reward["collision"]
        done = True
    else:
        reward = self.control.reward["vitesse"]*state.speed 
        reward += self.control.reward["distance"]*self.control.distance_parcourue()

    return(observation, reward, done, info)

  def reset(self):
    observation = None
    self.control.car_controls.throttle = 0
    self.control.car_controls.steering = 0
    self.control.car_controls.handbrake = True

    self.control.client.setCarControls(self.control.car_controls)
    self.control.checkpoint = 0

    time.sleep(1)

    self.control.client.simSetVehiclePose(self.control.reset_pose, True, 'Car1')

    time.sleep(1)

    # Clear the collision buffer by reading all collisions
    collision_info = controller.client.simGetCollisionInfo()
    while(collision_info.has_collided):
        collision_info = controller.client.simGetCollisionInfo()

    while(observation is None):
        lidarData = controller.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            observation = None
        else:
            observation = convert_lidar_data_to_polar(lidarData)
            observation = align_lidar(observation, NB_POINT)
        
    return observation  # reward, done, info can't be included

  def close (self):
    self.control.stop()
    return(0)

def convert_lidar_data_to_polar(lidar_data):
    """
    Parameters
    ----------
    lidar_data : TYPE LidarData
    
        Transforms the lidar data to convert it to a real life format, that is from
        (x_hit, y_hit, z_hit) to (angle_hit, distance_hit). Make sure to set
        "DatFrame": "SensorLocalFrame" in the settings.JSON to get relative
        coordinates from hit-points.
        
        Note : so far, only 2 dimensions lidar is supported. Thus, the Z coordinate
        will simply be ignored

    Returns
    -------
    converted_lidar_data=np.array([theta_1, ..., theta_n]) , np.array([r_1, ..., r_n]).
    """
    list=lidar_data.point_cloud
    X=np.array(list[0::3])
    Y=np.array(list[1::3])
    
    R=np.sqrt(X**2+Y**2)
    T=np.arctan2(Y,X)
    
    # TODO
    # Could somebody add the 3rd dimension ?
    
    return np.column_stack((T,R))

def align_lidar(observation, nb_points):
    observation = observation[:min(NB_POINT, len(observation))]
    observation = observation[observation[:, 0].argsort()]
    angles_vises = np.linspace(-np.pi, np.pi, 37)[:-1]
    count = 0
    observation_complete = []
    for a in angles_vises:
        if(count< len(observation) and np.abs(a-observation[count, 0]) < np.pi/nb_points):
            observation_complete.append(observation[count])
            count += 1
        else:
            observation_complete.append(np.array([a,200]))
    return(np.array(observation_complete)[:,1])

if __name__ == "__main__":
    defaultPose = airsim.Pose()
    defaultPose.position.z_val = 1

    controller = CarControl(defaultPose)
    size_action_space = 2

    env = CarRLEnv(controller, NB_POINT)
    # It will check your custom environment and output additional warnings if needed
    check_env(env, warn=True)

    # obs = env.reset()

    # print(env.observation_space)
    # print(env.action_space)
    # print(env.action_space.sample())

    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=100000, reset_num_timesteps=True)

    # Test the trained agent
    print("Model training over")
    obs = env.reset()
    n_steps = 100000
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break

    print("Over")