from contextlib import redirect_stderr
#import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from AirGym.air_gym.envs.airsim_env import AirSimEnv


class AirSimCarEnv(AirSimEnv):
    def __init__(self, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient()
        self.action_space = spaces.Discrete(7)

        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.lidar_dataR = np.zeros((1600,1))
        self.lidar_dataT = np.zeros((1600,1))
        self.reward = 0

        ### Definition des reset pose possible
        pose1, pose2, pose3 = [airsim.Pose() for i in range(3)]
        pose1.position.x_val, pose1.position.y_val, pose1.position.z_val = 0, 0, 0
        pose2.position.x_val, pose2.position.y_val, pose2.position.z_val = 40, -20, 0
        pose3.position.x_val, pose3.position.y_val, pose3.position.z_val = 10, 10, 0

        self.reset_pose = [pose1, pose2, pose3]

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.simSetVehiclePose(self.reset_pose[1], True, 'Car1')
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        elif action == 5:
            self.car_controls.brake = 0
            self.car_controls.throttle = 0.5
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(0.5)

    def transform_obs(self, data):
        w = 40
        img2d = np.reshape(data, (w, w))
        # print(img2d)
        #
        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((w, w)).convert("L"))

        return im_final.reshape([w, w, 1])

    def parse_lidarData(self, data):    

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    def convert_lidar_data_to_polar(self, points):
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
        pts = 1600
        X=np.array(points[:pts,0])
        Y=np.array(points[:pts,1])
    
        R=np.sqrt(np.square(X)+np.square(Y))
        T=np.arctan2(Y,X)/np.pi*180

        # TODO
        # Could somebody add the 3rd dimension ?
        Sort = np.concatenate((R,T), axis=0)
        Sort = np.reshape(Sort, (2,len(R))).T

        Sort = Sort[Sort[:, 1].argsort()]
        
        R = Sort[:,0]
        T = Sort[:,1]

        # print(R)
        # print(T.shape)
        return R,T

    def data_format(self, R, T):
        FOV_min = -80
        FOV_max = 80
        Nb_pts_rot = 1600

        theta = np.linspace(FOV_min, FOV_max, Nb_pts_rot) 

        if len(R) == Nb_pts_rot:
            self.lidar_dataR = R
            self.lidar_dataT = T
        else:
            R = self.lidar_dataR
            T = self.lidar_dataT

        if (len(theta) != len(T)):

            # if(len(theta) > len(T)):
            #     R = np.append(R, np.zeros((len(theta) - len(R))))
            #     T = np.append(T, np.zeros((len(theta) - len(T))))
            #     # print("append", len(theta), len(R))
            # else :
            #     R = np.delete(R, [range(len(theta), len(R) - len(theta))])
            #     T = np.delete(T, [range(len(theta), len(T) - len(theta))])
            #     # print("delete", len(theta), len(R))

            # print("ok",len(T),len(R)) 
            # Bourage de 0 lorsqu'il manque des points
            for i in range(0,len(theta)):
                if(theta[i] <= T[i]):
                    np.insert(T, i-1, theta[i])
                    np.insert(R, i-1, 0)

            # print("Fin", len(R))
        return R[:Nb_pts_rot],T[:Nb_pts_rot]

    # def write_lidarData_to_disk(self, points):
    #     # TODO
    #     print("not yet implemented")

    def _get_obs(self):
        # changer ici pour obtenir tensor du lidar
        data = airsim.CarClient.getLidarData(self.car, lidar_name = "LidarSensor1", vehicle_name = "Car1")
        points = self.parse_lidarData(data)
        R,T = self.convert_lidar_data_to_polar(points)
        R,T = self.data_format(R,T)
        image = self.transform_obs(R)

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 25
        MIN_SPEED = 1
        # THRESH_DIST = 1000
        BETA = 0.05
        
        car_pt = np.array((self.state["pose"].position.x_val, self.state["pose"].position.y_val))
        car_pt_prev = np.array((self.state["prev_pose"].position.x_val, self.state["prev_pose"].position.y_val))
        # print(car_pt)

        dist = np.linalg.norm(car_pt -car_pt_prev)

        reward_dist = abs(BETA * dist)
        reward_speed = (
            (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
        )
        self.reward += reward_dist + reward_speed

        done = 0
        if dist == 0:
            done = 1
            self.reward = -1
        if self.state["collision"]:
            self.reward = -1
            done = 1

        print("Reward : ", self.reward)
        return self.reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
