from contextlib import redirect_stderr
import setup_path
import airsim
import np as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
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

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)

        self.car_controls = airsim.CarControls()
        self.car_state = None

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
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
        else:
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, data):
        img2d = np.reshape(data, (50, 50))
        #
        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((50, 50)).convert("L"))

        return im_final.reshape([50, 50, 1])

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
        X=np.array(points[:2500,0])
        Y=np.array(points[:2500,1])
    
        R=np.sqrt(np.square(X)+np.square(Y))
        T=np.arctan2(Y,X)/pi*180
        
        # TODO
        # Could somebody add the 3rd dimension ?
        Sort = np.concatenate((R,T), axis=0)
        Sort = np.reshape(Sort, (len(R), 2))

        Sort = Sort[Sort[:, 1].argsort()]
        
        R = Sort[:,0]
        T = Sort[:,1]

        print(T)
        print(T.shape)
        return R,T

    def data_format(self, R, T):
        FOV_min = -80
        FOV_max = 80
        Nb_pts_rot = 2500

        theta = np.linspace(FOV_min, FOV_max, Nb_pts_rot) 

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
        BETA = 0.001
        
        car_pt = self.state["pose"].position.to_np_array()
        car_pt_prev = self.state["prev_pose"].position.to_np_array()
        # print(car_pt)

        dist = np.linalg.norm(car_pt -car_pt_prev)

        reward_dist = abs(BETA * dist)
        reward_speed = (
            (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
        )
        reward = reward_dist + reward_speed

        done = 0
        if dist > 10 :
            if self.car_state.speed <= 1:
                done = 1
        elif dist == 0:
            done = 1
            reward = -1
        if self.state["collision"]:
            reward = -1
            done = 1

        print("dist" + reward_dist + "  -   speed" + reward_speed)
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
