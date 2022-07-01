from contextlib import redirect_stderr
from unittest import skip
#import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from AirGym.air_gym.envs.airsim_env import AirSimEnv
from PIL import Image

QUAT_90 = [0.707,0,0,0.707]
QUAT_180 = [0,0,0,1]
QUAT_270 = [-0.707,0,0,0.707]
CLOCK_SPEED = 1
ACTION_SLEEP = 0.1 / CLOCK_SPEED

class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, port_number, image_shape):
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

        self.car = airsim.CarClient(ip=ip_address,port=port_number)
        self.action_space = spaces.Discrete(16)
        self.actions = [[0,0,0,0,0,0,1,1,1,1,1,.5,.5,.5,.5,.5],
                        [0,-0.5,-0.25,0,0.25,0.5,-0.5,-0.25,0,0.25,0.5,-0.5,-0.25,0,0.25,0.5]]

        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.lidar_dataR = np.zeros((200,1))
        self.lidar_dataT = np.zeros((200,1))
        self.reward = 0

        ### Definition des reset pose possible
        pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8 = [airsim.Pose() for i in range(8)]
        pose1.position.x_val, pose1.position.y_val, pose1.position.z_val = 0, 0, 0.5
        #pose1.orientation.w_val, pose1.orientation.x_val, pose1.orientation.y_val, pose1.orientation.z_val = QUAT_180
        pose6.position.x_val, pose6.position.y_val, pose6.position.z_val = 0, 0, 0.5
        pose6.orientation.w_val, pose6.orientation.x_val, pose6.orientation.y_val, pose6.orientation.z_val = QUAT_180

        pose2.position.x_val, pose2.position.y_val, pose2.position.z_val = 40, -20, 0.5
        pose2.orientation.w_val, pose2.orientation.x_val, pose2.orientation.y_val, pose2.orientation.z_val = QUAT_90
        pose7.position.x_val, pose7.position.y_val, pose7.position.z_val = 40, -20, 0.5
        pose7.orientation.w_val, pose7.orientation.x_val, pose7.orientation.y_val, pose7.orientation.z_val = QUAT_270

        pose3.position.x_val, pose3.position.y_val, pose3.position.z_val = 100, -90, 0.5
        pose3.orientation.w_val, pose3.orientation.x_val, pose3.orientation.y_val, pose3.orientation.z_val = QUAT_90
        pose8.position.x_val, pose8.position.y_val, pose8.position.z_val = 100, -90, 0.5
        pose8.orientation.w_val, pose8.orientation.x_val, pose8.orientation.y_val, pose8.orientation.z_val = QUAT_270

        pose4.position.x_val, pose4.position.y_val, pose4.position.z_val = 80, -120, 0.5
        pose4.orientation.w_val, pose4.orientation.x_val, pose4.orientation.y_val, pose4.orientation.z_val = QUAT_180
        pose5.position.x_val, pose5.position.y_val, pose5.position.z_val = 80, -120, 0.5
        #pose5.orientation.w_val, pose5.orientation.x_val, pose5.orientation.y_val, pose5.orientation.z_val = QUAT_180

        self.reset_pose = [pose1, pose2, pose3, pose4, pose5]

        self.count = 0
        self.old_reset = 0
        self.warning_stop = False

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        if(self.count==10):
            self.count = 0
            self.old_reset = np.random.randint(0,5)
        self.car.simSetVehiclePose(self.reset_pose[self.old_reset], True, 'Car1')
        self.count += 1 
        self.car.armDisarm(True)
        time.sleep(0.1)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        if action == 0:
            self.car_controls.brake = 1

        self.car_controls.throttle = self.actions[0][action]
        self.car_controls.steering = self.actions[1][action]

        self.car.setCarControls(self.car_controls)
        time.sleep(ACTION_SLEEP)

    def transform_obs(self, data):
        w = 200
        h = 1
        img2d = np.reshape(data, (w, h))
        # print(img2d)
        #

        # image = Image.fromarray(img2d)
        # im_final = np.array(image.resize((w, h)).convert("L"))

        # return im_final.reshape([w, h, 1])

        return(img2d.reshape((w,h,1)))

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
        pts = 200
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
        return(R ,T)

    def data_format(self, R, T):
        FOV_min = -120
        FOV_max = 120
        Nb_pts_rot = 200

        theta = np.linspace(FOV_min, FOV_max, Nb_pts_rot) 

        R = R[:min(Nb_pts_rot,len(R))]  # On prend un maximum de points jusqu'à la limite
        T = T[:min(Nb_pts_rot,len(T))]

        count = len(T)
        T_new = T
        R_new = R
        epsilon = 1.5*(FOV_max-FOV_min)/(Nb_pts_rot)
        while(count<=200):
            R_complete = []
            T_complete = []
            for t_i, t in enumerate(T_new[:-1]):
                if(count<=200 and np.abs(t-T_new[t_i+1])>epsilon):
                        T_complete.append(t)
                        T_complete.append((t+T_new[t_i+1])/2)
                        R_complete.append(R_new[t_i])
                        R_complete.append((R_new[t_i]+R_new[t_i+1])/2)
                        count += 1
                else:
                    T_complete.append(t)
                    R_complete.append(R_new[t_i])
            T_complete.append(T_new[-1])
            R_complete.append(R_new[-1])
            epsilon /= 1.5
            T_new = T_complete
            R_new = R_complete

        # skipped = 0
        # epsilon = 2.5#(FOV_max-FOV_min)/(Nb_pts_rot)
        # for t_i, t in enumerate(theta):
        #     if((t_i-skipped)<len(R) and np.abs(t-T[t_i-skipped])<epsilon):
        #         print((t_i-skipped)<len(R))
        #         print(np.abs(t-T[t_i-skipped])<epsilon)
        #         print(f"We have {t} and {T[t_i-skipped]}")
        #         print(f'S : {skipped} and E :{epsilon}')
        #         R_complete.append(R[t_i-skipped])
        #         T_complete.append(T[t_i-skipped])
        #     else:
        #         print((t_i-skipped)<len(R))
        #         print(np.abs(t-T[t_i-skipped])<epsilon)
        #         print(f"We have {t} and {T[t_i-skipped]}")
        #         print(f'S : {skipped} and E :{epsilon}')
        #         R_complete.append(150)
        #         T_complete.append(t)
        #         skipped += 1

        return(R_complete[:Nb_pts_rot],T_complete[:Nb_pts_rot])

    def _get_obs(self):
        # changer ici pour obtenir tensor du lidar
        points = np.array([])
        while(points.size == 0):
            data = airsim.CarClient.getLidarData(self.car, lidar_name = "LidarSensor1", vehicle_name = "Car1")
            points = self.parse_lidarData(data)
            time.sleep(ACTION_SLEEP)
        R ,T = self.convert_lidar_data_to_polar(points)
        R ,T = self.data_format(R,T)
        #print(np.shape(R1))
        image = self.transform_obs(R)

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image
    
    def _compute_reward(self, obs):
        MAX_SPEED = 25
        MIN_SPEED = 1
        # THRESH_DIST = 1000
        ALPHA = 0
        BETA = 2
        GAMMA = 0.3
        
        car_pt = np.array((self.state["pose"].position.x_val, self.state["pose"].position.y_val))
        car_pt_prev = np.array((self.state["prev_pose"].position.x_val, self.state["prev_pose"].position.y_val))
        # print(car_pt)

        dist = np.linalg.norm(car_pt - car_pt_prev)

        reward_dist = abs(dist) * BETA
        reward_speed = (
            (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
        )*ALPHA
        reward_prox = np.min(obs)*GAMMA
        
        self.reward = reward_dist + reward_speed + reward_prox
        print(f"Reward : {reward_dist}; {reward_speed}; {reward_prox}\n\tTotal : {self.reward}")
        done = 0
        if dist == 0:
            done = 1
            self.reward = -2
            self.warning_stop = False
        if self.state["collision"]:
            self.reward = -5
            done = 1

        #print("Reward : ", self.reward)
        return self.reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        #print((obs==0).all())
        reward, done = self._compute_reward(obs)

        return(obs, reward, done, self.state)

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
