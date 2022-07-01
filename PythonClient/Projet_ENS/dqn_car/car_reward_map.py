import airsim
import time
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ip_address="127.0.0.1"
port_number = 41452
image_shape=(200, 1, 1)

## Initialize the car
car = airsim.CarClient(ip=ip_address,port=port_number)

def parse_lidarData(data):    
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))
    return points

def convert_lidar_data_to_polar(points):
    pts = 200
    X=np.array(points[:pts,0])
    Y=np.array(points[:pts,1])

    R=np.sqrt(np.square(X)+np.square(Y))
    T=np.arctan2(Y,X)/np.pi*180

    Sort = np.concatenate((R,T), axis=0)
    Sort = np.reshape(Sort, (2,len(R))).T
    Sort = Sort[Sort[:, 1].argsort()]
    
    R = Sort[:,0]
    T = Sort[:,1]
    return(R ,T)

def data_format(R, T):
        FOV_min = -120
        FOV_max = 120
        Nb_pts_rot = 200

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
        return(R_complete[:Nb_pts_rot],T_complete[:Nb_pts_rot])

def _get_obs(car):
        # changer ici pour obtenir tensor du lidar
        points = np.array([])
        while(points.size == 0):
            data = airsim.CarClient.getLidarData(car, lidar_name = "LidarSensor1", vehicle_name = "Car1")
            points = parse_lidarData(data)
            time.sleep(0.05)
        R ,T = convert_lidar_data_to_polar(points)
        R ,T = data_format(R,T)
        return R

def _compute_reward(car, obs):

        GAMMA = 0.3
        
        reward_prox = np.min(obs)*GAMMA
        collision = car.simGetCollisionInfo().has_collided

        return reward_prox, collision

PAS = 2
reset_pose = airsim.Pose()
all_reward = np.zeros((int((160+60)/PAS),int((170+30)/PAS)))
line = np.zeros((1,int((170+30)/PAS)))
for x_i, x in enumerate(range(124,160,PAS)):
    for y_i, y in enumerate(range(-140,30,PAS)):
        reset_pose.position.x_val, reset_pose.position.y_val, reset_pose.position.z_val = x, y, -1  # Need to set the spawning position high enough so the car does not collide when spawning
        # reset_pose.orientation.w_val, reset_pose.orientation.x_val, reset_pose.orientation.y_val, reset_pose.orientation.z_val = reset_pose
        car.reset()
        car.enableApiControl(True)
        car.simSetVehiclePose(reset_pose, True, 'Car1')
        #car.armDisarm(True)
        time.sleep(0.5)
        obs = _get_obs(car)
        reward, done = _compute_reward(car, obs)
        if(done):
            all_reward[x_i, y_i] = -1
            line[0, y_i] = -1
        else:
            all_reward[x_i, y_i] = reward
            line[0, y_i] = reward
        print(reward, done)
        #time.sleep(0.5)
    np.save(f"reward_map2\\line{x_i}_x{x}_pas{PAS}.npy", line)
    time.sleep(5)
    line = np.zeros((1,int((170+30)/PAS)))



print(1)