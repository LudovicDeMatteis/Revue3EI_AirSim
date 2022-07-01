# Python client example to get Lidar data from a car
#

from airsim.types import Vector3r
import airsim

import sys
import math
import time
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt

NB_POINT = 36

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

# Makes the drone fly and get Lidar data
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
        self.client.simSetObjectPose("Car1", self.reset_pose, True)
        self.collision_points = []
        self.reward = {"collision" : -50,
                        "continu" : 2}
        lidarData = self.client.getLidarData()
        self.reset_time = lidarData.time_stamp
        self.lidar_rotation_speed = 20*math.pi # 10 tr/s

    def change_control(self, steering, throttle):
        self.car_controls.steering = steering
        self.car_controls.throttle = throttle
        self.client.setCarControls(self.car_controls) 

    def forward(self, T):
        state = self.client.getCarState()
        print(f"State of the car is : {state}")
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

            time.sleep(T)

        self.stop()
            
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

        airsim.wait_key('Press any key to reset to original state')

        self.client.simSetVehiclePose(self.reset_pose, True, 'Car1')
        #self.client.simSetObjectPose("Car1", self.reset_pose, True)
        #print(self.reset_pose)
        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")

def step(controller, action):
    observation = None
    reward = 0
    done = False
    info = ""
    state = controller.client.getCarState()

    controller.car_controls.steering = action[0]
    controller.car_controls.throttle = action[1]
    controller.client.setCarControls(controller.car_controls)

    while(observation is None):
        lidarData = controller.client.getLidarData()
        if (len(lidarData.point_cloud) < 3):
            observation = None
        else:
            # Pose du Lidar
            # orientation = lidarData.pose.orientation
            # q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
            # rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
            #                                 [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
            #                                 [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
            # position = lidarData.pose.position
            # Orientation du Lidar en fonction des timestamp des mesures
            # TESTER hypothèse : le lidar est reset avec la voiture et pointe dans la même direction
            # delta_t = lidarData.time_stamp - controller.reset_time
            # angle = (delta_t * controller.lidar_rotation_speed)%(2*math.pi)
            # print("Delta t = {}; orientation = {}".format(delta_t, angle))
            # rotation_matrix_lidar = np.array([[math.cos(angle),-1*math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]) # TESTER différentes formes de matrices, pas sûr que celle ci fonctionne...
            # # Fin
            # observation = []
            # for i in range(0, len(lidarData.point_cloud), 3):
            #     xyz = lidarData.point_cloud[i:i+3]
            #     corrected_x, corrected_y, corrected_z = np.matmul(rotation_matrix, np.asarray(xyz))
            #     final_x = corrected_x + position.x_val
            #     final_y = corrected_y + position.y_val
            #     final_z = corrected_z + position.z_val
            #     observation.append(np.array([final_x, final_y, final_z]))
            #print("On trace le point correspondant à ", observation[0])
            #point = Vector3r(observation[0][0],observation[0][1],observation[0][2])
            #controller.client.simPlotPoints([point], [1.0, 0.0, 0.0, 1.0], 10.0, -1.0, True)
            observation = convert_lidar_data_to_polar(lidarData)
            observation = align_lidar(observation, NB_POINT)
    collision = controller.detect_collisions()
    if collision or (state.speed == 0.0 and action[1] == 0.0):
        reward += controller.reward["collision"]
        done = True
    else:
        reward += controller.reward["continu"]  # Il faudrait ajouter un reward en fonction de la distance parcourue vers l'objectif

    return(observation, reward, done, info)

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
    observation = observation[observation[:, 0].argsort()]
    angles_vises = np.linspace(-np.pi, np.pi, 37)[:-1]
    count = 0
    observation_complete = []
    for a in angles_vises:
        if(np.abs(a-observation[count, 0]) < np.pi/nb_points):
            observation_complete.append(observation[count])
            count += 1
        else:
            observation_complete.append(np.array([a,200]))
    return(observation_complete)

# main
if __name__ == "__main__":
    
    defaultPose = airsim.Pose()
    defaultPose.position.z_val = 0.5

    controller = CarControl(defaultPose)
    action = [0,0]
    done = False
    while(not(done)):  
        observation, reward, done, info = step(controller, action)
        print(reward)