# Python client example to get Lidar data from a car
#

from airsim.types import Vector3r
import setup_path 
import airsim

import sys
import math
import time
import argparse
import pprint
import numpy

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
        #self.client.simSetVehiclePose(self.reset_pose, True, "Car1")
        self.client.simSetObjectPose("Car1", self.reset_pose, True)
        self.collision_points = []

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
        points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
        points = numpy.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    def stop(self):

        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.handbrake = 1

        airsim.wait_key('Press any key to reset to original state')

        #self.client.simSetVehiclePose(self.reset_pose, True, 'Car1')
        self.client.simSetObjectPose("Car1", self.reset_pose, True)
        #print(self.reset_pose)
        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")

# main
if __name__ == "__main__":
    
    defaultPose = airsim.Pose()
    defaultPose.position.z_val = 0.5

    controller = CarControl(defaultPose)
    controller.forward(0.1)