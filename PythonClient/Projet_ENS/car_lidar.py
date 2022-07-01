# Python client example to get Lidar data from a car
#

from turtle import forward

from torch import pi
import setup_path 
import airsim

import sys
import math
import time
import argparse
import pprint
import numpy

# Makes the drone fly and get Lidar data
class LidarTest:

    def __init__(self):

        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()

    def execute(self):

        for i in range(1):
           
            state = self.client.getCarState()
            s = pprint.pformat(state)
            #print("state: %s" % s)
            
            # go forward
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
            print("Go Forward")
            time.sleep(1)   # let car drive a bit
                    
            for i in range(1,2):
                lidarData = self.client.getLidarData()
                if (len(lidarData.point_cloud) < 3):
                    print("\tNo points received from Lidar data")
                else:
                    points = self.parse_lidarData(lidarData)
                    R,T = self.convert_lidar_data_to_polar(points)
                    R,T = self.data_format(R,T)
                    # print(R.shape)
                    # print("\tReading %d: time_stamp: %d number_of_points: %d" % (i, lidarData.time_stamp, len(T)))
                    # print("\t\tDistance: %s" % (pprint.pformat(T)))
                    # print("\t\tAngle: %s" % (pprint.pformat(R)))
                time.sleep(3)


    def parse_lidarData(self, data):    

        # reshape array of floats to array of [X,Y,Z]
        points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
        points = numpy.reshape(points, (int(points.shape[0]/3), 3))
       
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
        X=numpy.array(points[:2500,0])
        Y=numpy.array(points[:2500,1])
    
        R=numpy.sqrt(numpy.square(X)+numpy.square(Y))
        T=numpy.arctan2(Y,X)/pi*180
        
        # TODO
        # Could somebody add the 3rd dimension ?
        Sort = numpy.concatenate((R,T), axis=0)
        Sort = numpy.reshape(Sort, (len(R), 2))

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

        theta = numpy.linspace(FOV_min, FOV_max, Nb_pts_rot) 

        if (len(theta) != len(T)):

            # if(len(theta) > len(T)):
            #     R = numpy.append(R, numpy.zeros((len(theta) - len(R))))
            #     T = numpy.append(T, numpy.zeros((len(theta) - len(T))))
            #     # print("append", len(theta), len(R))
            # else :
            #     R = numpy.delete(R, [range(len(theta), len(R) - len(theta))])
            #     T = numpy.delete(T, [range(len(theta), len(T) - len(theta))])
            #     # print("delete", len(theta), len(R))

            # print("ok",len(T),len(R)) 
            # Bourage de 0 lorsqu'il manque des points
            for i in range(0,len(theta)):
                if(theta[i] <= T[i]):
                    numpy.insert(T, i-1, theta[i])
                    numpy.insert(R, i-1, 0)

            # print("Fin", len(R))
        return R[:Nb_pts_rot],T[:Nb_pts_rot]

    # def write_lidarData_to_disk(self, points):
    #     # TODO
    #     print("not yet implemented")

    def stop(self):
        airsim.wait_key('Press any key to reset to original state')

        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")

# main
if __name__ == "__main__":
    args = sys.argv
    args.pop(0)

    arg_parser = argparse.ArgumentParser("Lidar.py makes car move and gets Lidar data")

    arg_parser.add_argument('-save-to-disk', type=bool, help="save Lidar data to disk", default=False)
  
    args = arg_parser.parse_args(args)    
    lidarTest = LidarTest()
    try:
        lidarTest.execute()
    finally:
        lidarTest.stop()
