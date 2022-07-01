import airsim

QUAT_90 = [0.707,0,0,0.707]
QUAT_180 = [0,0,0,1]
QUAT_270 = [-0.707,0,0,0.707]

pose1, pose2, pose3, pose4, pose5, pose6, pose7, pose8 = [airsim.Pose() for i in range(8)]
pose1.position.x_val, pose1.position.y_val, pose1.position.z_val = 0, 0, 0.5

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

pos_initiales  = [pose1, pose2, pose3, pose4, pose5]