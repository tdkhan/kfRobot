# kfRobot
This repsitory contains ROS packages for fusing IMU and camera data using a Kalman filter to localize a TurtleBot. The exercise was done as part of the final project for the Robotic Perception course taught at Texas A&M University.

This code is developed with Ros Noetic and Gazebo 11.8.1. 

## Running Instructions (Simulation Environment)
Copy the package sim_environment in the src folder of your workspace form this repository. Then do:
```
cd ~/your_workspace
catkin_make
```

To source the catkin workspace:
```
source devel/setup.bash
```

To view the simulation environment for the robot:
```
roslaunch sim_environment world.launch
```


