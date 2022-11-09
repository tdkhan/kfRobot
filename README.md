# kfRobot
This repsitory contains ROS packages for fusing IMU and camera data using a Kalman filter to localize a TurtleBot. The exercise was done as part of the final project for the Robotic Perception course taught at Texas A&M University.

This code is developed with Ros Noetic and Gazebo 11.8.1. 

## Package Dependencies
The TurtleBot packages required for this simulation are the turtlbot3 and turtlebot3_msgs. In the src folder of your workspace clone the following packages
```
cd ~/<your_workspace>/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
```

## Run Instructions (TurtleBot in Simulation Environment)
Copy the package sim_environment in the src folder of your workspace form this repository. Then do:
```
cd ~/<your_workspace>
catkin_make
```

To source the catkin workspace:
```
source devel/setup.bash
```

To spawn the ArUco markers in scene make sure that 'GAZEBO_MODEL_PATH' is set to the models directory in the sim_environment pacakge. To run the TurtleBot in the simulation environment first set the environmental variable 'TURTLBOT3_MODEL' to 'waffle_pi' and then launch the world file
```
export GAZEBO_MODEL_PATH=<path to your workspace>/src/sim_environment/models
export TURTLEBOT3_MODEL=waffle_pi
roslaunch sim_environment world.launch
```

To drive the TurtleBot in the world, launch the turtlebot3_teleop_key node in another terminal
```
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch 
```

Drive the robot in the world using your keyboard.


