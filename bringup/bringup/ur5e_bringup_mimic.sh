#!/bin/bash

SESSION_UR5E_BRINGUP="ur5e_bringup"

# Check if the tmux session already exists
tmux has-session -t $SESSION_UR5E_BRINGUP 2>/dev/null

if [ $? != 0 ]; then
    # Create a new tmux session and run the commands in separate panes
    tmux new-session -d -s $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "export PYTHONPATH=\"\${PYTHONPATH}:/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/\"" C-m    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "cd ~/USTA1.1/ros_workspace/" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "source install/setup.bash" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 run robot_controller head_mimic"
    
    tmux split-window -v -t $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "cd ~/mocap4r2_ws/" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "source install/setup.bash" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py" C-m
    sleep 2

    tmux split-window -h -t $SESSION_UR5E_BRINGUP

    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 lifecycle set /mocap4r2_optitrack_driver_node activate" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 launch ur_robot_driver ur5e.launch.py ur_type:=ur5e robot_ip:=192.168.1.102 kinematics_params_file:=\"/home/kovan/USTA1.1/bringup/config/ur5e_calibration.yaml\"" C-m
    sleep 2
    
    tmux split-window -h -t $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "cd ~/USTA1.1/ros_workspace/" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "source install/setup.bash" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "colcon build"
    
    tmux split-window -h -t $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "cd ~/USTA1.1/ros_workspace/" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "source install/setup.bash" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 run static_transforms static_tf_broadcaster" C-m 
    
    tmux resize-pane -t 0 -y 95% 

    tmux select-pane -t 0
    tmux split-window -h -t $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "export PYTHONPATH=\"\${PYTHONPATH}:/home/kovan/USTA1.1/ros_workspace/src/robot_controller/robot_controller/\"" C-m    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "cd ~/USTA1.1/ros_workspace/" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "source install/setup.bash" C-m
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 run robot_controller watchdog" C-m

    tmux split-window -h -t $SESSION_UR5E_BRINGUP

    tmux send-keys -t $SESSION_UR5E_BRINGUP "a"

    tmux split-window -v -t $SESSION_UR5E_BRINGUP
    
    tmux send-keys -t $SESSION_UR5E_BRINGUP "ros2 run realsense2_camera realsense2_camera_node --ros-args --params-file /home/kovan/USTA1.1/bringup/config/rs_config.yaml" C-m

    tmux select-pane -t 1

    tmux split-window -v -t $SESSION_UR5E_BRINGUP

    tmux send-keys -t $SESSION_UR5E_BRINGUP "c"

    tmux resize-pane -t 2 -y 90%
    tmux resize-pane -t 4 -y 90%
    
fi
# Attach to the tmux session
tmux set -g mouse on 

tmux bind -n C-w kill-session -t $SESSION_UR5E_BRINGUP

tmux attach-session -t $SESSION_UR5E_BRINGUP
