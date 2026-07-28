from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    hri_overhead_camera_node = Node(
                package='realsense2_camera',
                namespace='hri',
                name='overhead',
                executable='realsense2_camera_node',
                parameters=['/home/kovan/USTA1.1/bringup/config/hri_overhead_rs_config.yaml'],
                output='screen'
            )
    
    yifan_wrist_camera_node = Node(
                package='realsense2_camera',
                namespace='yifan',
                name='wrist',
                executable='realsense2_camera_node',
                parameters=['/home/kovan/USTA1.1/bringup/config/yifan_wrist_rs_config.yaml'],
                output='screen'
            )
    
    return LaunchDescription([
        #hri_overhead_camera_node,
        yifan_wrist_camera_node,
    ])