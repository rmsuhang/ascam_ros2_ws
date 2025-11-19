from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('pointcloud_correct'),
        'config',
        'pointcloud_correct.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='pointcloud_correct',
            executable='pointcloud_correct_node',
            name='pointcloud_correct_node',
            output='screen',
            parameters=[config_file],
            # 优化节点资源分配
            ros_arguments=['--log-level', 'info'],
            emulate_tty=True,
        )
    ])