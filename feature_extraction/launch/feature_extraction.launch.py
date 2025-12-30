from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('feature_extraction'),
        'config',
        'feature_extraction.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='feature_extraction',
            executable='feature_extraction_node',
            name='feature_extraction_node',
            output='screen',
            parameters=[config_file],
            # 优化节点资源分配
            ros_arguments=['--log-level', 'debug'],
            emulate_tty=True,
        )
    ])