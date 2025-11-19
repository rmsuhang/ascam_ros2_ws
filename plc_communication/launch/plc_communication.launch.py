from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('plc_communication'),
        'config',
        'plc_communication.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='plc_communication',
            executable='plc_communication_node',
            name='plc_communication_node',
            output='screen',
            parameters=[config_file],
            # 优化节点资源分配
            ros_arguments=['--log-level', 'info'],
            emulate_tty=True,
        )
    ])