from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    ld = LaunchDescription()
    
    # 获取配置目录，所有对应的配置文件在bring_up包的config也有一份
    config_dir = get_package_share_directory('bring_up')
    
    # 1. 相机节点
    ascamera_node = Node(
        namespace= "ascamera_hp60c",
        package='ascamera',
        executable='ascamera_node',
        respawn=True,
        output='both',
        parameters=[
            {"usb_bus_no": -1},
            {"usb_path": "null"},
            {"confiPath": "/home/tanwei-nuc/ascam_ros2_ws/src/ascamera/configurationfiles"},
            {"color_pcl": False},
            {"pub_tfTree": True},
            {"depth_width": 640},
            {"depth_height": 480},
            {"rgb_width": 640},
            {"rgb_height": 480},
            {"fps": 25},
        ],
        remappings=[]
    )
    
    # 2. 点云校正节点
    pointcloud_correct_config = os.path.join(
        config_dir,
        'config',
        'pointcloud_correct.yaml'
    )
    
    pointcloud_correct_node = Node(
        package='pointcloud_correct',
        executable='pointcloud_correct_node',
        name='pointcloud_correct_node',
        output='screen',
        parameters=[pointcloud_correct_config],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # 3. 特征提取节点
    feature_extraction_config = os.path.join(
        config_dir,
        'config',
        'feature_extraction.yaml'
    )
    
    feature_extraction_node = Node(
        package='feature_extraction',
        executable='feature_extraction_node',
        name='feature_extraction_node',
        output='screen',
        parameters=[feature_extraction_config],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # 4. PLC通信节点
    plc_communication_config = os.path.join(
        config_dir,
        'config',
        'plc_communication.yaml'
    )
    
    plc_communication_node = Node(
        package='plc_communication',
        executable='plc_communication_node',
        name='plc_communication_node',
        output='screen',
        parameters=[plc_communication_config],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # 立即启动相机节点
    ld.add_action(ascamera_node)
    
    # 延迟5秒后启动点云校正节点
    ld.add_action(TimerAction(
        period=5.0,  
        actions=[pointcloud_correct_node]
    ))
    
    # 延迟7秒后启动特征提取节点
    ld.add_action(TimerAction(
        period=7.0,  
        actions=[feature_extraction_node]
    ))
    
    #延迟10秒后启动PLC通信节点
    ld.add_action(TimerAction(
        period=10.0,  
        actions=[plc_communication_node]
    ))
    
    return ld