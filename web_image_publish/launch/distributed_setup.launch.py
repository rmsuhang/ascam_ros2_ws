from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 主服务器
        Node(
            package='web_image_publish',
            executable='video_stream_server',
            name='main_server',
            output='screen',
            parameters=[{
                'image_topic': '/ascamera_hp60c/camera_publisher/rgb0/image',
                'port': 8080,
                'thread_count': 8
            }]
        ),
        
        # # 代理服务器1
        # Node(
        #     package='web_image_publish',
        #     executable='stream_proxy',
        #     name='stream_proxy_1',
        #     output='screen',
        #     parameters=[{
        #         'source_topic': '/ascamera_hp60c/camera_publisher/rgb0/image',
        #         'local_port': 8081,
        #         'target_servers': ['192.168.89.149:8080', '192.168.1.102:8080']
        #     }]
        # ),
        
        # # 代理服务器2
        # Node(
        #     package='web_image_publish',
        #     executable='stream_proxy',
        #     name='stream_proxy_2',
        #     output='screen',
        #     parameters=[{
        #         'source_topic': '/ascamera_hp60c/camera_publisher/rgb0/image',
        #         'local_port': 8082,
        #         'target_servers': ['192.168.1.103:8080']
        #     }]
        # )
    ])