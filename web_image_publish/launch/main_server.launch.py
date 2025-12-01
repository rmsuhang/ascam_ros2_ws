from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='web_image_publish',
            executable='video_stream_server',
            name='video_stream_server',
            output='screen',
            parameters=[{
                'image_topic': '/ascamera_hp60c/camera_publisher/rgb0/image',
                'port': 8080,
                'frame_rate': 30,
                'jpeg_quality': 80,
                'thread_count': 4
            }]
        )
    ])
