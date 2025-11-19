# import os
# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     venv_path = '/root/code/venv'
#     log_dir_prefix = '/root/app/logs'

#     baffle_node_env = os.environ.copy()
#     baffle_node_name = 'baffle_node'
#     baffle_node_env.update({
#     'PATH': '{}/bin:{}'.format(venv_path, os.environ['PATH']),
#     'PYTHONPATH': '{}/lib/python3.10/site-packages:{}'.format(
#         venv_path, os.environ.get('PYTHONPATH', '')),
#     'ROS_LOG_DIR': os.path.join(log_dir_prefix, baffle_node_name),
#     'RCUTILS_LOGGING_USE_STDOUT': '0'
#     })


#     return LaunchDescription([
#         Node(
#             package="calibrate_point_cloud",
#             executable="calibrate_node",
#             name="calibrate_node"
#         ),
#         Node(
#             package="baffle_node",
#             executable="baffle_node",
#             name="baffle_node",
#             output='screen',
#             arguments=['--ros-args', '--log-level', 'info'],
#             env=baffle_node_env
#         ),
#         Node(
#             package="detection_bbox_node",
#             executable="detection_bbox",
#             name="detection_node"
#         )
#     ])



import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 使用你提供的conda虚拟环境路径
    venv_path = '/home/tanwei-nuc/anaconda3/envs/autoloading'
    
    # 创建合适的日志目录
    log_dir_prefix = '/home/tanwei-nuc/autoloading_ros_logs'
    os.makedirs(log_dir_prefix, exist_ok=True)

    # 检查虚拟环境是否存在
    if not os.path.exists(venv_path):
        raise Exception(f"虚拟环境路径不存在: {venv_path}")

    # 自动检测Python版本和site-packages路径
    python_bin_dir = os.path.join(venv_path, 'bin')
    site_packages_path = None
    
    # 查找lib目录中的Python版本
    lib_dir = os.path.join(venv_path, 'lib')
    if os.path.exists(lib_dir):
        for item in os.listdir(lib_dir):
            if item.startswith('python'):
                python_version_dir = os.path.join(lib_dir, item)
                potential_site_packages = os.path.join(python_version_dir, 'site-packages')
                if os.path.exists(potential_site_packages):
                    site_packages_path = potential_site_packages
                    break
    
    if site_packages_path is None:
        # 如果找不到标准的site-packages，尝试其他可能的位置
        potential_paths = [
            os.path.join(venv_path, 'lib', 'site-packages'),
            os.path.join(venv_path, 'lib', 'python3.10', 'site-packages'),
            os.path.join(venv_path, 'lib', 'python3.9', 'site-packages'),
            os.path.join(venv_path, 'lib', 'python3.8', 'site-packages'),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                site_packages_path = path
                break
    
    if site_packages_path is None:
        raise Exception(f"在虚拟环境中找不到site-packages目录: {venv_path}")

    print(f"使用虚拟环境: {venv_path}")
    print(f"使用site-packages: {site_packages_path}")

    # 创建baffle_node的环境配置
    baffle_node_env = os.environ.copy()
    baffle_node_name = 'baffle_node'
    baffle_node_env.update({
        'PATH': f'{python_bin_dir}:{os.environ["PATH"]}',
        'PYTHONPATH': f'{site_packages_path}:{os.environ.get("PYTHONPATH", "")}',
        'ROS_LOG_DIR': os.path.join(log_dir_prefix, baffle_node_name),
        'RCUTILS_LOGGING_USE_STDOUT': '0'
    })

    return LaunchDescription([
        Node(
            package="calibrate_point_cloud",
            executable="calibrate_node",
            name="calibrate_node",
            output='screen'  # 添加输出到屏幕以便调试
        ),
        Node(
            package="baffle_node",  # 确保这是正确的包名
            executable="baffle_node",
            name="baffle_node",
            output='screen',
            arguments=['--ros-args', '--log-level', 'info'],
            env=baffle_node_env
        ),
        Node(
            package="detection_bbox_node",
            executable="detection_bbox",
            name="detection_node",
            output='screen'  # 添加输出到屏幕以便调试
        )
    ])