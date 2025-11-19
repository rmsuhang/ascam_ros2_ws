import os
from setuptools import find_packages, setup
from glob import glob


package_name = 'python_algorithms'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'scikit-learn',
        'yolov5'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='alwcmwttg@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'baffle_node = python_algorithms.baffle_node:main',
            'carriage_node = python_algorithms.carriage_node:main'
        ],
    },
)
