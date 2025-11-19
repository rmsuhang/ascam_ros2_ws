from setuptools import find_packages, setup

package_name = 'baffle_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'torch',
        'torchvision',
        'ultralytics',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='13145337058@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
        'baffle_node = baffle_node.baffle_node:main',
    ],
    },
)
