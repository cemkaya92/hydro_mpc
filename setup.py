from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hydro_mpc'

setup(
    name=package_name,
    version='0.2.1',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),  # ✅ This installs the marker
        ('share/' + package_name, ['package.xml']),
        # Include launch and config files
        (os.path.join('share', package_name, 'launch'), glob('hydro_mpc/launch/*.py')),
        (os.path.join('share', package_name, 'config', 'sitl'), glob('hydro_mpc/config/sitl/*.yaml')),
        (os.path.join('share', package_name, 'config', 'controller'), glob('hydro_mpc/config/controller/*.yaml')),
        (os.path.join('share', package_name, 'config', 'vehicle_parameters'), glob('hydro_mpc/config/vehicle_parameters/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Uluhan Cem Kaya',
    maintainer_email='uluhancem.kaya@uta.edu',
    description='ROS 2 PX4 control system with MPC and Offboard modes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_commander = hydro_mpc.control.motor_commander:main',
            'mpc_controller = hydro_mpc.control.mpc_controller:main',
            'offboard_manager_node = hydro_mpc.utils.offboard_manager_node:main',
            'gui_launcher = hydro_mpc.gui.gui_launcher:main'
        ],
    },
)
