from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kovan',
    maintainer_email='yusufhanali22@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'head_detector = robot_controller.head_detector.head_detector:main',
            'controller = robot_controller.robot_control.new_controller:main',
            'watchdog = robot_controller.robot_control.watchdog:main',
            'breathe_and_gazing = robot_controller.breathing_gazing.breathe_and_gazing:main',
            'fake_face_publisher = robot_controller.head_detector.fake_face_publisher:main',
            'head_mimic = robot_controller.experiment.head_mimic:main',
            'experiment_controller = robot_controller.experiment.exp_src:main',
        ],
    },
)
