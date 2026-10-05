from setuptools import find_packages, setup

package_name = "pick_place_mpc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/demo.launch.py",
            "launch/demo_obstacle.launch.py",
            "launch/mobile.launch.py",
            "launch/mobile_job.launch.py",
        ]),
        (f"share/{package_name}/config", [
            "config/slam_toolbox_mapping.yaml",
            "config/slam_toolbox_localization.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hojin",
    maintainer_email="hojinsong95@gmail.com",
    description=(
        "Receding-horizon acados MPC method for the container "
        "pick-and-place task."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mpc_controller = pick_place_mpc.mpc_controller_node:main",
            "task_node = pick_place_mpc.task_node:main",
            "dynamic_obstacle_node = pick_place_mpc.dynamic_obstacle_node:main",
            "detection_monitor_node = pick_place_mpc.detection_monitor_node:main",
            "obstacle_supervisor_node = pick_place_mpc.obstacle_supervisor_node:main",
            "lidar_detection_node = pick_place_mpc.lidar_detection_node:main",
            "camera_detection_node = pick_place_mpc.camera_detection_node:main",
            "base_node = pick_place_mpc.base_node:main",
            "scan_merger_node = pick_place_mpc.scan_merger_node:main",
            "slam_node = pick_place_mpc.slam_node:main",
            "localization_monitor_node = pick_place_mpc.localization_monitor_node:main",
            "mobile_scenario_node = pick_place_mpc.mobile_scenario_node:main",
            "people_node = pick_place_mpc.people_node:main",
            "people_monitor_node = pick_place_mpc.people_monitor_node:main",
            "nav_node = pick_place_mpc.nav_node:main",
            "safety_node = pick_place_mpc.safety_node:main",
            "nav_monitor_node = pick_place_mpc.nav_monitor_node:main",
            "job_monitor_node = pick_place_mpc.job_monitor_node:main",
        ],
    },
)
