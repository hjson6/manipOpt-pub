# The mobile pick and place (step 5): the plant in the stations layout (the arm starts in
# the carry pose at home), the arm's stack (task_node in the mobile job, mpc_controller,
# the obstacle supervisor on the people seen against the map, in the arm frame), the
# base's stack (odometry, our SLAM localizing in the saved map, people, navigation and
# its safety layer), people walking (people:=N, the job's crowd) and the monitors
# (validation). max_boxes:=N ends the job after N boxes (0: all); crowd:= picks the
# people's scenario (mobile_scenarios.CROWDS: job, walkers, dock_block, step_in); speed:= the
# base's top speed (m/s); base_safety:= / arm_safety:= scale the base's / the arm's distances
# to people; people_speed:= scales how fast the people walk; assumed_speed:= the walking speed
# the arm's safety assumes (m/s); conditions:= plant conditions that make localization harder
# ("spill worn_tyre gyro_drift dropout", plant_conditions.py); fusion:= the localization that
# drives (ekf: the filter, icp: the scan matcher alone; both run). Starts its own
# mujoco_sim_node: never run it with another launch against the same plant.
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

MAP_DIR = os.environ.get("MANIPOPT_MAP_DIR", str(Path(__file__).resolve().parents[4] / "data" / "maps"))


def _int(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _slam(context):
    lossy = "dropout" in LaunchConfiguration("conditions").perform(context)
    return [Node(package="pick_place_mpc", executable="slam_node", output="screen",
                 parameters=[{"mode": "localization", "map": LaunchConfiguration("map"), "map_dir": MAP_DIR,
                              "fusion": LaunchConfiguration("fusion")}],
                 remappings=[("/env/lidar_scan", "/env/lidar_scan_lossy")] if lossy else [])]


def generate_launch_description():
    plant = {"layout": "stations", "cell_file": LaunchConfiguration("cell_file"), "noise_seed": _int("noise_seed"),
             "conditions": LaunchConfiguration("conditions")}
    return LaunchDescription([
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        DeclareLaunchArgument("map", default_value="stations"),
        DeclareLaunchArgument("people", default_value="4"),
        DeclareLaunchArgument("crowd", default_value="job"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        DeclareLaunchArgument("base_safety", default_value="1.0"),
        DeclareLaunchArgument("arm_safety", default_value="1.0"),
        DeclareLaunchArgument("people_speed", default_value="1.0"),
        DeclareLaunchArgument("assumed_speed", default_value="1.6"),
        DeclareLaunchArgument("max_boxes", default_value="0"),
        DeclareLaunchArgument("mismatch_seed", default_value="0"),
        DeclareLaunchArgument("noise_seed", default_value="0"),
        DeclareLaunchArgument("cell_file", default_value="cell_container.xml"),
        DeclareLaunchArgument("sup_csv_path", default_value=""),
        DeclareLaunchArgument("conditions", default_value="none"),
        DeclareLaunchArgument("fusion", default_value="ekf"),
        Node(package="pick_place_common", executable="mujoco_sim_node", output="screen",
             parameters=[{**plant, "arm_start": "carry", "mismatch_seed": _int("mismatch_seed")}]),
        Node(package="pick_place_common", executable="sim_sensors_node", output="screen",
             additional_env={"LP_NUM_THREADS": "2"},
             parameters=[{**plant, "render": LaunchConfiguration("render"),
                          "visualize_pickup": LaunchConfiguration("visualize_pickup"),
                          "obstacle_view": False, "workspace_sensing": False, "lidar": True}]),
        # The base.
        Node(package="pick_place_mpc", executable="base_node", output="screen"),
        OpaqueFunction(function=_slam),
        Node(package="pick_place_mpc", executable="people_node", output="screen",
             parameters=[{"map": LaunchConfiguration("map"), "map_dir": MAP_DIR, "arm_detections": True}]),
        Node(package="pick_place_mpc", executable="nav_node", output="screen",
             parameters=[{"map": LaunchConfiguration("map"), "map_dir": MAP_DIR, "speed": _float("speed"),
                          "base_safety": _float("base_safety")}]),
        Node(package="pick_place_mpc", executable="safety_node", output="screen",
             parameters=[{"base_safety": _float("base_safety")}]),
        # The arm.
        Node(package="pick_place_mpc", executable="task_node", output="screen",
             parameters=[{"mobile": True, "max_boxes": _int("max_boxes")}]),
        Node(package="pick_place_mpc", executable="mpc_controller", output="screen"),
        Node(package="pick_place_mpc", executable="obstacle_supervisor_node", output="screen",
             parameters=[{"csv_path": LaunchConfiguration("sup_csv_path"), "lidar": True, "ceiling_camera": False,
                          "arm_safety": _float("arm_safety"), "human_speed": _float("assumed_speed")}]),
        # The environment and validation.
        Node(package="pick_place_mpc", executable="mobile_scenario_node", output="screen",
             parameters=[{"route": "none", "people": _int("people"), "layout": "stations", "wait_for_go": True,
                          "crowd": LaunchConfiguration("crowd"), "people_speed": _float("people_speed")}]),
        Node(package="pick_place_mpc", executable="people_monitor_node", output="screen"),
        Node(package="pick_place_mpc", executable="nav_monitor_node", output="screen"),
        Node(package="pick_place_mpc", executable="job_monitor_node", output="screen",
             parameters=[{"arm_safety": _float("arm_safety"), "assumed_speed": _float("assumed_speed")}]),
        Node(package="pick_place_mpc", executable="localization_monitor_node", output="screen",
             parameters=[{"layout": "stations"}]),
    ])
