# Mobile base drives with SLAM, no arm task (the plant holds the arm): the plant and
# its sensors, base_node (odometry, wheel commands), the chosen SLAM (own: slam_node;
# toolbox: slam_toolbox on the merged /scan), the scenario actor (a technician driving
# a route, or route:=nav: the navigation sent goals, people walking), people detection
# on the map (people:=0 to leave it out, unless navigating) and the monitors
# (validation). route:=nav adds nav_node and safety_node (goals:="pick place home",
# laps:=N; localization in a saved map). The arm starts tucked (arm_start:=home for the
# arm's home pose). slam:=own|toolbox|none,
# slam_mode:=mapping|localization, map:=<name in data/maps>, layout:=stations|cell,
# conditions:= plant conditions that make localization harder (plant_conditions.py),
# fusion:=ekf|icp the localization that drives (own SLAM, localization; both run).
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

MAP_DIR = os.environ.get("MANIPOPT_MAP_DIR", str(Path(__file__).resolve().parents[4] / "data" / "maps"))


def _slam(context):
    slam = LaunchConfiguration("slam").perform(context)
    mode = LaunchConfiguration("slam_mode").perform(context)
    name = LaunchConfiguration("map").perform(context)
    lossy = "dropout" in LaunchConfiguration("conditions").perform(context)
    if slam == "own":
        return [Node(package="pick_place_mpc", executable="slam_node", output="screen",
                     parameters=[{"mode": mode, "map": name, "map_dir": MAP_DIR,
                                  "fusion": LaunchConfiguration("fusion").perform(context)}],
                     remappings=[("/env/lidar_scan", "/env/lidar_scan_lossy")] if lossy else [])]
    if slam != "toolbox":
        return []
    config = Path(get_package_share_directory("pick_place_mpc")) / "config"
    nodes = [Node(package="pick_place_mpc", executable="scan_merger_node", output="screen")]
    if mode == "mapping":
        nodes.append(Node(package="slam_toolbox", executable="sync_slam_toolbox_node", name="slam_toolbox",
                          output="screen", parameters=[str(config / "slam_toolbox_mapping.yaml")]))
    else:
        nodes.append(Node(package="slam_toolbox", executable="localization_slam_toolbox_node", name="slam_toolbox",
                          output="screen", parameters=[str(config / "slam_toolbox_localization.yaml"),
                                                       {"map_file_name": str(Path(MAP_DIR) / name),
                                                        "map_start_pose": [0.0, 0.0, 0.0]}]))
    return nodes


def _people(context):
    nav = LaunchConfiguration("route").perform(context) == "nav"
    if int(LaunchConfiguration("people").perform(context)) == 0 and not nav:
        return []
    if LaunchConfiguration("slam_mode").perform(context) != "localization":
        return []
    name = LaunchConfiguration("map").perform(context)
    return [Node(package="pick_place_mpc", executable="people_node", output="screen",
                 parameters=[{"map": name, "map_dir": MAP_DIR}]),
            Node(package="pick_place_mpc", executable="people_monitor_node", output="screen")]


def _nav(context):
    if LaunchConfiguration("route").perform(context) != "nav":
        return []
    name = LaunchConfiguration("map").perform(context)
    return [Node(package="pick_place_mpc", executable="nav_node", output="screen",
                 parameters=[{"map": name, "map_dir": MAP_DIR}]),
            Node(package="pick_place_mpc", executable="safety_node", output="screen"),
            Node(package="pick_place_mpc", executable="nav_monitor_node", output="screen")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("slam", default_value="own"),  # own | toolbox | none
        DeclareLaunchArgument("slam_mode", default_value="mapping"),  # mapping | localization
        DeclareLaunchArgument("map", default_value="room"),
        DeclareLaunchArgument("route", default_value="loop"),  # loop | cross | nav | none
        DeclareLaunchArgument("goals", default_value="pick place home"),
        DeclareLaunchArgument("laps", default_value="1"),
        DeclareLaunchArgument("people", default_value="3"),
        DeclareLaunchArgument("mismatch_seed", default_value="0"),
        DeclareLaunchArgument("noise_seed", default_value="0"),
        DeclareLaunchArgument("cell_file", default_value="cell_container.xml"),
        DeclareLaunchArgument("layout", default_value="stations"),  # stations (the mobile job) | cell
        DeclareLaunchArgument("arm_start", default_value="tucked"),  # tucked (driving) | home
        DeclareLaunchArgument("conditions", default_value="none"),
        DeclareLaunchArgument("fusion", default_value="ekf"),
        Node(package="pick_place_common", executable="mujoco_sim_node", output="screen",
             parameters=[{"cell_file": LaunchConfiguration("cell_file"), "layout": LaunchConfiguration("layout"),
                          "arm_start": LaunchConfiguration("arm_start"), "conditions": LaunchConfiguration("conditions"),
                          "mismatch_seed": ParameterValue(LaunchConfiguration("mismatch_seed"), value_type=int),
                          "noise_seed": ParameterValue(LaunchConfiguration("noise_seed"), value_type=int)}]),
        Node(package="pick_place_common", executable="sim_sensors_node", output="screen",
             additional_env={"LP_NUM_THREADS": "2"},
             parameters=[{"cell_file": LaunchConfiguration("cell_file"), "layout": LaunchConfiguration("layout"),
                          "render": LaunchConfiguration("render"),
                          "visualize_pickup": False, "obstacle_view": False, "workspace_sensing": False,
                          "lidar": True, "conditions": LaunchConfiguration("conditions"),
                          "noise_seed": ParameterValue(LaunchConfiguration("noise_seed"), value_type=int)}]),
        Node(package="pick_place_mpc", executable="base_node", output="screen"),
        OpaqueFunction(function=_slam),
        Node(package="pick_place_mpc", executable="mobile_scenario_node", output="screen",
             parameters=[{"route": LaunchConfiguration("route"),
                          "laps": ParameterValue(LaunchConfiguration("laps"), value_type=int),
                          "people": ParameterValue(LaunchConfiguration("people"), value_type=int),
                          "goals": LaunchConfiguration("goals"),
                          "layout": LaunchConfiguration("layout")}]),
        OpaqueFunction(function=_people),
        OpaqueFunction(function=_nav),
        Node(package="pick_place_mpc", executable="localization_monitor_node", output="screen",
             parameters=[{"layout": LaunchConfiguration("layout")}]),
    ])
