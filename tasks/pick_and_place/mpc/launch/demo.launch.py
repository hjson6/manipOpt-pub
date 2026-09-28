# MPC method: plant, task_node, mpc_controller. Starts its own
# mujoco_sim_node; never run it with another launch against the same plant
# (both would publish /sim/joint_command).
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        # Declared so `render:=false` is passed through (undeclared args are
        # silently ignored).
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        DeclareLaunchArgument("mismatch_seed", default_value="0"),  # plant_mismatch.py draw
        DeclareLaunchArgument("noise_seed", default_value="0"),  # depth_noise.py draw
        Node(
            package="pick_place_common",
            executable="mujoco_sim_node",
            output="screen",
            parameters=[{
                "scene_file": "panda_scene_container.xml",
                "render": LaunchConfiguration("render"),
                "visualize_pickup": LaunchConfiguration("visualize_pickup"),
                "mismatch_seed": ParameterValue(LaunchConfiguration("mismatch_seed"), value_type=int),
                "noise_seed": ParameterValue(LaunchConfiguration("noise_seed"), value_type=int),
            }],
        ),
        Node(package="pick_place_mpc", executable="task_node", output="screen"),
        Node(package="pick_place_mpc", executable="mpc_controller", output="screen"),
    ])
