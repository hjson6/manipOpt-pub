"""MoveIt method, full run: stack.launch.py plus decision_node.

Starts its own mujoco_sim_node (via the stack); never run it with another
launch against the same plant.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    stack = os.path.join(
        get_package_share_directory("pick_place_moveit"), "launch", "stack.launch.py")
    return LaunchDescription([
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(stack),
            launch_arguments={
                "render": LaunchConfiguration("render"),
                "visualize_pickup": LaunchConfiguration("visualize_pickup"),
            }.items(),
        ),
        Node(package="pick_place_moveit", executable="decision_node.py", output="screen"),
    ])
