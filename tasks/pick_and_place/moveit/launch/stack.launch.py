"""The MoveIt execution stack without a decision node: the plant, bridge_node,
move_group and mtc_executor_node. Used alone to drive the Pick, Place and
MoveTo actions by hand, and included by demo.launch.py.

Starts its own mujoco_sim_node and sim_sensors_node; never run it with another launch. move_group
needs the ExecuteTaskSolutionCapability, or every MTC execute fails. There
is no joint_state_publisher: bridge_node republishes /sim/joint_states as a
named /joint_states.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("panda_arm", package_name="pick_place_moveit")
        .robot_description(file_path="config/panda_arm.urdf")
        .robot_description_semantic(file_path="config/panda_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    return LaunchDescription([
        # Declared so `render:=false` is passed through.
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[moveit_config.robot_description],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"capabilities": "move_group/ExecuteTaskSolutionCapability"},
            ],
        ),
        Node(
            package="pick_place_common",
            executable="mujoco_sim_node",
            output="screen",
            parameters=[{
                "cell_file": "cell_container.xml",
            }],
        ),
        Node(
            package="pick_place_common",
            executable="sim_sensors_node",
            output="screen",
            parameters=[{
                "cell_file": "cell_container.xml",
                "render": LaunchConfiguration("render"),
                "visualize_pickup": LaunchConfiguration("visualize_pickup"),
            }],
        ),
        Node(package="pick_place_moveit", executable="bridge_node", output="screen"),
        Node(
            package="pick_place_moveit",
            executable="mtc_executor_node",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                moveit_config.planning_pipelines,
            ],
        ),
    ])
