"""Planning check: move_group and RViz only, against the URDF/SRDF in config/.
No plant or bridge, so "Plan" works in RViz but "Execute" does not.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
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

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # panda_link0 is fixed to the world at the MJCF origin (the SRDF's fixed
    # virtual joint).
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # Something must publish /joint_states for move_group to plan from. `zeros`
    # starts it at the SRDF "home" pose: the per-joint midpoint pose self-collides
    # (link5 and link7).
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[
            moveit_config.robot_description,
            {
                "zeros": {
                    "panda_joint1": 0.0,
                    "panda_joint2": 0.0,
                    "panda_joint3": 0.0,
                    "panda_joint4": -1.57079,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.57079,
                    "panda_joint7": -0.7853,
                }
            },
        ],
    )

    rviz_config_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "launch",
        "moveit.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
        ],
    )

    return LaunchDescription(
        [
            static_tf_node,
            joint_state_publisher_node,
            robot_state_publisher_node,
            move_group_node,
            rviz_node,
        ]
    )
