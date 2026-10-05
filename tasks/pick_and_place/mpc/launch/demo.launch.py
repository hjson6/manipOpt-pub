# MPC method: plant, task_node, mpc_controller. Starts its own
# mujoco_sim_node; never run it with another launch against the same plant
# (both would publish /sim/joint_command).
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        # Declared so `render:=false` is passed through (undeclared args are
        # silently ignored).
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        DeclareLaunchArgument("obstacle_view", default_value="true"),  # the live people-sensing window
        DeclareLaunchArgument("lidar", default_value="true"),  # the safety lidars (people sensing)
        DeclareLaunchArgument("ceiling_camera", default_value="false"),  # the overhead camera (people sensing)
        DeclareLaunchArgument("plant_mismatch", default_value="true"),  # false: plant = controller's model
        DeclareLaunchArgument("mismatch_seed", default_value="0"),  # plant_mismatch.py draw
        DeclareLaunchArgument("noise_seed", default_value="0"),  # depth, encoder and load-cell noise draws
        DeclareLaunchArgument("sensor_noise", default_value="true"),  # encoder and load-cell noise
        DeclareLaunchArgument("lockstep", default_value="false"),  # true: the plant waits for each command
        DeclareLaunchArgument("cell_file", default_value="cell_container.xml"),  # the cell in the room
        DeclareLaunchArgument("tray_seed", default_value="-1"),  # >= 0: tray shifted up to 5 cm
        Node(
            package="pick_place_common",
            executable="mujoco_sim_node",
            output="screen",
            parameters=[{
                "cell_file": LaunchConfiguration("cell_file"),
                "tray_seed": ParameterValue(LaunchConfiguration("tray_seed"), value_type=int),
                "plant_mismatch": ParameterValue(LaunchConfiguration("plant_mismatch"), value_type=bool),
                "mismatch_seed": ParameterValue(LaunchConfiguration("mismatch_seed"), value_type=int),
                "noise_seed": ParameterValue(LaunchConfiguration("noise_seed"), value_type=int),
                "sensor_noise": ParameterValue(LaunchConfiguration("sensor_noise"), value_type=bool),
                "lockstep": ParameterValue(LaunchConfiguration("lockstep"), value_type=bool),
            }],
        ),
        # Cameras and windows, in their own process (drawing must not delay the physics).
        Node(
            package="pick_place_common",
            executable="sim_sensors_node",
            output="screen",
            # Software GL (llvmpipe, WSL) spreads each frame over every core by default,
            # starving the plant and the controller.
            additional_env={"LP_NUM_THREADS": "2"},
            parameters=[{
                "cell_file": LaunchConfiguration("cell_file"),
                "tray_seed": ParameterValue(LaunchConfiguration("tray_seed"), value_type=int),
                "render": LaunchConfiguration("render"),
                "visualize_pickup": LaunchConfiguration("visualize_pickup"),
                "obstacle_view": ParameterValue(LaunchConfiguration("obstacle_view"), value_type=bool),
                "noise_seed": ParameterValue(LaunchConfiguration("noise_seed"), value_type=int),
                "workspace_sensing": ParameterValue(LaunchConfiguration("ceiling_camera"), value_type=bool),
                "lidar": ParameterValue(LaunchConfiguration("lidar"), value_type=bool),
            }],
        ),
        Node(package="pick_place_mpc", executable="lidar_detection_node", output="screen",
             condition=IfCondition(LaunchConfiguration("lidar"))),
        Node(package="pick_place_mpc", executable="camera_detection_node", output="screen",
             condition=IfCondition(LaunchConfiguration("ceiling_camera"))),
        Node(package="pick_place_mpc", executable="base_node", output="screen"),  # odometry, wheel commands
        Node(package="pick_place_mpc", executable="task_node", output="screen"),
        Node(package="pick_place_mpc", executable="mpc_controller", output="screen",
             parameters=[{"lockstep": ParameterValue(LaunchConfiguration("lockstep"), value_type=bool)}]),
    ])
