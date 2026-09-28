# demo.launch.py plus the obstacle scenario: the environment actor
# (dynamic_obstacle_node, ground truth only), the detection monitor and the
# obstacle supervisor. `obstacle:=none` starts no actor (the no-false-positive
# run). Starts its own mujoco_sim_node; never run it with another launch.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _actor(context):
    mode = LaunchConfiguration("obstacle").perform(context)
    if mode == "none":
        return []
    return [Node(
        package="pick_place_mpc",
        executable="dynamic_obstacle_node",
        output="screen",
        parameters=[{
            "mode": mode,
            "trigger": LaunchConfiguration("trigger").perform(context),
            "delay_s": float(LaunchConfiguration("delay_s").perform(context)),
            "speed": float(LaunchConfiguration("speed").perform(context)),
            "passes": int(LaunchConfiguration("passes").perform(context)),
            "pause_s": float(LaunchConfiguration("pause_s").perform(context)),
            "dwell_s": float(LaunchConfiguration("dwell_s").perform(context)),
            "duration_s": float(LaunchConfiguration("duration_s").perform(context)),
            **({"static_position": [float(v) for v in pos.split(",")]}
               if (pos := LaunchConfiguration("static_position").perform(context)) else {}),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("render", default_value="true"),
        DeclareLaunchArgument("visualize_pickup", default_value="true"),
        DeclareLaunchArgument("mismatch_seed", default_value="0"),  # plant_mismatch.py draw
        DeclareLaunchArgument("noise_seed", default_value="0"),  # depth_noise.py draw
        DeclareLaunchArgument("obstacle", default_value="visit"),  # visit | walk | static | none
        DeclareLaunchArgument("trigger", default_value="first_pick"),  # launch | first_pick
        DeclareLaunchArgument("delay_s", default_value="3.0"),
        DeclareLaunchArgument("speed", default_value="0.55"),  # walking speed, m/s (half scale)
        DeclareLaunchArgument("passes", default_value="3"),  # visits or crossings; 0 = endless
        DeclareLaunchArgument("pause_s", default_value="15.0"),  # out of the cell between passes
        DeclareLaunchArgument("dwell_s", default_value="6.0"),  # visit: time standing at the tray
        DeclareLaunchArgument("static_position", default_value=""),  # "x,y,z" (static mode); empty: the default
        DeclareLaunchArgument("duration_s", default_value="0.0"),  # > 0: the obstacle leaves after this long
        DeclareLaunchArgument("csv_path", default_value=""),
        DeclareLaunchArgument("sup_csv_path", default_value=""),
        DeclareLaunchArgument("workspace_sensing", default_value="true"),
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
                "workspace_sensing": LaunchConfiguration("workspace_sensing"),
            }],
        ),
        Node(package="pick_place_mpc", executable="task_node", output="screen"),
        Node(package="pick_place_mpc", executable="mpc_controller", output="screen"),
        Node(
            package="pick_place_mpc",
            executable="detection_monitor_node",
            output="screen",
            parameters=[{"csv_path": LaunchConfiguration("csv_path")}],
        ),
        Node(
            package="pick_place_mpc",
            executable="obstacle_supervisor_node",
            output="screen",
            parameters=[{"csv_path": LaunchConfiguration("sup_csv_path")}],
        ),
        OpaqueFunction(function=_actor),
    ])
