# demo.launch.py plus the obstacle scenario: the environment actor
# (dynamic_obstacle_node, ground truth only), the detection monitor and the
# obstacle supervisor. People sensing: lidar (default) and/or the ceiling
# camera (`lidar:=`, `ceiling_camera:=`). `obstacle:=none` starts no actor (the no-false-positive
# run). Starts its own mujoco_sim_node; never run it with another launch.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _actor(context):
    mode = LaunchConfiguration("obstacle").perform(context)
    if mode == "static" and LaunchConfiguration("ceiling_camera").perform(context).lower() != "true":
        raise RuntimeError("obstacle:=static needs ceiling_camera:=true (the lidar cannot tell a static "
                           "object; the detour planner works from the camera)")
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
        DeclareLaunchArgument("obstacle", default_value="visit"),  # visit | walk | static | none
        DeclareLaunchArgument("trigger", default_value="first_pick"),  # launch | first_pick
        DeclareLaunchArgument("delay_s", default_value="3.0"),
        DeclareLaunchArgument("speed", default_value="1.2"),  # walking speed, m/s
        DeclareLaunchArgument("passes", default_value="3"),  # visits or crossings; 0 = endless
        DeclareLaunchArgument("pause_s", default_value="15.0"),  # out of the cell between passes
        DeclareLaunchArgument("dwell_s", default_value="6.0"),  # visit: time standing at the tray
        DeclareLaunchArgument("static_position", default_value=""),  # "x,y,z" (static mode); empty: the default
        DeclareLaunchArgument("duration_s", default_value="0.0"),  # > 0: the obstacle leaves after this long
        DeclareLaunchArgument("csv_path", default_value=""),
        DeclareLaunchArgument("sup_csv_path", default_value=""),
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
            parameters=[{"csv_path": LaunchConfiguration("sup_csv_path"),
                         "lidar": ParameterValue(LaunchConfiguration("lidar"), value_type=bool),
                         "ceiling_camera": ParameterValue(LaunchConfiguration("ceiling_camera"), value_type=bool)}],
        ),
        OpaqueFunction(function=_actor),
    ])
