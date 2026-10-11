cmake_minimum_required(VERSION 3.8)
project(pick_place_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(control_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
# Not an ament package: conda-forge's pinocchio, the one core/dynamics.py uses.
find_package(pinocchio REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(pick_place_interfaces REQUIRED)

add_executable(bridge_node src/bridge_node.cpp)
ament_target_dependencies(bridge_node
  rclcpp rclcpp_action ament_index_cpp control_msgs sensor_msgs trajectory_msgs)
target_link_libraries(bridge_node pinocchio::pinocchio)

add_executable(mtc_executor_node src/mtc_executor_node.cpp)
ament_target_dependencies(mtc_executor_node
  rclcpp rclcpp_action geometry_msgs shape_msgs std_msgs moveit_core
  moveit_ros_planning_interface moveit_task_constructor_core pick_place_interfaces)

install(TARGETS bridge_node mtc_executor_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/decision_node.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
