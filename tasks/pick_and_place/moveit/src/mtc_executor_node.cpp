// mtc_executor_node: serves the Pick, Place and MoveTo actions that
// decision_node sequences:
//   MoveTo(PARK) -> [pile scan] -> Pick -> MoveTo(DEST_SCAN) -> [tray scan] ->
//   Place -> ...
// Each action plans and executes MoveIt Task Constructor (MTC) stage graphs.
//
// Grasping is a rigid attach (no hand in the SRDF), so MTC's Pick/Place
// stages, which require an <end_effector> group, are not used; the graphs
// follow the MTC pick-and-place demo's stage sequence by hand, with
// GeneratePose for the fixed top-down grasp.
//
// MTC's attach/detach changes only the planning scene; the physical grasp is
// mujoco_sim_node's weld, triggered by "pick_at" / "place_held" on
// /task/action. task.execute() has no mid-execution hook, so each action is
// split into tasks at the physical event, published in between:
//   Pick:  "approach_and_grasp" -> pick_at -> "lift" (attach, lift)
//   Place: "transport_and_lower" -> place_held -> "retreat" (detach, retreat)

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "geometry_msgs/msg/vector3.hpp"
#include "moveit/planning_scene_interface/planning_scene_interface.h"
#include "moveit/task_constructor/solvers/cartesian_path.h"
#include "moveit/task_constructor/solvers/pipeline_planner.h"
#include "moveit/task_constructor/stages/compute_ik.h"
#include "moveit/task_constructor/stages/connect.h"
#include "moveit/task_constructor/stages/current_state.h"
#include "moveit/task_constructor/stages/generate_place_pose.h"
#include "moveit/task_constructor/stages/generate_pose.h"
#include "moveit/task_constructor/stages/modify_planning_scene.h"
#include "moveit/task_constructor/stages/move_relative.h"
#include "moveit/task_constructor/stages/move_to.h"
#include "moveit/task_constructor/task.h"
#include "pick_place_interfaces/action/move_to.hpp"
#include "pick_place_interfaces/action/pick.hpp"
#include "pick_place_interfaces/action/place.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"
#include "std_msgs/msg/string.hpp"

namespace
{

namespace mtc = moveit::task_constructor;

constexpr char kArmGroup[] = "panda_arm";
constexpr char kTcpLink[] = "panda_tcp";
constexpr char kAttachLink[] = "panda_link8";  // mujoco_sim_node's EE_BODY_NAME ("attachment")
constexpr char kWorldFrame[] = "world";

constexpr double kApproachMinM = 0.04;
constexpr double kApproachMaxM = 0.15;
constexpr double kLiftMinM = 0.04;
constexpr double kLiftMaxM = 0.15;
constexpr double kRetreatMinM = 0.04;
constexpr double kRetreatMaxM = 0.15;
constexpr double kConnectTimeoutS = 10.0;
constexpr int kMaxIkSolutions = 8;
constexpr double kBoxCollisionShrinkM = 0.003;  // see AddBoxCollisionObject
// Plans slowed to 30%: time-optimal ones were faster than bridge_node could
// settle within tolerance.
constexpr double kVelocityScaling = 0.3;
constexpr double kAccelerationScaling = 0.3;

// TCP z along world -z, x along world +x: 180 deg about x. Any yaw would do.
geometry_msgs::msg::Quaternion TopDownOrientation()
{
  geometry_msgs::msg::Quaternion q;
  q.x = 1.0;
  q.y = 0.0;
  q.z = 0.0;
  q.w = 0.0;
  return q;
}

geometry_msgs::msg::Vector3Stamped WorldDirection(double z)
{
  geometry_msgs::msg::Vector3Stamped v;
  v.header.frame_id = kWorldFrame;
  v.vector.z = z;
  return v;
}

}  // namespace

class MtcExecutorNode : public rclcpp::Node
{
public:
  using Pick = pick_place_interfaces::action::Pick;
  using Place = pick_place_interfaces::action::Place;
  using MoveTo = pick_place_interfaces::action::MoveTo;

  MtcExecutorNode()
  : Node("mtc_executor_node")
  {
    task_action_pub_ = create_publisher<std_msgs::msg::String>("/task/action", 10);
    grasped_size_sub_ = create_subscription<geometry_msgs::msg::Vector3>(
      "/sim/grasped_box_size", 10,
      [this](const geometry_msgs::msg::Vector3::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(grasped_size_mutex_);
        latest_grasped_size_ = *msg;
        have_grasped_size_ = true;
      });

    pick_server_ = Serve<Pick>(
      "pick", [this](const auto & gh, auto & result) {ExecutePick(gh, result);});
    place_server_ = Serve<Place>(
      "place", [this](const auto & gh, auto & result) {ExecutePlace(gh, result);});
    move_to_server_ = Serve<MoveTo>(
      "move_to", [this](const auto & gh, auto & result) {ExecuteMoveTo(gh, result);});
  }

private:
  template<typename ActionT>
  using GoalHandlePtr = std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionT>>;

  // One action at a time across all three servers (one arm, one held box); an
  // overlapping goal is rejected. The body runs on a detached thread and throws
  // std::runtime_error to abort with a reason.
  template<typename ActionT>
  typename rclcpp_action::Server<ActionT>::SharedPtr Serve(
    const std::string & name,
    std::function<void(const GoalHandlePtr<ActionT> &, typename ActionT::Result &)> body)
  {
    return rclcpp_action::create_server<ActionT>(
      this, name,
      [this, name](const rclcpp_action::GoalUUID &, std::shared_ptr<const typename ActionT::Goal>) {
        bool expected = false;
        if (!busy_.compare_exchange_strong(expected, true)) {
          RCLCPP_WARN(get_logger(), "rejecting %s goal: another action is running", name.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [](const GoalHandlePtr<ActionT> &) {return rclcpp_action::CancelResponse::REJECT;},
      [this, name, body](const GoalHandlePtr<ActionT> gh) {
        std::thread([this, name, body, gh]() {
          auto result = std::make_shared<typename ActionT::Result>();
          try {
            body(gh, *result);
            result->success = true;
            gh->succeed(result);
            RCLCPP_INFO(get_logger(), "%s succeeded", name.c_str());
          } catch (const std::exception & e) {
            result->success = false;
            result->failure_reason = e.what();
            RCLCPP_ERROR(get_logger(), "%s aborted: %s", name.c_str(), e.what());
            gh->abort(result);
          }
          busy_ = false;
        }).detach();
      });
  }

  template<typename ActionT>
  void Feedback(const GoalHandlePtr<ActionT> & gh, const std::string & stage)
  {
    auto fb = std::make_shared<typename ActionT::Feedback>();
    fb->stage = stage;
    gh->publish_feedback(fb);
    RCLCPP_INFO(get_logger(), "stage: %s", stage.c_str());
  }

  // ---------------------------------------------------------------- actions

  void ExecutePick(const GoalHandlePtr<Pick> & gh, Pick::Result & result)
  {
    if (!held_box_id_.empty()) {
      throw std::runtime_error("already holding '" + held_box_id_ + "'; Place it first");
    }
    const auto goal = gh->get_goal();
    // A new id per box: placed boxes stay in the planning scene.
    const std::string id = "box_" + std::to_string(next_box_index_++);

    // Height is unknown before the grasp; corrected once the size arrives.
    const double placeholder_height = 0.5 * std::max(goal->footprint_size.x, goal->footprint_size.y);
    AddBoxCollisionObject(
      id, goal->top_surface_point, goal->footprint_size.x, goal->footprint_size.y,
      placeholder_height);

    Feedback<Pick>(gh, "approach_and_grasp");
    try {
      auto task = BuildApproachTask(goal->top_surface_point);
      PlanAndExecute(task);
    } catch (...) {
      RemoveCollisionObject(id);  // never grasped: don't leave a guessed-height object
      throw;
    }

    Feedback<Pick>(gh, "grasp");
    {
      std::lock_guard<std::mutex> lock(grasped_size_mutex_);
      have_grasped_size_ = false;
    }
    {
      std::ostringstream os;
      os << "pick_at " << goal->top_surface_point.x << " " << goal->top_surface_point.y << " "
         << goal->top_surface_point.z;
      PublishTaskAction(os.str());
    }
    // Half-extents (mujoco geom size).
    const auto grasped_size = WaitForGraspedBoxSize();
    // Welded from here on: record it as held first, so a later Place still works.
    held_box_id_ = id;
    held_half_height_ = grasped_size.z;
    result.grasped_box_size = grasped_size;
    AddBoxCollisionObject(
      id, goal->top_surface_point, 2.0 * grasped_size.x, 2.0 * grasped_size.y,
      2.0 * grasped_size.z);

    Feedback<Pick>(gh, "lift");
    auto task = BuildLiftTask(id);
    PlanAndExecute(task);
  }

  void ExecutePlace(const GoalHandlePtr<Place> & gh, Place::Result &)
  {
    if (held_box_id_.empty()) {
      throw std::runtime_error("nothing held; Pick first");
    }
    const auto goal = gh->get_goal();
    geometry_msgs::msg::Point center = goal->surface_point;
    center.z += held_half_height_;  // surface point -> box-centre target

    Feedback<Place>(gh, "transport_and_lower");
    {
      auto task = BuildPlaceTask(held_box_id_, center);
      PlanAndExecute(task);
    }

    Feedback<Place>(gh, "release");
    PublishTaskAction("place_held");
    const std::string id = held_box_id_;
    held_box_id_.clear();

    Feedback<Place>(gh, "retreat");
    auto task = BuildRetreatTask(id);
    PlanAndExecute(task);
  }

  void ExecuteMoveTo(const GoalHandlePtr<MoveTo> & gh, MoveTo::Result &)
  {
    Feedback<MoveTo>(gh, "move_to");
    auto task = BuildMoveToTask(gh->get_goal()->tcp_position);
    PlanAndExecute(task);
  }

  // ------------------------------------------------------- planning scene

  // Add or replace a box hanging down from the sensed top point (attach needs a
  // named object in the scene). Called twice per Pick: with a placeholder
  // height, then with the real size, since the attach offset comes from this
  // geometry and a wrong height would place the box that far off.
  void AddBoxCollisionObject(
    const std::string & id, const geometry_msgs::msg::Point & top_point, double size_x,
    double size_y, double height)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = kWorldFrame;
    obj.id = id;
    obj.primitives.resize(1);
    obj.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    // Shrunk by kBoxCollisionShrinkM per side about the same centre: boxes placed
    // flush would otherwise count as touching, a collision, and fail the IK.
    obj.primitives[0].dimensions = {
      std::max(size_x - 2 * kBoxCollisionShrinkM, 0.001),
      std::max(size_y - 2 * kBoxCollisionShrinkM, 0.001),
      std::max(height - 2 * kBoxCollisionShrinkM, 0.001),
    };
    geometry_msgs::msg::Pose pose;
    pose.position.x = top_point.x;
    pose.position.y = top_point.y;
    pose.position.z = top_point.z - 0.5 * height;
    pose.orientation.w = 1.0;
    obj.primitive_poses.push_back(pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    if (!psi_.applyCollisionObject(obj)) {
      throw std::runtime_error("failed to add collision object '" + id + "'");
    }
  }

  void RemoveCollisionObject(const std::string & id)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = kWorldFrame;
    obj.id = id;
    obj.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    psi_.applyCollisionObject(obj);
  }

  // ------------------------------------------------------------- planners

  std::shared_ptr<mtc::solvers::PipelinePlanner> SamplingPlanner()
  {
    auto planner = std::make_shared<mtc::solvers::PipelinePlanner>(shared_from_this(), "ompl");
    planner->setProperty("goal_joint_tolerance", 1e-4);
    planner->setMaxVelocityScalingFactor(kVelocityScaling);
    planner->setMaxAccelerationScalingFactor(kAccelerationScaling);
    return planner;
  }

  std::shared_ptr<mtc::solvers::CartesianPath> CartesianPlanner()
  {
    auto planner = std::make_shared<mtc::solvers::CartesianPath>();
    planner->setMaxVelocityScalingFactor(kVelocityScaling);
    planner->setMaxAccelerationScalingFactor(kAccelerationScaling);
    planner->setStepSize(0.01);
    return planner;
  }

  mtc::Task NewTask(const std::string & name)
  {
    mtc::Task task;
    task.setName(name);
    task.loadRobotModel(shared_from_this());
    task.setProperty("group", std::string(kArmGroup));
    task.setProperty("ik_frame", std::string(kTcpLink));
    return task;
  }

  // ---------------------------------------------------------------- tasks

  // current state -> connect -> [approach, grasp-pose IK]; ends at grasp
  // contact, nothing attached.
  mtc::Task BuildApproachTask(const geometry_msgs::msg::Point & grasp_point)
  {
    auto task = NewTask("approach_and_grasp");
    auto sampling_planner = SamplingPlanner();
    auto cartesian_planner = CartesianPlanner();

    mtc::Stage * current_state_ptr = nullptr;
    {
      auto stage = std::make_unique<mtc::stages::CurrentState>("current state");
      current_state_ptr = stage.get();
      task.add(std::move(stage));
    }
    task.add(MakeConnect("move to pick", sampling_planner));
    {
      auto container = std::make_unique<mtc::SerialContainer>("grasp");
      task.properties().exposeTo(container->properties(), {"group", "ik_frame"});
      container->properties().configureInitFrom(mtc::Stage::PARENT, {"group", "ik_frame"});
      {
        auto stage = std::make_unique<mtc::stages::MoveRelative>("approach", cartesian_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        stage->setIKFrame(kTcpLink);
        stage->setMinMaxDistance(kApproachMinM, kApproachMaxM);
        // Along the TCP's +z, straight down with the top-down grasp.
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = kTcpLink;
        vec.vector.z = 1.0;
        stage->setDirection(vec);
        container->insert(std::move(stage));
      }
      {
        geometry_msgs::msg::PoseStamped target;
        target.header.frame_id = kWorldFrame;
        target.pose.position = grasp_point;
        target.pose.orientation = TopDownOrientation();

        auto generator = std::make_unique<mtc::stages::GeneratePose>("grasp pose");
        generator->properties().configureInitFrom(mtc::Stage::PARENT);
        generator->setPose(target);
        generator->setMonitoredStage(current_state_ptr);

        auto wrapper = std::make_unique<mtc::stages::ComputeIK>("grasp pose IK", std::move(generator));
        wrapper->setGroup(kArmGroup);
        wrapper->setIKFrame(kTcpLink);
        wrapper->setMaxIKSolutions(kMaxIkSolutions);
        wrapper->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
        container->insert(std::move(wrapper));
      }
      task.add(std::move(container));
    }
    return task;
  }

  // current state -> attach (planning scene only) -> lift.
  mtc::Task BuildLiftTask(const std::string & box_id)
  {
    auto task = NewTask("lift");
    task.add(std::make_unique<mtc::stages::CurrentState>("current state"));
    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach box");
      stage->attachObject(box_id, kAttachLink);
      task.add(std::move(stage));
    }
    task.add(MakeVerticalMove("lift", kLiftMinM, kLiftMaxM, 1.0));
    return task;
  }

  // current state (box attached) -> connect -> [lower, place-pose IK]; ends at
  // the place pose.
  mtc::Task BuildPlaceTask(const std::string & box_id, const geometry_msgs::msg::Point & box_center)
  {
    auto task = NewTask("transport_and_lower");
    auto sampling_planner = SamplingPlanner();
    auto cartesian_planner = CartesianPlanner();

    // GeneratePlacePose needs the box attached in its monitored stage's scene;
    // the attach from Pick's lift task persists.
    mtc::Stage * current_state_ptr = nullptr;
    {
      auto stage = std::make_unique<mtc::stages::CurrentState>("current state");
      current_state_ptr = stage.get();
      task.add(std::move(stage));
    }
    task.add(MakeConnect("move to place", sampling_planner));
    {
      auto container = std::make_unique<mtc::SerialContainer>("place");
      task.properties().exposeTo(container->properties(), {"group", "ik_frame"});
      container->properties().configureInitFrom(mtc::Stage::PARENT, {"group", "ik_frame"});
      {
        auto stage = std::make_unique<mtc::stages::MoveRelative>("lower", cartesian_planner);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        stage->setIKFrame(kTcpLink);
        stage->setMinMaxDistance(kApproachMinM, kApproachMaxM);
        stage->setDirection(WorldDirection(-1.0));
        container->insert(std::move(stage));
      }
      {
        // The box's target pose (upright), not the tool's: GeneratePlacePose derives
        // the tool target from the attach offset.
        geometry_msgs::msg::PoseStamped target;
        target.header.frame_id = kWorldFrame;
        target.pose.position = box_center;
        target.pose.orientation.w = 1.0;

        auto generator = std::make_unique<mtc::stages::GeneratePlacePose>("place pose");
        generator->properties().configureInitFrom(mtc::Stage::PARENT, {"ik_frame"});
        generator->setObject(box_id);
        generator->setPose(target);
        generator->setMonitoredStage(current_state_ptr);

        // ik_frame comes from the interface (the attached object). Forcing panda_tcp
        // put the tool pointing up under the box.
        auto wrapper = std::make_unique<mtc::stages::ComputeIK>("place pose IK", std::move(generator));
        wrapper->setGroup(kArmGroup);
        wrapper->setMaxIKSolutions(kMaxIkSolutions);
        wrapper->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        wrapper->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose", "ik_frame"});
        container->insert(std::move(wrapper));
      }
      task.add(std::move(container));
    }
    return task;
  }

  // current state -> detach (planning scene only) -> retreat. The box stays in
  // the scene at its placed pose.
  mtc::Task BuildRetreatTask(const std::string & box_id)
  {
    auto task = NewTask("retreat");
    task.add(std::make_unique<mtc::stages::CurrentState>("current state"));
    {
      auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach box");
      stage->detachObject(box_id, kAttachLink);
      task.add(std::move(stage));
    }
    task.add(MakeVerticalMove("retreat", kRetreatMinM, kRetreatMaxM, 1.0));
    return task;
  }

  // current state -> free-space move to a top-down TCP pose.
  mtc::Task BuildMoveToTask(const geometry_msgs::msg::Point & tcp_position)
  {
    auto task = NewTask("move_to");
    task.add(std::make_unique<mtc::stages::CurrentState>("current state"));
    geometry_msgs::msg::PoseStamped goal;
    goal.header.frame_id = kWorldFrame;
    goal.pose.position = tcp_position;
    goal.pose.orientation = TopDownOrientation();
    auto stage = std::make_unique<mtc::stages::MoveTo>("move to", SamplingPlanner());
    stage->setGroup(kArmGroup);
    stage->setIKFrame(kTcpLink);
    stage->setGoal(goal);
    stage->setTimeout(kConnectTimeoutS);
    task.add(std::move(stage));
    return task;
  }

  std::unique_ptr<mtc::stages::Connect> MakeConnect(
    const std::string & name, const std::shared_ptr<mtc::solvers::PipelinePlanner> & planner)
  {
    auto stage = std::make_unique<mtc::stages::Connect>(
      name, mtc::stages::Connect::GroupPlannerVector{{kArmGroup, planner}});
    stage->setTimeout(kConnectTimeoutS);
    stage->properties().configureInitFrom(mtc::Stage::PARENT);
    return stage;
  }

  std::unique_ptr<mtc::stages::MoveRelative> MakeVerticalMove(
    const std::string & name, double min_m, double max_m, double direction_z)
  {
    auto stage = std::make_unique<mtc::stages::MoveRelative>(name, CartesianPlanner());
    stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
    stage->setIKFrame(kTcpLink);
    stage->setMinMaxDistance(min_m, max_m);
    stage->setDirection(WorldDirection(direction_z));
    return stage;
  }

  // Plan, then execute; throws std::runtime_error with the reason on failure.
  void PlanAndExecute(mtc::Task & task)
  {
    try {
      task.init();
    } catch (const mtc::InitStageException & e) {
      std::ostringstream os;
      os << e;
      throw std::runtime_error("MTC init failed for task '" + task.name() + "': " + os.str());
    }
    if (!task.plan(1)) {
      std::ostringstream os;
      task.explainFailure(os);
      throw std::runtime_error("MTC planning failed for task '" + task.name() + "': " + os.str());
    }
    const auto result = task.execute(*task.solutions().front());
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      throw std::runtime_error(
        "MTC execution failed for task '" + task.name() + "' (error code " +
        std::to_string(result.val) + ")");
    }
  }

  // ------------------------------------------------------------ sim link

  void PublishTaskAction(const std::string & action)
  {
    std_msgs::msg::String msg;
    msg.data = action;
    task_action_pub_->publish(msg);
  }

  // The caller clears have_grasped_size_ before publishing "pick_at": the sim
  // answers within a tick, often before this is called, and the answer comes
  // only once.
  geometry_msgs::msg::Vector3 WaitForGraspedBoxSize()
  {
    const auto deadline = now() + rclcpp::Duration::from_seconds(2.0);
    while (rclcpp::ok() && now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(grasped_size_mutex_);
        if (have_grasped_size_) {
          return latest_grasped_size_;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error(
      "timed out waiting for /sim/grasped_box_size after pick_at (no box resolved at that point?)");
  }

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr task_action_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr grasped_size_sub_;
  rclcpp_action::Server<Pick>::SharedPtr pick_server_;
  rclcpp_action::Server<Place>::SharedPtr place_server_;
  rclcpp_action::Server<MoveTo>::SharedPtr move_to_server_;
  moveit::planning_interface::PlanningSceneInterface psi_;

  std::mutex grasped_size_mutex_;
  geometry_msgs::msg::Vector3 latest_grasped_size_;
  bool have_grasped_size_ = false;

  // Worker threads only, at most one at a time (busy_).
  std::atomic<bool> busy_{false};
  std::string held_box_id_;
  double held_half_height_ = 0.0;
  int next_box_index_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MtcExecutorNode>();
  // Multi-threaded: MTC's loadRobotModel()/plan() make service calls through
  // this node, which need a free executor thread while an action blocks.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
