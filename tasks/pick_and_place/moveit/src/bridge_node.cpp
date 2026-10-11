// bridge_node: stands in for ros2_control in the MoveIt method. The plant is
// torque-controlled, so this node is the FollowJointTrajectory server
// ("panda_arm_controller/follow_joint_trajectory", from moveit_controllers.yaml)
// and turns trajectories into torques on /sim/joint_command.
//
// One control step per /sim/joint_states message (see OnJointState). It tracks
// a trajectory (cubic Hermite through each waypoint's position and velocity)
// or holds the last target; it always commands, like a real controller, rather
// than leaving the plant's stale-command hold to do it.
//
// Control law: computed torque, tau = ID(q, qdot, a) with
// a = a_ref + Kp (q_ref - q) + Kd (qdot_ref - qdot) + integral, ID = Pinocchio
// RNEA on panda_arm.urdf plus the MJCF's armature and damping. Plain PD
// without gravity compensation sagged 0.2-0.5 rad; with it, fixed gains still
// rang at the end of every move (damping ratio 0.28-0.6 with pose). Scaling by
// the inertia makes every joint critically damped at every pose.
//
// Single-threaded executor: callbacks and the control step never run
// concurrently, so the shared state is not mutex-guarded.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace
{

constexpr int kNumJoints = 7;
constexpr double kControlPeriodS = 0.02;  // matches mujoco_sim_node.CONTROL_PERIOD_S

const std::array<std::string, kNumJoints> kJointNames = {
  "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
  "panda_joint5", "panda_joint6", "panda_joint7",
};

// Computed-torque gains, in acceleration units, from one natural frequency wn
// (parameter "natural_frequency", rad/s): Kp = wn^2, Kd = 2 wn (critically
// damped), Ki = kIntegralFraction * wn^3 (well inside the Routh bound 2 wn^3).
constexpr double kDefaultNaturalFrequency = 10.0;
constexpr double kIntegralFraction = 0.25;
// Anti-windup clamp on the integral (rad/s^2); a held box needs well under 1.
constexpr double kIntegralAccelLimit = 5.0;
// MuJoCo armature and damping (panda_robot.xml's "panda" class); not in the URDF.
constexpr double kArmature = 0.1;
constexpr double kJointDamping = 1.0;

// Matches the URDF's effort limits and panda_robot.xml's motor ctrlrange.
const std::array<double, kNumJoints> kEffortLimit = {87, 87, 87, 87, 12, 12, 12};

// panda_arm.urdf's position limits. Targets are kept 5 mrad inside them: MoveIt
// may plan exactly to a limit, the arm settles just past it, and the next plan
// then fails with an invalid start state.
const std::array<double, kNumJoints> kPosLower = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
const std::array<double, kNumJoints> kPosUpper = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
constexpr double kSoftLimitMarginRad = 0.005;

void ClampToSoftLimits(std::array<double, kNumJoints> & q)
{
  for (int i = 0; i < kNumJoints; ++i) {
    q[i] = std::clamp(q[i], kPosLower[i] + kSoftLimitMarginRad, kPosUpper[i] - kSoftLimitMarginRad);
  }
}

// Matches the URDF's velocity limits; used only to retime degenerate
// trajectories.
const std::array<double, kNumJoints> kJointVelocityLimit = {
  2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61,
};
// Fraction of the velocity limit used when retiming, leaving the feedback some
// authority. 0.5 gave references that could not settle before the timeout.
constexpr double kSynthesizedVelocityFraction = 0.2;

// Success: past the final waypoint and within these tolerances for
// kSettleTicksRequired consecutive ticks.
constexpr double kGoalPosTolRad = 0.01;
constexpr double kGoalVelTolRadS = 0.05;
constexpr double kSettleWindowS = 0.1;
constexpr int kSettleTicksRequired = static_cast<int>(kSettleWindowS / kControlPeriodS);

// Timeout: 3x the trajectory's duration plus a settle grace (short segments
// timed out on 3x alone while still converging).
constexpr double kTimeoutDurationScale = 3.0;
constexpr double kSettleGraceS = 4.0;
// Minimum segment duration (avoids dividing by zero).
constexpr double kMinSegmentDurationS = 0.05;

double ToSeconds(const builtin_interfaces::msg::Duration & d)
{
  return static_cast<double>(d.sec) + static_cast<double>(d.nanosec) * 1e-9;
}

builtin_interfaces::msg::Duration SecondsToDuration(double t)
{
  builtin_interfaces::msg::Duration d;
  d.sec = static_cast<int32_t>(t);
  d.nanosec = static_cast<uint32_t>((t - d.sec) * 1e9);
  return d;
}

// MoveIt merges an MTC task's sub-trajectories into one before sending it, and
// the merged points arrive with time_from_start == 0 and no velocities. Such
// input is retimed from the position deltas at a fraction of each joint's
// velocity limit.
bool HasUsableTiming(const std::vector<trajectory_msgs::msg::JointTrajectoryPoint> & points)
{
  if (ToSeconds(points.back().time_from_start) <= 1e-6) {
    return false;
  }
  double prev_t = -1.0;
  for (const auto & p : points) {
    const double t = ToSeconds(p.time_from_start);
    if (t < prev_t) {
      return false;
    }
    prev_t = t;
  }
  return true;
}

void SynthesizeTimingIfDegenerate(std::vector<trajectory_msgs::msg::JointTrajectoryPoint> & points)
{
  if (points.size() < 2 || HasUsableTiming(points)) {
    return;
  }
  double t = 0.0;
  points.front().time_from_start = SecondsToDuration(0.0);
  for (size_t i = 1; i < points.size(); ++i) {
    double seg_time = kMinSegmentDurationS;
    for (int j = 0; j < kNumJoints; ++j) {
      const double delta = std::abs(points[i].positions[j] - points[i - 1].positions[j]);
      const double vmax = kSynthesizedVelocityFraction * kJointVelocityLimit[j];
      seg_time = std::max(seg_time, delta / vmax);
    }
    t += seg_time;
    points[i].time_from_start = SecondsToDuration(t);
  }

  // Central-difference velocities at interior points (zero at the ends).
  // Without them every short segment would stop and start, peaking well above
  // the speed the timing was sized for.
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    const double dt = ToSeconds(points[i + 1].time_from_start) - ToSeconds(points[i - 1].time_from_start);
    points[i].velocities.resize(kNumJoints);
    for (int j = 0; j < kNumJoints; ++j) {
      const double v = (points[i + 1].positions[j] - points[i - 1].positions[j]) / dt;
      const double vmax = kSynthesizedVelocityFraction * kJointVelocityLimit[j];
      points[i].velocities[j] = std::clamp(v, -vmax, vmax);
    }
  }
  points.front().velocities.assign(kNumJoints, 0.0);
  points.back().velocities.assign(kNumJoints, 0.0);
}

struct Reference
{
  std::array<double, kNumJoints> position{};
  std::array<double, kNumJoints> velocity{};
  std::array<double, kNumJoints> acceleration{};  // feedforward for computed torque
};

// Cubic Hermite spline per joint through each waypoint's position and velocity
// (zero if absent); its second derivative is the acceleration feedforward.
Reference InterpolateTrajectory(
  const std::vector<trajectory_msgs::msg::JointTrajectoryPoint> & points, double t)
{
  const double t_final = ToSeconds(points.back().time_from_start);
  const auto AsReference = [](const trajectory_msgs::msg::JointTrajectoryPoint & p) {
    Reference ref;
    for (int i = 0; i < kNumJoints; ++i) {
      ref.position[i] = p.positions[i];
      ref.velocity[i] = p.velocities.empty() ? 0.0 : p.velocities[i];
    }
    return ref;
  };
  if (t <= 0.0) {
    return AsReference(points.front());
  }
  if (t >= t_final) {
    return AsReference(points.back());
  }

  size_t k = 0;
  while (k + 1 < points.size() && ToSeconds(points[k + 1].time_from_start) < t) {
    ++k;
  }
  const auto & p0 = points[k];
  const auto & p1 = points[k + 1];
  const double t0 = ToSeconds(p0.time_from_start);
  const double t1 = ToSeconds(p1.time_from_start);
  const double dt = std::max(t1 - t0, kMinSegmentDurationS);
  const double u = (t - t0) / dt;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double h00 = 2 * u3 - 3 * u2 + 1;
  const double h10 = u3 - 2 * u2 + u;
  const double h01 = -2 * u3 + 3 * u2;
  const double h11 = u3 - u2;
  const double dh00 = 6 * u2 - 6 * u;
  const double dh10 = 3 * u2 - 4 * u + 1;
  const double dh01 = -6 * u2 + 6 * u;
  const double dh11 = 3 * u2 - 2 * u;
  const double ddh00 = 12 * u - 6;
  const double ddh10 = 6 * u - 4;
  const double ddh01 = 6 - 12 * u;
  const double ddh11 = 6 * u - 2;

  Reference ref;
  for (int i = 0; i < kNumJoints; ++i) {
    const double p0i = p0.positions[i];
    const double p1i = p1.positions[i];
    const double v0i = p0.velocities.empty() ? 0.0 : p0.velocities[i];
    const double v1i = p1.velocities.empty() ? 0.0 : p1.velocities[i];
    ref.position[i] = h00 * p0i + h10 * dt * v0i + h01 * p1i + h11 * dt * v1i;
    ref.velocity[i] = dh00 / dt * p0i + dh10 * v0i + dh01 / dt * p1i + dh11 * v1i;
    ref.acceleration[i] =
      (ddh00 * p0i + ddh01 * p1i) / (dt * dt) + (ddh10 * v0i + ddh11 * v1i) / dt;
  }
  return ref;
}

// Inverse dynamics of panda_arm.urdf: RNEA plus the MJCF's armature and joint
// damping. Pinocchio's g(q) matched MuJoCo's qfrc_bias at rest. Fixed base.
class DynamicsModel
{
public:
  explicit DynamicsModel(const std::string & urdf_path)
  {
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);
    q_full_ = Eigen::VectorXd::Zero(model_.nq);
    v_full_ = Eigen::VectorXd::Zero(model_.nv);
    a_full_ = Eigen::VectorXd::Zero(model_.nv);
    for (int i = 0; i < kNumJoints; ++i) {
      const auto joint_id = model_.getJointId(kJointNames[i]);
      q_idx_[i] = model_.joints[joint_id].idx_q();
      v_idx_[i] = model_.joints[joint_id].idx_v();
    }
  }

  std::array<double, kNumJoints> InverseDynamics(
    const std::array<double, kNumJoints> & q, const std::array<double, kNumJoints> & qdot,
    const std::array<double, kNumJoints> & qddot)
  {
    for (int i = 0; i < kNumJoints; ++i) {
      q_full_[q_idx_[i]] = q[i];
      v_full_[v_idx_[i]] = qdot[i];
      a_full_[v_idx_[i]] = qddot[i];
    }
    pinocchio::rnea(model_, data_, q_full_, v_full_, a_full_);
    std::array<double, kNumJoints> tau{};
    for (int i = 0; i < kNumJoints; ++i) {
      tau[i] = data_.tau[v_idx_[i]] + kArmature * qddot[i] + kJointDamping * qdot[i];
    }
    return tau;
  }

private:
  pinocchio::Model model_;
  pinocchio::Data data_;
  Eigen::VectorXd q_full_;
  Eigen::VectorXd v_full_;
  Eigen::VectorXd a_full_;
  std::array<int, kNumJoints> q_idx_{};
  std::array<int, kNumJoints> v_idx_{};
};

}  // namespace

class BridgeNode : public rclcpp::Node
{
public:
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJointTrajectory>;

  BridgeNode()
  : Node("bridge_node"),
    dynamics_(
      ament_index_cpp::get_package_share_directory("pick_place_moveit") + "/config/panda_arm.urdf")
  {
    const double wn = declare_parameter("natural_frequency", kDefaultNaturalFrequency);
    kp_ = wn * wn;
    kd_ = 2.0 * wn;
    ki_ = kIntegralFraction * wn * wn * wn;
    RCLCPP_INFO(get_logger(), "computed-torque gains: wn=%.1f Kp=%.1f Kd=%.1f Ki=%.1f", wn, kp_, kd_, ki_);
    cmd_pub_ = create_publisher<sensor_msgs::msg::JointState>("/sim/joint_command", 10);
    // move_group and RViz need named JointStates on /joint_states;
    // /sim/joint_states has no names.
    named_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/sim/joint_states", 10,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {OnJointState(msg);});

    action_server_ = rclcpp_action::create_server<FollowJointTrajectory>(
      this, "panda_arm_controller/follow_joint_trajectory",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const FollowJointTrajectory::Goal> goal) {
        return HandleGoal(goal);
      },
      [this](const std::shared_ptr<GoalHandle> goal_handle) {return HandleCancel(goal_handle);},
      [this](const std::shared_ptr<GoalHandle> goal_handle) {HandleAccepted(goal_handle);});
  }

private:
  // /sim/joint_states has no names; its order is joint1..7, which is
  // kJointNames' order.
  void OnJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->position.size() < kNumJoints || msg->velocity.size() < kNumJoints) {
      return;
    }
    for (int i = 0; i < kNumJoints; ++i) {
      q_[i] = msg->position[i];
      qdot_[i] = msg->velocity[i];
    }
    if (!have_state_) {
      have_state_ = true;
      hold_target_ = q_;
      ClampToSoftLimits(hold_target_);
    }

    sensor_msgs::msg::JointState named;
    named.header.stamp = msg->header.stamp;
    named.name.assign(kJointNames.begin(), kJointNames.end());
    named.position.assign(q_.begin(), q_.end());
    named.velocity.assign(qdot_.begin(), qdot_.end());
    named_state_pub_->publish(named);

    // Control step on each sim state, not on a timer: a separate timer (51.8 Hz vs
    // 50.1 Hz) added an intermittent period of delay, which made the loop
    // oscillate after a grasp. If states stop, commands stop and the plant's
    // stale-command hold takes over.
    ControlTick();
  }

  static int JointIndex(const std::string & name)
  {
    for (int i = 0; i < kNumJoints; ++i) {
      if (kJointNames[i] == name) {
        return i;
      }
    }
    return -1;
  }

  rclcpp_action::GoalResponse HandleGoal(
    std::shared_ptr<const FollowJointTrajectory::Goal> goal)
  {
    if (!have_state_) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: no /sim/joint_states received yet");
      return rclcpp_action::GoalResponse::REJECT;
    }
    const auto & names = goal->trajectory.joint_names;
    if (names.size() != kNumJoints) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: expected %d joints, got %zu",
        kNumJoints, names.size());
      return rclcpp_action::GoalResponse::REJECT;
    }
    for (const auto & name : names) {
      if (JointIndex(name) < 0) {
        RCLCPP_WARN(get_logger(), "Rejecting goal: unrecognized joint name '%s'", name.c_str());
        return rclcpp_action::GoalResponse::REJECT;
      }
    }
    if (goal->trajectory.points.empty()) {
      RCLCPP_WARN(get_logger(), "Rejecting goal: empty trajectory");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse HandleCancel(const std::shared_ptr<GoalHandle> &)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void HandleAccepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    const auto & names = goal->trajectory.joint_names;
    std::array<int, kNumJoints> src_index{};
    for (size_t i = 0; i < names.size(); ++i) {
      src_index[i] = JointIndex(names[i]);
    }

    // Reindex into panda_joint1..7 order (HandleGoal checked the names).
    std::vector<trajectory_msgs::msg::JointTrajectoryPoint> reindexed;
    reindexed.reserve(goal->trajectory.points.size());
    for (const auto & src_point : goal->trajectory.points) {
      trajectory_msgs::msg::JointTrajectoryPoint p;
      p.time_from_start = src_point.time_from_start;
      p.positions.resize(kNumJoints);
      const bool has_vel = src_point.velocities.size() == names.size();
      if (has_vel) {
        p.velocities.resize(kNumJoints);
      }
      for (size_t i = 0; i < names.size(); ++i) {
        p.positions[src_index[i]] = src_point.positions[i];
        if (has_vel) {
          p.velocities[src_index[i]] = src_point.velocities[i];
        }
      }
      reindexed.push_back(std::move(p));
    }

    // A single waypoint gets the current state as its start.
    if (reindexed.size() == 1) {
      trajectory_msgs::msg::JointTrajectoryPoint start;
      start.time_from_start.sec = 0;
      start.time_from_start.nanosec = 0;
      start.positions.assign(q_.begin(), q_.end());
      start.velocities.assign(qdot_.begin(), qdot_.end());
      if (ToSeconds(reindexed.front().time_from_start) <= 0.0) {
        reindexed.front().time_from_start.sec = 0;
        reindexed.front().time_from_start.nanosec =
          static_cast<uint32_t>(kMinSegmentDurationS * 1e9);
      }
      reindexed.insert(reindexed.begin(), std::move(start));
    }

    const bool had_usable_timing = HasUsableTiming(reindexed);
    SynthesizeTimingIfDegenerate(reindexed);
    RCLCPP_INFO(
      get_logger(), "accepted goal: %zu points, duration %.3fs%s",
      reindexed.size(), ToSeconds(reindexed.back().time_from_start),
      had_usable_timing ? "" : " (synthesized -- input had no usable timing)");

    active_points_ = std::move(reindexed);
    active_goal_ = goal_handle;
    traj_start_time_ = now();
    traj_duration_s_ = ToSeconds(active_points_.back().time_from_start);
    settle_ticks_ = 0;
    mode_ = Mode::kTracking;
  }

  void ControlTick()
  {
    if (!have_state_) {
      return;
    }

    // Commanded acceleration (feedforward + PD + integral), turned into torque by
    // the dynamics model.
    std::array<double, kNumJoints> accel{};
    if (mode_ == Mode::kTracking) {
      HandleTracking(accel);  // may finish the goal and switch to hold
    }
    if (mode_ == Mode::kHold) {
      std::array<double, kNumJoints> pos_error{};
      for (int i = 0; i < kNumJoints; ++i) {
        pos_error[i] = hold_target_[i] - q_[i];
        accel[i] = kp_ * pos_error[i] - kd_ * qdot_[i];
      }
      Integrate(pos_error);
    }
    for (int i = 0; i < kNumJoints; ++i) {
      accel[i] += integral_accel_[i];
    }
    std::array<double, kNumJoints> tau = dynamics_.InverseDynamics(q_, qdot_, accel);
    for (int i = 0; i < kNumJoints; ++i) {
      tau[i] = std::clamp(tau[i], -kEffortLimit[i], kEffortLimit[i]);
    }
    PublishCommand(tau);
  }

  // Integral action, accumulated only while the reference is stationary, for
  // loads the model lacks (a held box sagged ~0.013 rad). Kept across goals,
  // since the payload stays.
  void Integrate(const std::array<double, kNumJoints> & pos_error)
  {
    for (int i = 0; i < kNumJoints; ++i) {
      integral_accel_[i] = std::clamp(
        integral_accel_[i] + ki_ * pos_error[i] * kControlPeriodS,
        -kIntegralAccelLimit, kIntegralAccelLimit);
    }
  }

  void HandleTracking(std::array<double, kNumJoints> & accel)
  {
    if (active_goal_->is_canceling()) {
      auto result = std::make_shared<FollowJointTrajectory::Result>();
      result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
      result->error_string = "canceled";
      active_goal_->canceled(result);
      // Hold where we are, not at the trajectory's target.
      hold_target_ = q_;
      ClampToSoftLimits(hold_target_);
      mode_ = Mode::kHold;
      active_goal_.reset();
      return;
    }

    const double elapsed = (now() - traj_start_time_).seconds();
    Reference ref = InterpolateTrajectory(active_points_, elapsed);
    ClampToSoftLimits(ref.position);
    std::array<double, kNumJoints> pos_error{};
    std::array<double, kNumJoints> vel_error{};
    for (int i = 0; i < kNumJoints; ++i) {
      pos_error[i] = ref.position[i] - q_[i];
      vel_error[i] = ref.velocity[i] - qdot_[i];
      accel[i] = ref.acceleration[i] + kp_ * pos_error[i] + kd_ * vel_error[i];
    }
    PublishFeedback(ref, pos_error, vel_error);

    if (elapsed < traj_duration_s_) {
      return;  // still en route
    }
    Integrate(pos_error);
    if (elapsed > kTimeoutDurationScale * traj_duration_s_ + kSettleGraceS) {
      std::ostringstream os;
      os << "timed out before settling within tolerance; final pos error [";
      for (int i = 0; i < kNumJoints; ++i) {
        os << (i ? ", " : "") << pos_error[i];
      }
      os << "] vel error [";
      for (int i = 0; i < kNumJoints; ++i) {
        os << (i ? ", " : "") << vel_error[i];
      }
      os << "]";
      RCLCPP_WARN(get_logger(), "%s", os.str().c_str());
      auto result = std::make_shared<FollowJointTrajectory::Result>();
      result->error_code = FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
      result->error_string = os.str();
      active_goal_->abort(result);
      hold_target_ = q_;
      ClampToSoftLimits(hold_target_);
      mode_ = Mode::kHold;
      active_goal_.reset();
      return;
    }

    const bool within_tol = std::all_of(pos_error.begin(), pos_error.end(),
        [](double e) {return std::abs(e) < kGoalPosTolRad;}) &&
      std::all_of(vel_error.begin(), vel_error.end(),
        [](double e) {return std::abs(e) < kGoalVelTolRadS;});
    settle_ticks_ = within_tol ? settle_ticks_ + 1 : 0;
    if (settle_ticks_ < kSettleTicksRequired) {
      return;
    }

    auto result = std::make_shared<FollowJointTrajectory::Result>();
    result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
    active_goal_->succeed(result);
    std::copy_n(active_points_.back().positions.begin(), kNumJoints, hold_target_.begin());
    ClampToSoftLimits(hold_target_);
    mode_ = Mode::kHold;
    active_goal_.reset();
  }

  void PublishFeedback(
    const Reference & ref, const std::array<double, kNumJoints> & pos_error,
    const std::array<double, kNumJoints> & vel_error)
  {
    auto feedback = std::make_shared<FollowJointTrajectory::Feedback>();
    feedback->header.stamp = now();
    feedback->joint_names.assign(kJointNames.begin(), kJointNames.end());
    feedback->desired.positions.assign(ref.position.begin(), ref.position.end());
    feedback->desired.velocities.assign(ref.velocity.begin(), ref.velocity.end());
    feedback->actual.positions.assign(q_.begin(), q_.end());
    feedback->actual.velocities.assign(qdot_.begin(), qdot_.end());
    feedback->error.positions.assign(pos_error.begin(), pos_error.end());
    feedback->error.velocities.assign(vel_error.begin(), vel_error.end());
    active_goal_->publish_feedback(feedback);
  }

  void PublishCommand(const std::array<double, kNumJoints> & tau)
  {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.effort.assign(tau.begin(), tau.end());
    cmd_pub_->publish(msg);
  }

  enum class Mode {kHold, kTracking};

  DynamicsModel dynamics_;
  double kp_ = 0.0;
  double kd_ = 0.0;
  double ki_ = 0.0;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr named_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp_action::Server<FollowJointTrajectory>::SharedPtr action_server_;

  std::array<double, kNumJoints> q_{};
  std::array<double, kNumJoints> qdot_{};
  bool have_state_ = false;

  Mode mode_ = Mode::kHold;
  std::array<double, kNumJoints> hold_target_{};
  std::array<double, kNumJoints> integral_accel_{};

  std::shared_ptr<GoalHandle> active_goal_;
  std::vector<trajectory_msgs::msg::JointTrajectoryPoint> active_points_;
  rclcpp::Time traj_start_time_;
  double traj_duration_s_ = 0.0;
  int settle_ticks_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Single-threaded on purpose (see the header).
  rclcpp::spin(std::make_shared<BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
