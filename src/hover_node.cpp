// hover_node.cpp: C++ port of rl_infer/hover_jax_node.py (HoverJaxNode).
// Drop-in replacement driven by the same hover_jax_params.yaml; loads the
// .npw weights exported next to the .pkl checkpoint. gt_source: "odom"
// (gz SITL) | "pose"/"vrpn" (mocap); the deprecated "px4" path is not ported.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "rl_infer_cpp/base_node.hpp"
#include "rl_infer_cpp/hover_task.hpp"
#include "rl_infer_cpp/policy.hpp"

namespace rl_infer_cpp {

// checkpoint params name the .pkl; the C++ node loads the exported .npw twin.
static std::string pkl_to_npw(std::string path) {
  const std::string ext = ".pkl";
  if (path.size() >= ext.size()
      && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    path.replace(path.size() - ext.size(), ext.size(), ".npw");
  return path;
}

static bool is_absolute(const std::string& p) {
  return !p.empty() && p[0] == '/';
}

class HoverCppNode : public RlInferNodeBase {
 public:
  HoverCppNode() : RlInferNodeBase("hover_jax_node") {
    declare_parameter<std::string>("checkpoint_path", "");
    declare_parameter<std::string>("model_dir", "model/hover_jax");
    declare_parameter<std::string>("checkpoint_name", "best_policy.pkl");
    declare_parameter<double>("max_throttle", 0.4);
    declare_parameter<double>("min_throttle", 0.15);
    declare_parameter<double>("max_body_rate_xy", 6.0);
    declare_parameter<double>("max_body_rate_z", 6.0);
    declare_parameter<bool>("apply_policy_enu_yaw_offset", false);
    declare_parameter<bool>("debug_network_input_orientation", false);
    declare_parameter<std::vector<double>>(
        "target_pose", {0.0, 0.0, 1.5, 0.0, 0.0, 0.0});
    declare_parameter<std::string>("target_mode", "hold");
    declare_parameter<bool>("hold_position_on_engage", true);
    declare_parameter<std::string>("gt_source", "odom");
    declare_parameter<std::string>("gt_pose_topic",
                                   "/model/charpi_vision_0/odometry");
    // Min stamp dt for the velocity/ang-vel finite-diff: guards VRPN burst
    // near-duplicates that spike the obs. Below the floor the pose is stored
    // but the previous velocity kept. Half the 120Hz frame period.
    declare_parameter<double>("gt_min_dt", 0.004);
    gt_min_dt_ = get_parameter("gt_min_dt").as_double();

    HoverActionScale scale;
    scale.min_throttle = get_parameter("min_throttle").as_double();
    scale.max_throttle = get_parameter("max_throttle").as_double();
    scale.max_body_rate_xy = get_parameter("max_body_rate_xy").as_double();
    scale.max_body_rate_z = get_parameter("max_body_rate_z").as_double();
    task_ = std::make_unique<HoverTaskLogic>(
        scale, get_parameter("apply_policy_enu_yaw_offset").as_bool());

    const auto tp = get_parameter("target_pose").as_double_array();
    if (tp.size() == 6) task_->set_target_policy_pose(tp.data());

    debug_net_input_ =
        get_parameter("debug_network_input_orientation").as_bool();
    target_mode_ = get_parameter("target_mode").as_string();
    if (target_mode_ == "hold"
        && !get_parameter("hold_position_on_engage").as_bool())
      target_mode_ = "absolute";  // backward-compat override
    if (target_mode_ != "hold" && target_mode_ != "relative"
        && target_mode_ != "absolute")
      target_mode_ = "hold";

    sub_target_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/rl_policy/target_pose", 10,
        std::bind(&HoverCppNode::target_cb, this, std::placeholders::_1));

    gt_source_ = get_parameter("gt_source").as_string();
    const std::string gt_topic = get_parameter("gt_pose_topic").as_string();
    if (gt_source_ == "odom") {
      sub_gt_odom_ = create_subscription<nav_msgs::msg::Odometry>(
          gt_topic, sensor_qos(),
          std::bind(&HoverCppNode::gt_odom_cb, this, std::placeholders::_1));
    } else if (gt_source_ == "pose" || gt_source_ == "vrpn") {
      sub_gt_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          gt_topic, sensor_qos(),
          std::bind(&HoverCppNode::gt_pose_cb, this, std::placeholders::_1));
    } else {
      throw std::runtime_error(
          "hover_jax_node (cpp): gt_source must be odom|pose|vrpn — the "
          "deprecated 'px4' EKF path is not ported (use the Python node).");
    }
    gt_topic_desc_ = gt_source_ + ":" + gt_topic;
    RCLCPP_INFO(get_logger(),
                "PX4 EKF feedback subs DISABLED (no /fmu/out/vehicle_odometry,"
                " no /fmu/out/esc_status) — obs sourced from %s '%s'.",
                gt_source_.c_str(), gt_topic.c_str());

    load_policy();
    finish_setup();
  }

 protected:
  int num_obs() const override { return NUM_HOVER_OBS; }

  void build_obs(float* obs) override {
    task_->build_obs(gt_pos_, gt_vel_, gt_quat_wxyz_, gt_ang_vel_,
                     last_action_, obs);
    if (debug_net_input_) log_network_input(obs);
  }

  // 1 Hz INFO line decoding the policy input (Python parity:
  // _log_network_orientation_debug).
  void log_network_input(const float* obs) {
    const double now = mono_now();
    if (now - last_dbg_log_mono_ < 1.0) return;
    last_dbg_log_mono_ = now;
    auto rpy_of = [](const float* m) {  // row-major rotmat -> ZYX euler deg
      const Mat3 r = (Mat3() << m[0], m[1], m[2], m[3], m[4], m[5], m[6],
                      m[7], m[8]).finished();
      const Vec4 q = rotmat_to_quat(r);
      const double roll = std::atan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                     1 - 2 * (q[1] * q[1] + q[2] * q[2]));
      const double pitch = std::asin(std::min(
          1.0, std::max(-1.0, 2 * (q[0] * q[2] - q[3] * q[1]))));
      const double yaw = std::atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                    1 - 2 * (q[2] * q[2] + q[3] * q[3]));
      constexpr double r2d = 180.0 / M_PI;
      return Vec3(roll * r2d, pitch * r2d, yaw * r2d);
    };
    const Vec3 drpy = rpy_of(obs), trpy = rpy_of(obs + 18);
    RCLCPP_INFO(get_logger(),
                "network policy input: drone_rpy=[%.2f, %.2f, %.2f], "
                "target_rpy=[%.2f, %.2f, %.2f], lin_vel_b=[%.3f, %.3f, %.3f],"
                " ang_vel_b=[%.3f, %.3f, %.3f], target_pos_b=[%.3f, %.3f, "
                "%.3f], last_action=[%.3f, %.3f, %.3f, %.3f]",
                drpy[0], drpy[1], drpy[2], trpy[0], trpy[1], trpy[2], obs[9],
                obs[10], obs[11], obs[12], obs[13], obs[14], obs[15], obs[16],
                obs[17], obs[27], obs[28], obs[29], obs[30]);
  }

  const float* infer_raw(const float* obs) override {
    return policy_.infer(obs);
  }

  RatesCommand scale_action(const float* raw) override {
    return task_->scale_action(raw);
  }

  bool state_ready() override { return gt_ready_ && gt_fresh(); }

  void on_inference_start() override {
    task_->reset();  // zero the command slew-limiter state
    gt_vel_.setZero();
    has_gt_stamp_ = false;
    if (gt_ready_) {
      // Engage pose (world ENU) anchors hold/relative; seed the target in
      // EVERY mode so the drone holds until the first sequence pose arrives.
      engage_yaw_ = quat_wxyz_yaw(gt_quat_wxyz_);
      engage_pos_ = gt_pos_;
      has_engage_ = true;
      const double pose[6] = {gt_pos_[0], gt_pos_[1], gt_pos_[2],
                              0.0, 0.0, engage_yaw_};
      task_->set_target_policy_pose(pose);
      RCLCPP_INFO(get_logger(),
                  "hover engage (mode=%s) at (%.2f, %.2f, %.2f) yaw=%.0f deg",
                  target_mode_.c_str(), gt_pos_[0], gt_pos_[1], gt_pos_[2],
                  engage_yaw_ * 180.0 / M_PI);
    }
    RCLCPP_INFO(get_logger(), "hover policy engaged: deploy state reset");
  }

  std::vector<std::pair<std::string, std::pair<int, int>>> obs_terms()
      const override {
    return {{"rot_mat_w", {0, 9}},
            {"lin_vel_b", {9, 3}},
            {"ang_vel_b", {12, 3}},
            {"target_pos_b_ori_w", {15, 12}},
            {"last_action", {27, 4}}};
  }

  // Debug terms derived from the obs (body->world via rot_mat_w, ENU->NED).
  std::vector<std::pair<std::string, std::vector<float>>> derived_dbg_terms(
      const float* obs) const override {
    Eigen::Matrix3f rot;
    rot << obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7],
        obs[8];
    const Eigen::Vector3f v = rot * Eigen::Vector3f(obs[9], obs[10], obs[11]);
    const Eigen::Vector3f t =
        rot * Eigen::Vector3f(obs[15], obs[16], obs[17]);
    return {{"lin_vel_w_ned", {v[1], v[0], -v[2]}},
            {"target_pos_w_ned", {t[1], t[0], -t[2]}}};
  }

 private:
  void load_policy() {
    const std::string pkg_share =
        ament_index_cpp::get_package_share_directory("rl_infer");
    std::string ckpt = get_parameter("checkpoint_path").as_string();
    if (!ckpt.empty() && !is_absolute(ckpt)) ckpt = pkg_share + "/" + ckpt;
    if (ckpt.empty()) {
      std::string dir = get_parameter("model_dir").as_string();
      if (!is_absolute(dir)) dir = pkg_share + "/" + dir;
      ckpt = dir + "/" + get_parameter("checkpoint_name").as_string();
    }
    ckpt = pkl_to_npw(ckpt);
    policy_.load(ckpt);
    if (static_cast<int>(policy_.num_obs()) != NUM_HOVER_OBS)
      throw std::runtime_error("hover checkpoint num_obs mismatch");
    RCLCPP_INFO(get_logger(), "C++ hover policy loaded from %s (%s)",
                ckpt.c_str(), policy_.arch_name());
  }

  // GT sink (odom + pose/vrpn): world ENU position, Hamilton wxyz attitude.
  // Velocity from position finite-diff with the REAL message dt; body
  // ang-vel from quaternion finite-diff: omega_b = 2*(q_prev^-1 ⊗ q_now).xyz/dt.
  void store_gt(const Vec3& new_pos, const Vec4& new_q, double t_sec) {
    if (gt_ready_ && has_gt_stamp_) {
      const double dt = t_sec - gt_stamp_;
      if (dt >= gt_min_dt_) {
        gt_vel_ = (new_pos - gt_pos_) / dt;
        Vec4 qd = quat_mul(quat_inv(gt_quat_wxyz_), new_q);
        if (qd[0] < 0.0) qd = -qd;
        gt_ang_vel_ = 2.0 * Vec3(qd[1], qd[2], qd[3]) / dt;
      }
    }
    gt_pos_ = new_pos;
    gt_quat_wxyz_ = new_q;
    gt_stamp_ = t_sec;
    has_gt_stamp_ = true;
    gt_ready_ = true;
    note_gt_received();  // obs-availability freshness (safety)
  }

  void gt_odom_cb(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    store_gt(Vec3(p.x, p.y, p.z), Vec4(q.w, q.x, q.y, q.z),
             stamp_to_sec(msg->header.stamp));
  }

  void gt_pose_cb(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
    const auto& p = msg->pose.position;
    const auto& q = msg->pose.orientation;
    store_gt(Vec3(p.x, p.y, p.z), Vec4(q.w, q.x, q.y, q.z),
             stamp_to_sec(msg->header.stamp));
  }

  void target_cb(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
    if (target_mode_ == "hold") return;  // sequence ignored
    const auto& p = msg->pose.position;
    const auto& q = msg->pose.orientation;
    if (target_mode_ == "relative") {
      if (!has_engage_) return;
      // XY = engage XY + target's NED offset (N=ENU y, E=ENU x);
      // Z = absolute altitude (-NED down). Keep engage heading.
      const double pose[6] = {engage_pos_[0] + p.y, engage_pos_[1] + p.x,
                              -p.z, 0.0, 0.0, engage_yaw_};
      task_->set_target_policy_pose(pose);
    } else {  // "absolute"
      task_->set_target_ned_frd(Vec3(p.x, p.y, p.z),
                                Vec4(q.w, q.x, q.y, q.z));
    }
  }

  PolicyMlp policy_;
  std::unique_ptr<HoverTaskLogic> task_;
  std::string target_mode_, gt_source_;
  bool debug_net_input_ = false;
  double last_dbg_log_mono_ = 0.0;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_target_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_gt_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_gt_pose_;

  double gt_min_dt_ = 0.004;
  Vec3 gt_pos_ = Vec3::Zero(), gt_vel_ = Vec3::Zero(),
       gt_ang_vel_ = Vec3::Zero();
  Vec4 gt_quat_wxyz_ = Vec4(1, 0, 0, 0);
  bool gt_ready_ = false, has_gt_stamp_ = false;
  double gt_stamp_ = 0.0;

  Vec3 engage_pos_ = Vec3::Zero();
  double engage_yaw_ = 0.0;
  bool has_engage_ = false;
};

}  // namespace rl_infer_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<rl_infer_cpp::HoverCppNode>());
  } catch (const std::exception& e) {
    fprintf(stderr, "hover_jax_node (cpp) fatal: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
