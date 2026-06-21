// gate_node.cpp: C++ port of rl_infer/gate_jax_node.py (GateJaxNode).
// Same node name, parameters, topics and behavior; driven by the same
// gate_jax_params.yaml (checkpoint_path names the .pkl; the exported .npw
// twin is loaded).

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "rl_infer_cpp/base_node.hpp"
#include "rl_infer_cpp/gate_task.hpp"
#include "rl_infer_cpp/policy.hpp"

namespace rl_infer_cpp {

static std::string pkl_to_npw(std::string path) {
  const std::string ext = ".pkl";
  if (path.size() >= ext.size()
      && path.compare(path.size() - ext.size(), ext.size(), ext) == 0)
    path.replace(path.size() - ext.size(), ext.size(), ".npw");
  return path;
}

class GateCppNode : public RlInferNodeBase {
 public:
  GateCppNode() : RlInferNodeBase("gate_jax_node") {
    declare_parameter<std::string>("checkpoint_path", "");
    // goal_local: hover target relative to the gate (gate-local frame); goal_pos
    // = gate_pos + R(gate_quat) * goal_local, recomputed from the live gate pose.
    declare_parameter<std::vector<double>>("goal_local", {1.8, 0.0, -0.4});
    declare_parameter<double>("accel_lpf_alpha", 1.0);
    declare_parameter<bool>("br_filter", true);
    declare_parameter<std::string>("gt_source", "tf");
    declare_parameter<std::string>("gt_pose_topic", "/gate/gt_pose");
    declare_parameter<std::string>("gt_child_frame", "charpi_vision_0");
    // Min stamp dt for the velocity finite-diff: guards VRPN burst
    // near-duplicates.
    declare_parameter<double>("gt_min_dt", 0.004);
    gt_min_dt_ = get_parameter("gt_min_dt").as_double();
    // Aligns the PX4 EKF local frame with the scene frame (px4 source only);
    // calibrate /vrpn pose z vs vehicle_local_position -z on the ground.
    declare_parameter<std::vector<double>>("gt_pos_offset_enu",
                                           {0.0, 0.0, 0.0});
    // gate_pos/gate_quat (-> corners_b/goal_b) ALWAYS track this live mocap
    // pose of the physical gate. gate_frame_offset_rpy rotates the mocap
    // rigid-body frame to the policy gate frame (opening normal = +x, up = +z);
    // identity assumes the rigid body is already defined that way. ggg is
    // BLOCKED until a gate pose arrives fresh within gate_pose_timeout.
    declare_parameter<std::string>("gate_pose_topic", "/vrpn_mocap/gate/pose");
    declare_parameter<std::vector<double>>("gate_frame_offset_rpy",
                                           {0.0, 0.0, 0.0});
    declare_parameter<double>("gate_pose_timeout", 0.5);
    gate_pose_timeout_ = get_parameter("gate_pose_timeout").as_double();

    const auto gl = get_parameter("goal_local").as_double_array();
    goal_local_ = Vec3(gl[0], gl[1], gl[2]);
    // Placeholder scene; the live gate pose sets it before any inference (the
    // safety gate blocks until then).
    task_ = std::make_unique<GateTaskLogic>(
        GateScene{}, 1.0 / inference_rate_,
        get_parameter("accel_lpf_alpha").as_double(),
        get_parameter("br_filter").as_bool());

    gate_pose_topic_ = get_parameter("gate_pose_topic").as_string();
    if (gate_pose_topic_.empty())
      throw std::runtime_error(
          "gate_jax_node: gate_pose_topic is required (live mocap gate pose).");
    const auto off = get_parameter("gate_frame_offset_rpy").as_double_array();
    gate_frame_offset_ = euler_to_quat_xyzw(off[0], off[1], off[2]);
    sub_gate_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        gate_pose_topic_, sensor_qos(),
        std::bind(&GateCppNode::gate_pose_cb, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(),
                "gate pose from '%s' drives corners_b/goal_b "
                "(frame offset rpy %.1f %.1f %.1f deg); ggg blocked until it "
                "arrives.",
                gate_pose_topic_.c_str(), off[0] * 180.0 / M_PI,
                off[1] * 180.0 / M_PI, off[2] * 180.0 / M_PI);

    const std::string gt_source = get_parameter("gt_source").as_string();
    const std::string gt_topic = get_parameter("gt_pose_topic").as_string();
    gt_child_frame_ = get_parameter("gt_child_frame").as_string();
    if (gt_source == "tf") {
      sub_gt_tf_ = create_subscription<tf2_msgs::msg::TFMessage>(
          gt_topic, sensor_qos(),
          std::bind(&GateCppNode::tf_cb, this, std::placeholders::_1));
    } else if (gt_source == "pose" || gt_source == "vrpn") {
      sub_gt_pose_ = create_subscription<geometry_msgs::msg::PoseStamped>(
          gt_topic, sensor_qos(),
          std::bind(&GateCppNode::pose_cb, this, std::placeholders::_1));
    } else if (gt_source == "odom") {
      sub_gt_odom_ = create_subscription<nav_msgs::msg::Odometry>(
          gt_topic, sensor_qos(),
          std::bind(&GateCppNode::odom_cb, this, std::placeholders::_1));
    } else if (gt_source == "px4") {
      // PX4 EKF state: position/velocity/acceleration from
      // vehicle_local_position (NED) + attitude from vehicle_attitude
      // (FRD->NED). Smooth EKF states, no finite differences anywhere.
      const auto off = get_parameter("gt_pos_offset_enu").as_double_array();
      ekf_pos_offset_ = Vec3(off[0], off[1], off[2]);
      sub_ekf_lpos_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
          px4_ns_ + "out/vehicle_local_position", sensor_qos(),
          std::bind(&GateCppNode::ekf_lpos_cb, this, std::placeholders::_1));
      sub_ekf_att_ = create_subscription<px4_msgs::msg::VehicleAttitude>(
          px4_ns_ + "out/vehicle_attitude", sensor_qos(),
          std::bind(&GateCppNode::ekf_att_cb, this, std::placeholders::_1));
      use_ekf_ = true;
    } else {
      throw std::runtime_error(
          "gate_jax_node (cpp): gt_source must be tf|pose|vrpn|odom|px4, got '"
          + gt_source + "'");
    }

    gt_topic_desc_ = gt_source + ":" + gt_topic;
    RCLCPP_INFO(get_logger(), "drone pose source=%s topic=%s child=%s",
                gt_source.c_str(), gt_topic.c_str(), gt_child_frame_.c_str());

    load_policy();
    finish_setup();
  }

 protected:
  int num_obs() const override { return NUM_GATE_OBS; }

  void build_obs(float* obs) override {
    if (use_ekf_) {
      task_->build_obs(gt_pos_, gt_vel_, gt_quat_wxyz_, last_action_, obs,
                       &ekf_acc_enu_);
    } else {
      task_->build_obs(gt_pos_, gt_vel_, gt_quat_wxyz_, last_action_, obs);
    }
  }

  const float* infer_raw(const float* obs) override {
    return policy_->infer(obs);
  }

  RatesCommand scale_action(const float* raw) override {
    return task_->scale_action(raw);
  }

  bool state_ready() override {
    if (!gate_pose_fresh()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "gate pose '%s' not available (no message in %.2fs)"
                           " — ggg blocked. Is the gate mocap publishing?",
                           gate_pose_topic_.c_str(), gate_pose_timeout_);
      return false;
    }
    if (use_ekf_) return gt_ready_ && ekf_att_ready_ && gt_fresh();
    return gt_ready_ && gt_fresh();
  }

  bool gate_pose_fresh() const {
    return gate_pose_recv_mono_ >= 0.0
           && (mono_now() - gate_pose_recv_mono_) < gate_pose_timeout_;
  }

  void on_inference_start() override {
    task_->reset();
    gt_vel_.setZero();
    has_gt_stamp_ = false;
    RCLCPP_INFO(get_logger(), "gate policy engaged — task state reset");
  }

  std::vector<std::pair<std::string, std::pair<int, int>>> obs_terms()
      const override {
    return {{"vel_b", {0, 3}},
            {"roll_pitch", {3, 2}},
            {"accel_b", {5, 3}},
            {"corners_b", {8, 12}},
            {"goal_b", {20, 3}},
            {"prev_action", {23, 4}},
            {"cg_passed", {27, 1}}};
  }

 private:
  std::string resolve_ckpt(const std::string& path) const {
    if (!path.empty() && path[0] == '/') return pkl_to_npw(path);
    return pkl_to_npw(
        ament_index_cpp::get_package_share_directory("rl_infer") + "/" + path);
  }

  void load_policy() {
    const std::string ckpt = get_parameter("checkpoint_path").as_string();
    if (ckpt.empty())
      throw std::runtime_error(
          "gate_jax_node requires 'checkpoint_path' (the gate policy).");
    policy_ = std::make_shared<PolicyMlp>();
    policy_->load(resolve_ckpt(ckpt));
    if (static_cast<int>(policy_->num_obs()) != NUM_GATE_OBS)
      throw std::runtime_error("gate checkpoint num_obs mismatch");
    RCLCPP_INFO(get_logger(), "C++ gate policy loaded from %s (%s)",
                resolve_ckpt(ckpt).c_str(), policy_->arch_name());
  }

  void store_pose(double px, double py, double pz, double qx, double qy,
                  double qz, double qw, double t_sec) {
    const Vec3 new_pos(px, py, pz);
    // World velocity = GT position finite-diff with the REAL message dt.
    if (gt_ready_ && has_gt_stamp_) {
      const double dt = t_sec - gt_stamp_;
      if (dt >= gt_min_dt_) gt_vel_ = (new_pos - gt_pos_) / dt;
    }
    gt_pos_ = new_pos;
    gt_quat_wxyz_ = Vec4(qw, qx, qy, qz);  // ROS xyzw -> Hamilton wxyz
    gt_stamp_ = t_sec;
    has_gt_stamp_ = true;
    gt_ready_ = true;
    note_gt_received();  // obs-availability freshness (safety)
  }

  void pose_cb(geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
    const auto& p = msg->pose.position;
    const auto& q = msg->pose.orientation;
    store_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w,
               stamp_to_sec(msg->header.stamp));
  }

  void odom_cb(nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    store_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w,
               stamp_to_sec(msg->header.stamp));
  }

  // vehicle_local_position (gt_source=px4): NED, converted to the policy's
  // ENU world frame. Validity flags gate freshness so a degrading EKF
  // fail-safes the policy.
  void ekf_lpos_cb(px4_msgs::msg::VehicleLocalPosition::ConstSharedPtr m) {
    if (!(m->xy_valid && m->z_valid && m->v_xy_valid && m->v_z_valid)) return;
    gt_pos_ = ned_to_enu_vec(Vec3(m->x, m->y, m->z)) + ekf_pos_offset_;
    gt_vel_ = ned_to_enu_vec(Vec3(m->vx, m->vy, m->vz));
    ekf_acc_enu_ = ned_to_enu_vec(Vec3(m->ax, m->ay, m->az));
    gt_ready_ = true;
    note_gt_received();  // obs-availability freshness (safety)
  }

  // vehicle_attitude.q: Hamilton wxyz, FRD body -> NED world; converted to
  // the policy's ENU/FLU convention.
  void ekf_att_cb(px4_msgs::msg::VehicleAttitude::ConstSharedPtr m) {
    const Vec4 q_ned_frd(m->q[0], m->q[1], m->q[2], m->q[3]);
    gt_quat_wxyz_ = ned_frd_quat_to_enu_flu(q_ned_frd);
    ekf_att_ready_ = true;
  }

  void tf_cb(tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
    for (const auto& tf : msg->transforms) {
      if (tf.child_frame_id == gt_child_frame_) {
        const auto& t = tf.transform.translation;
        const auto& r = tf.transform.rotation;
        store_pose(t.x, t.y, t.z, r.x, r.y, r.z, r.w,
                   stamp_to_sec(tf.header.stamp));
        return;
      }
    }
  }

  // Live gate pose (mocap of the physical gate) -> scene gate_pos/gate_quat,
  // recompute goal. Drives corners_b/goal_b; runs on the same executor as the
  // inference timer, so no locking needed.
  void gate_pose_cb(geometry_msgs::msg::PoseStamped::ConstSharedPtr m) {
    const Vec3 gate_pos(m->pose.position.x, m->pose.position.y,
                        m->pose.position.z);
    const Vec4 mocap_q(m->pose.orientation.x, m->pose.orientation.y,
                       m->pose.orientation.z, m->pose.orientation.w);  // xyzw
    const Vec4 gate_quat = quat_mul_xyzw(mocap_q, gate_frame_offset_);
    GateScene s;
    s.gate_pos = gate_pos;
    s.gate_quat = gate_quat;
    s.goal_pos = gate_pos + quat_apply_xyzw(gate_quat, goal_local_);
    task_->set_scene(s);
    gate_pose_recv_mono_ = mono_now();   // freshness for the safety gate
  }

  Vec3 goal_local_ = Vec3(1.8, 0.0, -0.4);
  std::unique_ptr<GateTaskLogic> task_;
  std::shared_ptr<PolicyMlp> policy_;
  std::string gt_child_frame_;

  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_gt_tf_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_gt_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_gate_pose_;
  Vec4 gate_frame_offset_ = Vec4(0, 0, 0, 1);  // xyzw, mocap -> opening frame
  std::string gate_pose_topic_;
  double gate_pose_recv_mono_ = -1.0, gate_pose_timeout_ = 0.5;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_gt_odom_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr
      sub_ekf_lpos_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr sub_ekf_att_;
  bool use_ekf_ = false, ekf_att_ready_ = false;
  Vec3 ekf_acc_enu_ = Vec3::Zero(), ekf_pos_offset_ = Vec3::Zero();

  double gt_min_dt_ = 0.004;
  Vec3 gt_pos_ = Vec3::Zero(), gt_vel_ = Vec3::Zero();
  Vec4 gt_quat_wxyz_ = Vec4(1, 0, 0, 0);
  bool gt_ready_ = false, has_gt_stamp_ = false;
  double gt_stamp_ = 0.0;
};

}  // namespace rl_infer_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<rl_infer_cpp::GateCppNode>());
  } catch (const std::exception& e) {
    fprintf(stderr, "gate_jax_node (cpp) fatal: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
