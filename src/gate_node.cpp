// gate_node.cpp: C++ port of rl_infer/gate_jax_node.py (GateJaxNode).
//
// Same node name, parameters, topics and behavior as the Python node, driven
// by the same gate_jax_params.yaml (checkpoint_path names the .pkl; the .npw
// twin exported by rl_infer/scripts/export_policy_cpp.py is loaded).

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
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
    declare_parameter<std::vector<std::string>>("roll_checkpoints", {""});
    declare_parameter<std::vector<double>>("gate_pos_enu", {0.0, 0.0, 1.9});
    declare_parameter<std::vector<double>>("standby_pos_enu",
                                           {1.27, -1.27, 1.2});
    declare_parameter<double>("gate_roll", DEFAULT_GATE_ROLL);
    declare_parameter<std::vector<double>>("goal_local", {1.8, 0.0, -0.4});
    declare_parameter<double>("accel_lpf_alpha", 1.0);
    declare_parameter<bool>("br_filter", true);
    declare_parameter<std::string>("gt_source", "tf");
    declare_parameter<std::string>("gt_pose_topic", "/gate/gt_pose");
    declare_parameter<std::string>("gt_child_frame", "charpi_vision_0");
    // Min stamp dt for the velocity finite-diff — guards against VRPN burst
    // near-duplicates (see hover_node.cpp gt_min_dt for the measured data).
    declare_parameter<double>("gt_min_dt", 0.004);
    gt_min_dt_ = get_parameter("gt_min_dt").as_double();

    const auto gp = get_parameter("gate_pos_enu").as_double_array();
    const auto sp = get_parameter("standby_pos_enu").as_double_array();
    const auto gl = get_parameter("goal_local").as_double_array();
    goal_local_ = Vec3(gl[0], gl[1], gl[2]);
    scene_ = GateScene::from_standby(
        Vec3(gp[0], gp[1], gp[2]), Vec3(sp[0], sp[1], sp[2]),
        get_parameter("gate_roll").as_double(), goal_local_);
    // Fixed gate normal yaw — kept when the roll changes at runtime (tttN).
    gate_yaw_ = quat_xyzw_yaw(scene_.gate_quat);

    task_ = std::make_unique<GateTaskLogic>(
        scene_, 1.0 / inference_rate_,
        get_parameter("accel_lpf_alpha").as_double(),
        get_parameter("br_filter").as_bool());

    sub_set_roll_ = create_subscription<std_msgs::msg::Float64>(
        "/gate/set_roll", 10,
        std::bind(&GateCppNode::set_roll_cb, this, std::placeholders::_1));

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
    } else {
      throw std::runtime_error(
          "gate_jax_node (cpp): gt_source must be tf|pose|vrpn|odom, got '"
          + gt_source + "'");
    }

    gt_topic_desc_ = gt_source + ":" + gt_topic;
    RCLCPP_INFO(get_logger(),
                "Gate scene: gate_pos=(%.3f, %.3f, %.3f) goal_pos=(%.3f, %.3f,"
                " %.3f) | pose source=%s topic=%s child=%s",
                scene_.gate_pos[0], scene_.gate_pos[1], scene_.gate_pos[2],
                scene_.goal_pos[0], scene_.goal_pos[1], scene_.goal_pos[2],
                gt_source.c_str(), gt_topic.c_str(), gt_child_frame_.c_str());

    load_policies();
    finish_setup();
  }

 protected:
  int num_obs() const override { return NUM_GATE_OBS; }

  void build_obs(float* obs) override {
    task_->build_obs(gt_pos_, gt_vel_, gt_quat_wxyz_, last_action_, obs);
  }

  const float* infer_raw(const float* obs) override {
    return policy_->infer(obs);
  }

  RatesCommand scale_action(const float* raw) override {
    return task_->scale_action(raw);
  }

  bool state_ready() override { return gt_ready_ && gt_fresh(); }

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

  std::shared_ptr<PolicyMlp> load_one(const std::string& path) const {
    auto p = std::make_shared<PolicyMlp>();
    p->load(resolve_ckpt(path));
    if (static_cast<int>(p->num_obs()) != NUM_GATE_OBS)
      throw std::runtime_error("gate checkpoint num_obs mismatch");
    return p;
  }

  void load_policies() {
    const std::string ckpt = get_parameter("checkpoint_path").as_string();
    if (ckpt.empty())
      throw std::runtime_error(
          "gate_jax_node requires 'checkpoint_path' (the default/curriculum "
          "gate-traversal policy).");
    default_policy_ = load_one(ckpt);
    policy_ = default_policy_;
    RCLCPP_INFO(get_logger(), "C++ default gate policy loaded from %s (%s)",
                resolve_ckpt(ckpt).c_str(), policy_->arch_name());

    // Pre-load per-roll fine-tuned policies so tttN switches instantly.
    // NOTE: as_string_array() returns a reference INTO the temporary
    // Parameter: copy it before iterating (range-for over the temporary's
    // innards is a dangling reference -> segfault).
    const std::vector<std::string> specs =
        get_parameter("roll_checkpoints").as_string_array();
    for (const auto& spec : specs) {
      const auto colon = spec.find(':');
      if (spec.empty() || colon == std::string::npos) continue;
      int deg = 0;
      try {
        deg = static_cast<int>(std::lround(std::stod(spec.substr(0, colon))));
      } catch (const std::exception&) {
        RCLCPP_WARN(get_logger(), "roll_checkpoints: bad angle in '%s'",
                    spec.c_str());
        continue;
      }
      const std::string path = spec.substr(colon + 1);
      try {
        roll_policies_[deg] = load_one(path);
        RCLCPP_INFO(get_logger(), "pre-loaded fine-tuned policy: %d deg -> %s",
                    deg, path.c_str());
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(),
                    "roll_checkpoints: %d deg checkpoint not loadable: %s",
                    deg, e.what());
      }
    }
  }

  void store_pose(double px, double py, double pz, double qx, double qy,
                  double qz, double qw, double t_sec) {
    const Vec3 new_pos(px, py, pz);
    // World velocity = clean GT position finite-diff with the REAL message dt.
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

  // Rebuild the scene with a new roll about the FIXED normal yaw; switch to
  // a per-roll fine-tuned policy if one is loaded (else the default).
  void set_roll_cb(std_msgs::msg::Float64::ConstSharedPtr msg) {
    const double roll = msg->data;
    scene_ = GateScene::with_roll(scene_, roll, gate_yaw_, goal_local_);
    task_->set_scene(scene_);

    const int deg = static_cast<int>(std::lround(roll * 180.0 / M_PI));
    auto it = roll_policies_.find(deg);
    auto new_policy = it != roll_policies_.end() ? it->second : default_policy_;
    if (new_policy && new_policy != policy_) {
      policy_ = new_policy;
      RCLCPP_INFO(get_logger(), "policy switched to %s",
                  it != roll_policies_.end()
                      ? ("fine-tuned " + std::to_string(deg) + " deg").c_str()
                      : "default (curriculum)");
    }
    RCLCPP_INFO(get_logger(),
                "gate roll set to %.1f deg (yaw kept %.1f deg)",
                roll * 180.0 / M_PI, gate_yaw_ * 180.0 / M_PI);
  }

  GateScene scene_;
  Vec3 goal_local_ = Vec3(1.8, 0.0, -0.4);
  double gate_yaw_ = 0.0;
  std::unique_ptr<GateTaskLogic> task_;
  std::shared_ptr<PolicyMlp> policy_, default_policy_;
  std::map<int, std::shared_ptr<PolicyMlp>> roll_policies_;
  std::string gt_child_frame_;

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_set_roll_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_gt_tf_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_gt_pose_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_gt_odom_;

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
