// base_node.hpp: C++ port of rl_infer/base_node.py (RLInferNodeBase).
//
// Handles everything common across tasks: FSM-state + /active_traj_source
// gating, the inference timer, VehicleRatesSetpoint publishing (FLU -> FRD,
// thrust_body[2] = -thrust), the RL->FSM neutral-hover handoff burst, the
// ground-truth obs-availability safety gate, and the /dbg topics.
//
// C++ note vs the Python base: virtual calls don't dispatch in constructors,
// so derived nodes do their own setup in their constructor and then call
// finish_setup() (warm-up inference + timer) as the LAST statement.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

#include "rl_infer_cpp/hover_task.hpp"  // RatesCommand

namespace rl_infer_cpp {

constexpr int FSM_TRAJ = 5;

class RlInferNodeBase : public rclcpp::Node {
 public:
  static rclcpp::QoS sensor_qos() {
    return rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
  }

 protected:
  explicit RlInferNodeBase(const std::string& node_name);

  // Derived constructor calls this LAST (policy loaded, subs created).
  void finish_setup();

  // ---- task interface ----
  virtual int num_obs() const = 0;
  // Fill obs (num_obs floats); last_action_ holds the previous clipped action.
  virtual void build_obs(float* obs) = 0;
  virtual const float* infer_raw(const float* obs) = 0;
  virtual RatesCommand scale_action(const float* raw) = 0;
  virtual bool state_ready() = 0;
  virtual void on_inference_start() {}
  // Term name -> (offset, length) slices of the obs vector for the
  // /dbg/policy_obs/<term> topics. NOTE: content is RAW policy-frame values;
  // the Python node publishes frame-converted (NED/FRD) debug terms: the
  // names match, the frames differ.
  virtual std::vector<std::pair<std::string, std::pair<int, int>>>
  obs_terms() const {
    return {};
  }
  // Extra computed /dbg/policy_obs/<name> terms that are NOT slices of the
  // obs vector (e.g. the hover node's lin_vel_w_ned / target_pos_w_ned).
  virtual std::vector<std::pair<std::string, std::vector<float>>>
  derived_dbg_terms(const float* /*obs*/) const {
    return {};
  }

  // ---- helpers for task nodes ----
  void note_gt_received() { gt_recv_mono_ = mono_now(); }
  bool gt_fresh() const {
    if (gt_timeout_ <= 0.0) return true;
    return gt_recv_mono_ >= 0.0 && (mono_now() - gt_recv_mono_) < gt_timeout_;
  }
  static double mono_now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  static double stamp_to_sec(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<double>(stamp.sec)
           + static_cast<double>(stamp.nanosec) * 1e-9;
  }

  int drone_id_ = 0;
  double inference_rate_ = 50.0;
  std::string required_source_;
  std::string px4_ns_;
  float last_action_[4] = {0.f, 0.f, 0.f, 0.f};

 private:
  void tick();
  void publish_neutral_hover();
  void publish_rates(const RatesCommand& cmd, bool reset_integral);
  void publish_dbg(const float* obs, int n);

  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_fsm_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_source_;
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr pub_rates_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_dbg_obs_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_dbg_infer_time_;
  std::map<std::string,
           rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr>
      pub_dbg_terms_;
  rclcpp::TimerBase::SharedPtr timer_;

  int fsm_state_ = 0;
  std::string active_source_ = "none";
  bool was_active_ = false;

  int handoff_neutral_frames_ = 25;
  double handoff_hover_thrust_ = 0.26;
  int handoff_remaining_ = 0;

  double gt_timeout_ = 0.3;
  double gt_recv_mono_ = -1.0;
  double gt_block_warn_mono_ = 0.0;

  std::vector<float> obs_buf_;
};

}  // namespace rl_infer_cpp
