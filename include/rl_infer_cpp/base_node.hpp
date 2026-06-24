// base_node.hpp: C++ port of rl_infer/base_node.py (RLInferNodeBase).
// Common task plumbing: FSM/source gating, inference timer, rates publishing
// (FLU -> FRD, thrust_body[2] = -thrust), neutral-hover handoff, ground-truth
// obs-availability safety gate, /dbg topics.
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
#include <std_msgs/msg/bool.hpp>
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

  // Derived constructor calls this LAST (virtual calls don't dispatch in
  // C++ constructors; policy loaded, subs created).
  void finish_setup();

  // ---- task interface ----
  virtual int num_obs() const = 0;
  // Fill obs (num_obs floats); last_action_ holds the previous clipped action.
  virtual void build_obs(float* obs) = 0;
  virtual const float* infer_raw(const float* obs) = 0;
  virtual RatesCommand scale_action(const float* raw) = 0;
  virtual bool state_ready() = 0;
  virtual void on_inference_start() {}
  // Term name -> (offset, length) obs slices for /dbg/policy_obs/<term>.
  // NOTE: values are RAW policy-frame; the Python node publishes NED/FRD.
  virtual std::vector<std::pair<std::string, std::pair<int, int>>>
  obs_terms() const {
    return {};
  }
  // Computed /dbg/policy_obs/<name> terms that are NOT slices of the obs.
  virtual std::vector<std::pair<std::string, std::vector<float>>>
  derived_dbg_terms(const float* /*obs*/) const {
    return {};
  }

  // ---- helpers for task nodes ----
  void note_gt_received() {
    gt_recv_mono_ = mono_now();
    ++gt_msg_count_;
  }
  // GT feed status line: distinguishes never-received from stale.
  std::string gt_feed_status() const {
    if (gt_msg_count_ == 0)
      return "feed '" + gt_topic_desc_ + "': ZERO messages ever received — "
             "wrong topic name, publisher not running, or QoS/discovery";
    char buf[160];
    snprintf(buf, sizeof(buf),
             "feed '%s': %ld msgs total, last %.2fs ago (gt_timeout %.2fs) — "
             "intermittent/bursty source",
             gt_topic_desc_.c_str(), static_cast<long>(gt_msg_count_),
             mono_now() - gt_recv_mono_, gt_timeout_);
    return std::string(buf);
  }
  std::string gt_topic_desc_ = "?";
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
  // Throttle notch (rank-2: real-side 8Hz throttle limit-cycle mitigation). RBJ
  // biquad band-stop on cmd.thrust_z, active-inference path only (NOT neutral
  // hover). Unity DC gain → hover trim + ~1-2Hz maneuver band pass untouched;
  // only the ~8Hz limit-cycle is rejected (unlike a low-pass, no broadband lag).
  // throttle_notch_freq<=0 disables (default). See setup_throttle_notch().
  void setup_throttle_notch();
  float apply_throttle_notch(float x);

  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_fsm_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_source_;
  rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr pub_rates_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_ready_;
  rclcpp::TimerBase::SharedPtr ready_timer_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_dbg_obs_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_dbg_infer_time_;
  std::map<std::string,
           rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr>
      pub_dbg_terms_;
  rclcpp::TimerBase::SharedPtr timer_;

  int fsm_state_ = 0;
  std::string active_source_ = "none";
  bool was_active_ = false;
  // Set on mid-flight disengage due to stale obs; must NOT auto-clear —
  // re-engagement requires a fresh FSM/source command (see tick()).
  bool failsafe_latched_ = false;

  int handoff_neutral_frames_ = 25;
  double handoff_hover_thrust_ = 0.26;
  int handoff_remaining_ = 0;

  double gt_timeout_ = 0.3;
  double gt_recv_mono_ = -1.0;
  double gt_block_warn_mono_ = 0.0;
  int64_t gt_msg_count_ = 0;

  // Throttle-notch state/coeffs (a0-normalised RBJ band-stop, DF-II transposed).
  double notch_freq_ = 0.0;  // Hz, <=0 disables
  double notch_q_ = 3.0;     // ~3 → ~2.7Hz -3dB bandwidth at 8Hz (covers 7-9Hz)
  double nb0_ = 1.0, nb1_ = 0.0, nb2_ = 0.0, na1_ = 0.0, na2_ = 0.0;
  double nz1_ = 0.0, nz2_ = 0.0;
  bool notch_primed_ = false;

  std::vector<float> obs_buf_;
};

}  // namespace rl_infer_cpp
