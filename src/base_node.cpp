#include "rl_infer_cpp/base_node.hpp"

#include <cmath>

namespace rl_infer_cpp {

RlInferNodeBase::RlInferNodeBase(const std::string& node_name)
    : rclcpp::Node(node_name) {
  declare_parameter<int>("drone_id", 0);
  declare_parameter<double>("inference_rate", 50.0);
  declare_parameter<std::string>("required_source", "");
  declare_parameter<int>("handoff_neutral_frames", 25);
  declare_parameter<double>("handoff_hover_thrust", 0.26);
  declare_parameter<double>("gt_timeout", 0.3);
  declare_parameter<double>("throttle_notch_freq", 0.0);
  declare_parameter<double>("throttle_notch_q", 3.0);

  drone_id_ = get_parameter("drone_id").as_int();
  inference_rate_ = get_parameter("inference_rate").as_double();
  required_source_ = get_parameter("required_source").as_string();
  handoff_neutral_frames_ = get_parameter("handoff_neutral_frames").as_int();
  handoff_hover_thrust_ = get_parameter("handoff_hover_thrust").as_double();
  gt_timeout_ = get_parameter("gt_timeout").as_double();
  notch_freq_ = get_parameter("throttle_notch_freq").as_double();
  notch_q_ = get_parameter("throttle_notch_q").as_double();
  setup_throttle_notch();

  px4_ns_ = drone_id_ == 0 ? "/fmu/"
                           : "/px4_" + std::to_string(drone_id_) + "/fmu/";

  sub_fsm_ = create_subscription<std_msgs::msg::Int32>(
      "/state/state_drone_" + std::to_string(drone_id_), 10,
      [this](std_msgs::msg::Int32::ConstSharedPtr msg) {
        fsm_state_ = msg->data;
      });
  // /active_traj_source is latched (RELIABLE + TRANSIENT_LOCAL, depth 1).
  sub_source_ = create_subscription<std_msgs::msg::String>(
      "/active_traj_source",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        active_source_ = msg->data;
      });

  pub_rates_ = create_publisher<px4_msgs::msg::VehicleRatesSetpoint>(
      px4_ns_ + "in/vehicle_rates_setpoint", 10);
  pub_dbg_obs_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "/dbg/policy_obs", 10);
  pub_dbg_infer_time_ =
      create_publisher<std_msgs::msg::Float32>("/dbg/policy_infer_time", 10);
}

void RlInferNodeBase::finish_setup() {
  obs_buf_.assign(num_obs(), 0.0f);

  // Throwaway inference so the first engaged tick pays no first-call cost.
  const double t0 = mono_now();
  infer_raw(obs_buf_.data());
  RCLCPP_INFO(get_logger(),
              "policy warm-up done (%.1f ms) — first inference will be fast.",
              (mono_now() - t0) * 1e3);
  RCLCPP_INFO(get_logger(),
              "inference backend = C++/Eigen (rl_infer_cpp — no Python at "
              "runtime; same MLP math as the deployed NumPy backend)");

  // Readiness heartbeat (commander rrr/ggg interlock) at 10 Hz: the commander
  // cannot see THIS node's subscription state (DDS discovery is per-process),
  // and a dead node going silent reads as not-ready (staleness).
  pub_ready_ = create_publisher<std_msgs::msg::Bool>(
      "/rl/ready/" + (required_source_.empty() ? std::string("any")
                                               : required_source_),
      10);
  // Reports not-ready while the fail-safe latch is set, so the commander
  // refuses rrr/ggg until an operator reset (iii/lll).
  ready_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
    std_msgs::msg::Bool m;
    m.data = state_ready() && !failsafe_latched_;
    pub_ready_->publish(m);
  });

  timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / inference_rate_),
      std::bind(&RlInferNodeBase::tick, this));

  RCLCPP_INFO(get_logger(), "%s started: drone=%d rate=%.0fHz",
              get_name(), drone_id_, inference_rate_);
}

void RlInferNodeBase::tick() {
  const bool source_ok =
      required_source_.empty() || active_source_ == required_source_;
  const bool commanded = fsm_state_ == FSM_TRAJ && source_ok;
  const bool obs_ready = state_ready();
  bool active = commanded && obs_ready;

  // Fail-safe LATCH: after a mid-flight disengage on stale obs, do NOT
  // auto-re-engage when the feed returns (the drone may be far out of
  // distribution). Requires a fresh operator command (FSM leaves TRAJ or
  // the source changes).
  if (failsafe_latched_) {
    if (!commanded) {
      failsafe_latched_ = false;
      RCLCPP_INFO(get_logger(),
                  "obs fail-safe latch cleared by operator command — RL may "
                  "be engaged again");
    } else {
      active = false;
      const double now = mono_now();
      if (now - gt_block_warn_mono_ > 2.0) {
        gt_block_warn_mono_ = now;
        RCLCPP_ERROR(get_logger(),
                     "RL engage LATCHED OUT after mid-flight obs fail-safe — "
                     "NOT re-engaging automatically; publishing NEUTRAL HOVER "
                     "until an operator command (iii/lll). %s",
                     gt_feed_status().c_str());
      }
      was_active_ = false;
      // CONTINUOUS neutral hover while latched: PX4 stays in body-rate
      // offboard (FSM still TRAJ); going silent leaves it executing the
      // last latched command forever.
      publish_neutral_hover();
      return;
    }
  }

  if (commanded && !obs_ready) {
    // Commanded but obs missing/stale: do not run the policy on dead state.
    const double now = mono_now();
    if (now - gt_block_warn_mono_ > 2.0) {
      gt_block_warn_mono_ = now;
      RCLCPP_WARN(get_logger(),
                  "RL engage BLOCKED: observation source not available "
                  "(no/stale ground-truth pose). NOT running the policy. %s",
                  gt_feed_status().c_str());
    }
  }

  if (!active) {
    if (was_active_) {
      // Just disengaged: neutral-hover burst so the last (often high-thrust)
      // command can't persist through the offboard mode switch.
      handoff_remaining_ = handoff_neutral_frames_;
      if (commanded && !obs_ready) {
        // MID-FLIGHT fail-safe (still commanded, obs died) -> latch.
        failsafe_latched_ = true;
        RCLCPP_ERROR(get_logger(),
                     "MID-FLIGHT OBS FAIL-SAFE: policy disengaged (%s). "
                     "Re-engagement is LATCHED OUT until a fresh operator "
                     "command.", gt_feed_status().c_str());
      }
    }
    was_active_ = false;
    if (commanded) {
      // We own the rates topic (PX4 in body-rate offboard): while blocked,
      // publish neutral hover CONTINUOUSLY — going silent leaves PX4
      // executing a stale command indefinitely.
      publish_neutral_hover();
    } else if (handoff_remaining_ > 0) {
      --handoff_remaining_;
      publish_neutral_hover();
    }
    return;
  }

  if (!was_active_) {
    // Rising edge: fresh engagement. prev_action starts at zero (training).
    was_active_ = true;
    for (float& v : last_action_) v = 0.0f;
    notch_primed_ = false;  // re-prime the throttle notch on fresh engagement
    on_inference_start();
  }

  build_obs(obs_buf_.data());
  publish_dbg(obs_buf_.data(), static_cast<int>(obs_buf_.size()));

  const auto t0 = std::chrono::steady_clock::now();
  const float* raw = infer_raw(obs_buf_.data());
  const double infer_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std_msgs::msg::Float32 t_msg;
  t_msg.data = static_cast<float>(infer_s);
  pub_dbg_infer_time_->publish(t_msg);

  RatesCommand cmd = scale_action(raw);
  if (notch_freq_ > 0.0)
    cmd.thrust_z = apply_throttle_notch(static_cast<float>(cmd.thrust_z));
  publish_rates(cmd, false);

  for (int i = 0; i < 4; ++i)
    last_action_[i] = std::min(1.0f, std::max(-1.0f, raw[i]));
}

void RlInferNodeBase::publish_rates(const RatesCommand& cmd,
                                    bool reset_integral) {
  px4_msgs::msg::VehicleRatesSetpoint msg;
  msg.timestamp =
      static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
  // FLU (policy) -> FRD (PX4): negate pitch & yaw rates.
  msg.roll = static_cast<float>(cmd.roll);
  msg.pitch = static_cast<float>(-cmd.pitch);
  msg.yaw = static_cast<float>(-cmd.yaw);
  msg.thrust_body[0] = 0.0f;
  msg.thrust_body[1] = 0.0f;
  msg.thrust_body[2] = static_cast<float>(-cmd.thrust_z);
  msg.reset_integral = reset_integral;
  pub_rates_->publish(msg);
}

void RlInferNodeBase::setup_throttle_notch() {
  if (notch_freq_ <= 0.0 || inference_rate_ <= 0.0) return;
  // RBJ band-stop (notch), a0-normalised. Unity gain at DC and Nyquist; deep
  // null at notch_freq_ with -3dB bandwidth set by Q.
  const double w0 = 2.0 * M_PI * notch_freq_ / inference_rate_;
  const double c = std::cos(w0), s = std::sin(w0);
  const double alpha = s / (2.0 * notch_q_);
  const double a0 = 1.0 + alpha;
  nb0_ = 1.0 / a0;
  nb1_ = -2.0 * c / a0;
  nb2_ = 1.0 / a0;
  na1_ = -2.0 * c / a0;
  na2_ = (1.0 - alpha) / a0;
  RCLCPP_INFO(get_logger(),
              "throttle notch ON: f0=%.1fHz Q=%.1f @ %.0fHz "
              "(rank-2 8Hz limit-cycle mitigation)",
              notch_freq_, notch_q_, inference_rate_);
}

float RlInferNodeBase::apply_throttle_notch(float x) {
  // Prime state so a constant input passes unchanged (no engage transient).
  if (!notch_primed_) {
    nz1_ = (nb1_ + nb2_ - na1_ - na2_) * x;
    nz2_ = (nb2_ - na2_) * x;
    notch_primed_ = true;
    return x;
  }
  const double y = nb0_ * static_cast<double>(x) + nz1_;
  nz1_ = nb1_ * static_cast<double>(x) - na1_ * y + nz2_;
  nz2_ = nb2_ * static_cast<double>(x) - na2_ * y;
  return static_cast<float>(y);
}

void RlInferNodeBase::publish_neutral_hover() {
  RatesCommand neutral;
  neutral.thrust_z = handoff_hover_thrust_;
  publish_rates(neutral, true);  // reset_integral clears rate-ctrl windup
}

void RlInferNodeBase::publish_dbg(const float* obs, int n) {
  std_msgs::msg::Float32MultiArray msg;
  msg.data.assign(obs, obs + n);
  pub_dbg_obs_->publish(msg);

  for (const auto& term : obs_terms()) {
    auto it = pub_dbg_terms_.find(term.first);
    if (it == pub_dbg_terms_.end()) {
      it = pub_dbg_terms_
               .emplace(term.first,
                        create_publisher<std_msgs::msg::Float32MultiArray>(
                            "/dbg/policy_obs/" + term.first, 10))
               .first;
    }
    std_msgs::msg::Float32MultiArray t;
    t.data.assign(obs + term.second.first,
                  obs + term.second.first + term.second.second);
    it->second->publish(t);
  }

  for (const auto& term : derived_dbg_terms(obs)) {
    auto it = pub_dbg_terms_.find(term.first);
    if (it == pub_dbg_terms_.end()) {
      it = pub_dbg_terms_
               .emplace(term.first,
                        create_publisher<std_msgs::msg::Float32MultiArray>(
                            "/dbg/policy_obs/" + term.first, 10))
               .first;
    }
    std_msgs::msg::Float32MultiArray t;
    t.data = term.second;
    it->second->publish(t);
  }
}

}  // namespace rl_infer_cpp
