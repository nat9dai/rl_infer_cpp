// gate_task.hpp: C++ port of rl_infer/gate/task.py (GateScene, GateTaskLogic,
// compute_gate_obs). Pure logic, no ROS. Double math, float32 obs output.
//
// 28-D obs: vel_b(3) roll_pitch(2) accel_b(3) corners_b(12) goal_b(3)
//           prev_action(4) cg_passed(1)
//
// Deploy-config env vars (same names/defaults as the Python task):
//   GATE_THRUST_REMAP (default 1; the launch deploys with 0)
//   GATE_MAX_RATE_ROLL/PITCH/YAW (defaults 6.0; the launch deploys 4/2.5/3)
//   GATE_SLEW (default 1), GATE_SLEW_ROLL/PITCH/YAW (110/50/25 rad/s²)
#pragma once

#include <cstdlib>

#include "rl_infer_cpp/hover_task.hpp"  // env_double/env_flag, RatesCommand
#include "rl_infer_cpp/math_util.hpp"

namespace rl_infer_cpp {

constexpr int NUM_GATE_OBS = 28;
constexpr int NUM_GATE_ACT = 4;
constexpr double GATE_WIDTH = 0.60;
constexpr double GATE_HEIGHT = 0.25;
constexpr double GATE_GRAVITY = 9.81;
constexpr double GATE_MIN_THROTTLE = 0.15;
constexpr double GATE_MAX_THROTTLE = 0.40;
constexpr double DEFAULT_GATE_ROLL = M_PI / 4.0;

// Body-rate 2nd-order response model (charpi_physics_gz) for the optional
// br_filter (deploys with br_filter=false; kept for parity).
constexpr double BR_WN[3] = {62.86, 67.41, 8.22};
constexpr double BR_DAMPING[3] = {0.707, 0.592, 0.484};
constexpr double BR_GAIN[3] = {0.994, 0.992, 0.837};
constexpr double BR_PHYSICS_DT = 0.002;
constexpr int BR_N_SUBSTEPS = 5;

struct GateScene {
  Vec3 gate_pos = Vec3::Zero();
  Vec4 gate_quat = Vec4(0, 0, 0, 1);  // xyzw
  Vec3 goal_pos = Vec3::Zero();

  static GateScene from_standby(const Vec3& gate_pos_enu,
                                const Vec3& standby_pos_enu, double gate_roll,
                                const Vec3& goal_local) {
    GateScene s;
    s.gate_pos = gate_pos_enu;
    const double dx = gate_pos_enu[0] - standby_pos_enu[0];
    const double dy = gate_pos_enu[1] - standby_pos_enu[1];
    const double norm = std::sqrt(dx * dx + dy * dy);
    const double yaw = (norm < 1e-6) ? 0.0 : std::atan2(dy, dx);
    s.gate_quat = euler_to_quat_xyzw(gate_roll, 0.0, yaw);
    s.goal_pos = s.gate_pos + quat_apply_xyzw(s.gate_quat, goal_local);
    return s;
  }

  // Rebuild with a new roll about the SAME fixed normal yaw (/gate/set_roll).
  static GateScene with_roll(const GateScene& base, double roll, double yaw,
                             const Vec3& goal_local) {
    GateScene s;
    s.gate_pos = base.gate_pos;
    s.gate_quat = euler_to_quat_xyzw(roll, 0.0, yaw);
    s.goal_pos = s.gate_pos + quat_apply_xyzw(s.gate_quat, goal_local);
    return s;
  }
};

class GateTaskLogic {
 public:
  GateTaskLogic(const GateScene& scene, double ctrl_dt, double accel_lpf_alpha,
                bool br_filter)
      : scene_(scene),
        ctrl_dt_(ctrl_dt),
        br_filter_(br_filter),
        accel_lpf_alpha_(std::min(1.0, std::max(0.0, accel_lpf_alpha))) {
    max_rate_ = Vec3(env_double("GATE_MAX_RATE_ROLL", 6.0),
                     env_double("GATE_MAX_RATE_PITCH", 6.0),
                     env_double("GATE_MAX_RATE_YAW", 6.0));
    slew_on_ = env_flag("GATE_SLEW", true);
    slew_ = Vec3(env_double("GATE_SLEW_ROLL", 110.0),
                 env_double("GATE_SLEW_PITCH", 50.0),
                 env_double("GATE_SLEW_YAW", 25.0));
    thrust_remap_on_ = env_flag("GATE_THRUST_REMAP", true);
    reset();
  }

  void set_scene(const GateScene& scene) { scene_ = scene; }
  const GateScene& scene() const { return scene_; }
  bool cg_passed() const { return cg_passed_; }

  void reset() {
    has_prev_vel_ = false;
    prev_vel_w_.setZero();
    cg_passed_ = false;
    has_prev_xg_ = false;
    prev_xg_ = 0.0;
    has_accel_filt_ = false;
    accel_filt_.setZero();
    br_.setZero();
    br_dot_.setZero();
    prev_br_cmd_.setZero();
  }

  // Build the 28-D obs; quat in Hamilton wxyz (converted internally to xyzw).
  // Mutates the cross-step state (accel finite-diff, cg_passed) exactly like
  // GateTaskLogic.build_obs_from_state.
  void build_obs(const Vec3& position_w, const Vec3& velocity_w,
                 const Vec4& quat_wxyz, const float last_action[NUM_GATE_ACT],
                 float out[NUM_GATE_OBS]) {
    const Vec4 quat = wxyz_to_xyzw(quat_wxyz);

    Vec3 accel_w = Vec3::Zero();
    if (has_prev_vel_) accel_w = (velocity_w - prev_vel_w_) / ctrl_dt_;
    if (accel_lpf_alpha_ < 1.0) {
      if (!has_accel_filt_) {
        accel_filt_ = accel_w;
        has_accel_filt_ = true;
      } else {
        accel_filt_ += accel_lpf_alpha_ * (accel_w - accel_filt_);
      }
      accel_w = accel_filt_;
    }

    update_cg_passed(position_w);

    int k = 0;
    // 1) body-frame linear velocity
    const Vec3 vel_b = quat_rotate_inverse_xyzw(quat, velocity_w);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(vel_b[i]);

    // 2) roll, pitch: exact env formulas incl. degenerate guard
    const double qx = quat[0], qy = quat[1], qz = quat[2], qw = quat[3];
    const double roll_num = 2.0 * (qw * qx + qy * qz);
    double roll_den = 1.0 - 2.0 * (qx * qx + qy * qy);
    if (std::abs(roll_num) < 1e-12 && std::abs(roll_den) < 1e-12)
      roll_den = 1.0;
    const double roll = std::atan2(roll_num, roll_den);
    const double sp = std::min(1.0 - 1e-6,
                               std::max(-1.0 + 1e-6, 2.0 * (qw * qy - qz * qx)));
    const double pitch = std::asin(sp);
    out[k++] = static_cast<float>(roll);
    out[k++] = static_cast<float>(pitch);

    // 3) IMU specific force: R^T (a_world - g_world), g_world = (0,0,-g)
    const Vec3 g_world(0.0, 0.0, -GATE_GRAVITY);
    const Vec3 accel_b = quat_rotate_inverse_xyzw(quat, accel_w - g_world);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(accel_b[i]);

    // 4) gate corners in body frame: order TR, TL, BL, BR
    constexpr double hw = GATE_WIDTH / 2.0, hh = GATE_HEIGHT / 2.0;
    const Vec3 corners_local[4] = {Vec3(0, hw, hh), Vec3(0, -hw, hh),
                                   Vec3(0, -hw, -hh), Vec3(0, hw, -hh)};
    for (const auto& cl : corners_local) {
      const Vec3 cw = scene_.gate_pos + quat_apply_xyzw(scene_.gate_quat, cl);
      const Vec3 cb = quat_rotate_inverse_xyzw(quat, cw - position_w);
      for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(cb[i]);
    }

    // 5) goal position in body frame
    const Vec3 goal_b =
        quat_rotate_inverse_xyzw(quat, scene_.goal_pos - position_w);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(goal_b[i]);

    // 6) previous action
    for (int i = 0; i < NUM_GATE_ACT; ++i) out[k++] = last_action[i];

    // 7) sticky cg_passed flag
    out[k++] = cg_passed_ ? 1.0f : 0.0f;

    prev_vel_w_ = velocity_w;
    has_prev_vel_ = true;
  }

  RatesCommand scale_action(const float raw[NUM_GATE_ACT]) {
    double a[4];
    for (int i = 0; i < 4; ++i)
      a[i] = std::min(1.0, std::max(-1.0, static_cast<double>(raw[i])));
    double thrust = GATE_MIN_THROTTLE
        + (a[0] + 1.0) * 0.5 * (GATE_MAX_THROTTLE - GATE_MIN_THROTTLE);
    if (thrust_remap_on_) thrust = remap_throttle(thrust);
    Vec3 br_ref(a[1] * max_rate_[0], a[2] * max_rate_[1], a[3] * max_rate_[2]);
    Vec3 out = br_filter_ ? filter_body_rates(br_ref) : br_ref;
    if (slew_on_) {
      const Vec3 step = slew_ * ctrl_dt_;
      out = prev_br_cmd_ + (out - prev_br_cmd_).cwiseMax(-step).cwiseMin(step);
      prev_br_cmd_ = out;
    }
    RatesCommand cmd;
    cmd.roll = out[0];
    cmd.pitch = out[1];
    cmd.yaw = out[2];
    cmd.thrust_z = std::min(GATE_MAX_THROTTLE,
                            std::max(GATE_MIN_THROTTLE, thrust));
    return cmd;
  }

 private:
  static double remap_throttle(double thr) {
    constexpr double c0 = 1.613356, c1 = -2.856781, c2 = 2.09172,
                     c3 = -0.106211;
    const double t = c0 * thr * thr * thr + c1 * thr * thr + c2 * thr + c3;
    return std::min(GATE_MAX_THROTTLE, std::max(GATE_MIN_THROTTLE, t));
  }

  // Sticky gate-pass: CG sweeps gate-local x from <0 to >=0 inside the opening.
  void update_cg_passed(const Vec3& pos) {
    const Vec3 drone_g =
        quat_rotate_inverse_xyzw(scene_.gate_quat, pos - scene_.gate_pos);
    const double x_g = drone_g[0], y_g = drone_g[1], z_g = drone_g[2];
    if (has_prev_xg_ && !cg_passed_) {
      const bool crossed = (prev_xg_ < 0.0) && (x_g >= 0.0);
      const bool within = (std::abs(y_g) < GATE_WIDTH / 2.0)
                          && (std::abs(z_g) < GATE_HEIGHT / 2.0);
      if (crossed && within) cg_passed_ = true;
    }
    prev_xg_ = x_g;
    has_prev_xg_ = true;
  }

  // charpi_physics_gz approx_substep, BR_FF=0 (matches _filter_body_rates).
  Vec3 filter_body_rates(const Vec3& br_ref) {
    for (int s = 0; s < BR_N_SUBSTEPS; ++s) {
      Vec3 br_ddot, new_br;
      for (int i = 0; i < 3; ++i) {
        br_ddot[i] = BR_GAIN[i] * BR_WN[i] * BR_WN[i] * br_ref[i]
                     - 2.0 * BR_DAMPING[i] * BR_WN[i] * br_dot_[i]
                     - BR_WN[i] * BR_WN[i] * br_[i];
        new_br[i] = br_[i] + br_dot_[i] * BR_PHYSICS_DT;  // old br_dot (env)
      }
      br_dot_ += br_ddot * BR_PHYSICS_DT;
      br_ = new_br;
    }
    return br_;
  }

  GateScene scene_;
  double ctrl_dt_;
  bool br_filter_;
  double accel_lpf_alpha_;
  Vec3 max_rate_, slew_;
  bool slew_on_ = true, thrust_remap_on_ = true;

  // cross-step state
  bool has_prev_vel_ = false;
  Vec3 prev_vel_w_ = Vec3::Zero();
  bool cg_passed_ = false;
  bool has_prev_xg_ = false;
  double prev_xg_ = 0.0;
  bool has_accel_filt_ = false;
  Vec3 accel_filt_ = Vec3::Zero();
  Vec3 br_ = Vec3::Zero(), br_dot_ = Vec3::Zero();
  Vec3 prev_br_cmd_ = Vec3::Zero();
};

}  // namespace rl_infer_cpp
