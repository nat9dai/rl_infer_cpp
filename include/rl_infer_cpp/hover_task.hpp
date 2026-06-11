// hover_task.hpp: C++ port of rl_infer/hover/task.py (HoverTaskLogic).
// Pure logic, no ROS. Double-precision math, float32 obs output, exactly
// mirroring the Python reference (which computes in float64 then casts).
//
// Deploy-config env vars (same names/defaults as the Python task):
//   HOVER_THRUST_REMAP (default 1) : cubic gz thrust-curve remap
//   HOVER_RATE_CAP     (default 1) : per-axis cap to the SITL envelope
//   HOVER_MAX_RATE_ROLL/PITCH/YAW   (defaults 4.0 / 2.5 / 3.0 rad/s)
//   HOVER_SLEW         (default 1) : per-tick body-rate slew limit
//   HOVER_SLEW_ROLL/PITCH/YAW       (defaults 110 / 50 / 25 rad/s²)
#pragma once

#include <cstdlib>
#include <string>

#include "rl_infer_cpp/math_util.hpp"

namespace rl_infer_cpp {

constexpr int NUM_HOVER_OBS = 31;
constexpr int NUM_HOVER_ACT = 4;
constexpr double POLICY_ENU_YAW_OFFSET_RAD = 0.5 * M_PI;

inline double env_double(const char* name, double dflt) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atof(v) : dflt;
}

inline bool env_flag(const char* name, bool dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  return std::string(v) == "1";
}

struct HoverActionScale {
  double min_throttle = 0.15;
  double max_throttle = 0.40;
  double max_body_rate_xy = 6.0;
  double max_body_rate_z = 6.0;
};

struct HoverTarget {
  Vec3 pos = Vec3::Zero();
  Vec4 quat = Vec4(1, 0, 0, 0);  // wxyz
  Mat3 rotmat = Mat3::Identity();

  // From [x, y, z, roll, pitch, yaw] in policy ENU/FLU.
  static HoverTarget from_policy_pose(const double pose[6]) {
    HoverTarget t;
    t.pos = Vec3(pose[0], pose[1], pose[2]);
    t.quat = rpy_to_quat_zyx(pose[3], pose[4], pose[5]);
    t.rotmat = quat_to_rotmat(t.quat);
    return t;
  }
};

struct RatesCommand {
  double roll = 0.0, pitch = 0.0, yaw = 0.0, thrust_z = 0.0;
};

class HoverTaskLogic {
 public:
  HoverTaskLogic(const HoverActionScale& scale, bool apply_policy_enu_yaw_offset)
      : scale_(scale),
        apply_yaw_offset_(apply_policy_enu_yaw_offset),
        yaw_offset_quat_(yaw_to_quat(POLICY_ENU_YAW_OFFSET_RAD)) {
    rate_cap_on_ = env_flag("HOVER_RATE_CAP", true);
    rate_cap_ = Vec3(env_double("HOVER_MAX_RATE_ROLL", 4.0),
                     env_double("HOVER_MAX_RATE_PITCH", 2.5),
                     env_double("HOVER_MAX_RATE_YAW", 3.0));
    slew_on_ = env_flag("HOVER_SLEW", true);
    slew_ = Vec3(env_double("HOVER_SLEW_ROLL", 110.0),
                 env_double("HOVER_SLEW_PITCH", 50.0),
                 env_double("HOVER_SLEW_YAW", 25.0));
    thrust_remap_on_ = env_flag("HOVER_THRUST_REMAP", true);
  }

  void reset() { prev_br_cmd_.setZero(); }

  void set_target_policy_pose(const double pose[6]) {
    target_ = HoverTarget::from_policy_pose(pose);
  }

  // Set target from PX4-style NED position and NED/FRD wxyz attitude
  // ("absolute" target mode).
  void set_target_ned_frd(const Vec3& pos_ned, const Vec4& quat_wxyz_ned_frd) {
    HoverTarget t;
    t.pos = ned_to_enu_vec(pos_ned);
    Vec4 q = ned_frd_quat_to_enu_flu(quat_wxyz_ned_frd);
    if (apply_yaw_offset_) q = quat_mul(q, yaw_offset_quat_);
    t.quat = q;
    t.rotmat = quat_to_rotmat(q);
    target_ = t;
  }

  const HoverTarget& target() const { return target_; }

  // 31-D obs: rot_mat_w(9) lin_vel_b(3) ang_vel_b(3) target_pos_b(3)
  //           target_ori_w(9) last_action(4). Mirrors build_obs_from_state.
  void build_obs(const Vec3& position_w, const Vec3& velocity_w,
                 const Vec4& quat_wxyz, const Vec3& ang_vel_b,
                 const float last_action[NUM_HOVER_ACT],
                 float out[NUM_HOVER_OBS]) const {
    Vec4 dq = quat_wxyz;
    if (apply_yaw_offset_) dq = quat_mul(dq, yaw_offset_quat_);

    const Mat3 rot = quat_to_rotmat(dq);
    int k = 0;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) out[k++] = static_cast<float>(rot(r, c));

    const Vec4 dq_inv = quat_inv(dq);
    const Vec3 lin_vel_b = quat_apply(dq_inv, velocity_w);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(lin_vel_b[i]);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(ang_vel_b[i]);

    const Vec3 rel = target_.pos - position_w;
    const Vec3 tgt_b = quat_apply(dq_inv, rel);
    for (int i = 0; i < 3; ++i) out[k++] = static_cast<float>(tgt_b[i]);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        out[k++] = static_cast<float>(target_.rotmat(r, c));

    for (int i = 0; i < NUM_HOVER_ACT; ++i) out[k++] = last_action[i];
  }

  // Raw [-1,1] action -> body rates (FLU) + positive-up normalized thrust.
  RatesCommand scale_action(const float raw[NUM_HOVER_ACT]) {
    double a[4];
    for (int i = 0; i < 4; ++i)
      a[i] = std::min(1.0, std::max(-1.0, static_cast<double>(raw[i])));
    double thrust = scale_.min_throttle
        + (a[0] + 1.0) * 0.5 * (scale_.max_throttle - scale_.min_throttle);
    if (thrust_remap_on_)
      thrust = remap_throttle(thrust, scale_.min_throttle, scale_.max_throttle);
    Vec3 br(a[1] * scale_.max_body_rate_xy,
            a[2] * scale_.max_body_rate_xy,
            a[3] * scale_.max_body_rate_z);
    if (rate_cap_on_)
      br = br.cwiseMax(-rate_cap_).cwiseMin(rate_cap_);
    if (slew_on_) {
      const Vec3 step = slew_ * ctrl_dt_;
      br = prev_br_cmd_
           + (br - prev_br_cmd_).cwiseMax(-step).cwiseMin(step);
      prev_br_cmd_ = br;
    }
    RatesCommand cmd;
    cmd.roll = br[0];
    cmd.pitch = br[1];
    cmd.yaw = br[2];
    cmd.thrust_z =
        std::min(scale_.max_throttle, std::max(scale_.min_throttle, thrust));
    return cmd;
  }

 private:
  // _hover_remap_throttle: cubic gz thrust-curve correction, clipped.
  static double remap_throttle(double thr, double lo, double hi) {
    constexpr double c0 = 1.613356, c1 = -2.856781, c2 = 2.09172,
                     c3 = -0.106211;
    const double t = c0 * thr * thr * thr + c1 * thr * thr + c2 * thr + c3;
    return std::min(hi, std::max(lo, t));
  }

  HoverActionScale scale_;
  bool apply_yaw_offset_;
  Vec4 yaw_offset_quat_;
  HoverTarget target_;
  double ctrl_dt_ = 0.01;  // 100 Hz, hard-coded like the Python task
  Vec3 prev_br_cmd_ = Vec3::Zero();
  bool rate_cap_on_ = true, slew_on_ = true, thrust_remap_on_ = true;
  Vec3 rate_cap_, slew_;
};

}  // namespace rl_infer_cpp
