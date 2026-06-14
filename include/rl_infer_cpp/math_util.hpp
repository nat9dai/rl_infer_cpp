// math_util.hpp: C++ port of rl_infer/util/math.py + the gate task's
// scalar-last quaternion helpers. Double math like the Python reference, so
// observations agree to float32 rounding. Quaternion conventions: Hamilton
// [w,x,y,z] for the hover task; scalar-last [x,y,z,w] for the gate task.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace rl_infer_cpp {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;  // quaternion storage; order per function contract
using Mat3 = Eigen::Matrix3d;

// ───────────────────────── Hamilton [w,x,y,z] helpers ─────────────────────────

inline Vec4 quat_inv(const Vec4& q) {  // unit quaternion conjugate
  return Vec4(q[0], -q[1], -q[2], -q[3]);
}

inline Vec4 quat_mul(const Vec4& q1, const Vec4& q2) {
  const double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  return Vec4(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
              w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);
}

// Rotate vector v by quaternion q (w,x,y,z): exactly util.math.quat_apply
// (two Hamilton products, take the vector part).
inline Vec3 quat_apply(const Vec4& q, const Vec3& v) {
  const Vec4 qv(0.0, v[0], v[1], v[2]);
  const Vec4 r = quat_mul(quat_mul(q, qv), quat_inv(q));
  return Vec3(r[1], r[2], r[3]);
}

inline Mat3 quat_to_rotmat(const Vec4& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  Mat3 m;
  m << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
       2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
       2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y);
  return m;
}

// Branchy Shepperd recovery: port of util.math.rotmat_to_quat (wxyz out).
inline Vec4 rotmat_to_quat(const Mat3& m) {
  const double trace = m.trace();
  Vec4 q;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q = Vec4(0.25 * s,
             (m(2, 1) - m(1, 2)) / s,
             (m(0, 2) - m(2, 0)) / s,
             (m(1, 0) - m(0, 1)) / s);
  } else if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2)) {
    const double s = std::sqrt(1.0 + m(0, 0) - m(1, 1) - m(2, 2)) * 2.0;
    q = Vec4((m(2, 1) - m(1, 2)) / s, 0.25 * s,
             (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s);
  } else if (m(1, 1) > m(2, 2)) {
    const double s = std::sqrt(1.0 + m(1, 1) - m(0, 0) - m(2, 2)) * 2.0;
    q = Vec4((m(0, 2) - m(2, 0)) / s, (m(0, 1) + m(1, 0)) / s,
             0.25 * s, (m(1, 2) + m(2, 1)) / s);
  } else {
    const double s = std::sqrt(1.0 + m(2, 2) - m(0, 0) - m(1, 1)) * 2.0;
    q = Vec4((m(1, 0) - m(0, 1)) / s, (m(0, 2) + m(2, 0)) / s,
             (m(1, 2) + m(2, 1)) / s, 0.25 * s);
  }
  return q / std::max(q.norm(), 1e-12);
}

// Euler (roll,pitch,yaw), ZYX order -> quaternion (w,x,y,z).
inline Vec4 rpy_to_quat_zyx(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll / 2), sr = std::sin(roll / 2);
  const double cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
  const double cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
  return Vec4(cr * cp * cy + sr * sp * sy,
              sr * cp * cy - cr * sp * sy,
              cr * sp * cy + sr * cp * sy,
              cr * cp * sy - sr * sp * cy);
}

inline Vec4 yaw_to_quat(double yaw) {
  return Vec4(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2));
}

// Yaw of a Hamilton wxyz quaternion (atan2 form used at hover engage).
inline double quat_wxyz_yaw(const Vec4& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

// ─────────────────────────── frame conversions ───────────────────────────────
// NED_TO_ENU swaps x/y and negates z; FRD_TO_FLU = diag(1,-1,-1) (involution).

inline Vec3 ned_to_enu_vec(const Vec3& v) { return Vec3(v[1], v[0], -v[2]); }
inline Vec3 enu_to_ned_vec(const Vec3& v) { return Vec3(v[1], v[0], -v[2]); }
inline Vec3 frd_to_flu_vec(const Vec3& v) { return Vec3(v[0], -v[1], -v[2]); }
inline Vec3 flu_to_frd_vec(const Vec3& v) { return Vec3(v[0], -v[1], -v[2]); }

inline Mat3 ned_to_enu_mat() {
  Mat3 m;
  m << 0, 1, 0,
       1, 0, 0,
       0, 0, -1;
  return m;
}

inline Mat3 frd_to_flu_mat() {
  return Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
}

// Body-to-world attitude quaternion NED/FRD -> ENU/FLU (Hamilton wxyz).
inline Vec4 ned_frd_quat_to_enu_flu(const Vec4& q_ned_frd) {
  const Mat3 r = ned_to_enu_mat() * quat_to_rotmat(q_ned_frd)
                 * frd_to_flu_mat().transpose();
  return rotmat_to_quat(r);
}

// ───────────────── scalar-last [x,y,z,w] helpers (gate task) ─────────────────

// v_world = R(q) v_body: matches ju.quat_apply (cross-product form).
inline Vec3 quat_apply_xyzw(const Vec4& q, const Vec3& v) {
  const Vec3 xyz(q[0], q[1], q[2]);
  const double w = q[3];
  const Vec3 t = 2.0 * xyz.cross(v);
  return v + w * t + xyz.cross(t);
}

// v_body = R(q)^T v_world: matches ju.quat_rotate_inverse.
inline Vec3 quat_rotate_inverse_xyzw(const Vec4& q, const Vec3& v) {
  const Vec3 xyz(q[0], q[1], q[2]);
  const double w = q[3];
  const Vec3 t = 2.0 * xyz.cross(v);
  return v - w * t + xyz.cross(t);
}

// ZYX Tait-Bryan -> [x,y,z,w]; matches dva_utils.jax_utils.euler_to_quat.
inline Vec4 euler_to_quat_xyzw(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  Vec4 q(sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy,
         cr * cp * cy + sr * sp * sy);
  return q / std::max(q.norm(), 1e-9);
}

// Hamilton [w,x,y,z] -> scalar-last [x,y,z,w].
inline Vec4 wxyz_to_xyzw(const Vec4& q) { return Vec4(q[1], q[2], q[3], q[0]); }

// Quaternion product q ⊗ r, scalar-last [x,y,z,w] (compose r in q's frame).
inline Vec4 quat_mul_xyzw(const Vec4& q, const Vec4& r) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double x2 = r[0], y2 = r[1], z2 = r[2], w2 = r[3];
  return Vec4(w * x2 + x * w2 + y * z2 - z * y2,
              w * y2 - x * z2 + y * w2 + z * x2,
              w * z2 + x * y2 - y * x2 + z * w2,
              w * w2 - x * x2 - y * y2 - z * z2);
}

// Yaw of an [x,y,z,w] quaternion (GateScene.gazebo_pose yaw formula).
inline double quat_xyzw_yaw(const Vec4& q) {
  const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  return std::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

}  // namespace rl_infer_cpp
