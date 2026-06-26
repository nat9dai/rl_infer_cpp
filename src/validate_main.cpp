// validate_main.cpp: offline numerical validation + latency benchmark for
// the C++ inference port. No ROS.
// replay: closed-loop replay of rl_infer/test/cpp/gen_vectors.py vectors vs
//   the Python reference; run with the same deploy env vars the generator
//   used (file layout spec lives in gen_vectors.py).
// bench: per-inference latency stats (mean/p50/p99/max microseconds).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rl_infer_cpp/gate_task.hpp"
#include "rl_infer_cpp/hover_task.hpp"
#include "rl_infer_cpp/policy.hpp"

using namespace rl_infer_cpp;

namespace {

void must_read(std::FILE* f, void* dst, size_t n) {
  if (std::fread(dst, 1, n, f) != n) {
    std::fprintf(stderr, "truncated vector file\n");
    std::exit(2);
  }
}
uint32_t read_u32(std::FILE* f) {
  uint32_t v;
  must_read(f, &v, 4);
  return v;
}
double read_f64(std::FILE* f) {
  double v;
  must_read(f, &v, 8);
  return v;
}

struct DiffTracker {
  double max_obs = 0, max_raw = 0, max_cmd = 0;
  int worst_obs_step = -1, worst_raw_step = -1, worst_cmd_step = -1;
  void track(double& mx, int& at, double d, int step) {
    if (d > mx) { mx = d; at = step; }
  }
};

int replay(const std::string& vec_path, const std::string& npw_path) {
  std::FILE* f = std::fopen(vec_path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", vec_path.c_str()); return 2; }
  char magic[4];
  must_read(f, magic, 4);
  if (std::memcmp(magic, "RLTV", 4) != 0) {
    std::fprintf(stderr, "bad magic\n");
    return 2;
  }
  const uint32_t task = read_u32(f);
  const uint32_t N = read_u32(f);

  PolicyMlp policy;
  policy.load(npw_path);
  const int num_obs = static_cast<int>(policy.num_obs());

  std::unique_ptr<HoverTaskLogic> hover;
  std::unique_ptr<GateTaskLogic> gate;
  if (task == 0) {
    const uint32_t yaw_off = read_u32(f);
    double pose[6];
    for (double& v : pose) v = read_f64(f);
    HoverActionScale scale;  // deploy defaults (yaml: 0.15/0.40/6/6)
    hover = std::make_unique<HoverTaskLogic>(scale, yaw_off != 0);
    hover->set_target_policy_pose(pose);
    hover->reset();
    if (num_obs != NUM_HOVER_OBS) { std::fprintf(stderr, "obs dim\n"); return 2; }
  } else {
    Vec3 gate_pos, standby, goal_local;
    for (int i = 0; i < 3; ++i) gate_pos[i] = read_f64(f);
    for (int i = 0; i < 3; ++i) standby[i] = read_f64(f);
    const double roll = read_f64(f);
    for (int i = 0; i < 3; ++i) goal_local[i] = read_f64(f);
    const double lpf = read_f64(f);
    const uint32_t br_filter = read_u32(f);
    const GateScene scene =
        GateScene::from_standby(gate_pos, standby, roll, goal_local);
    gate = std::make_unique<GateTaskLogic>(scene, 0.01, lpf, br_filter != 0);
    if (num_obs != NUM_GATE_OBS) { std::fprintf(stderr, "obs dim\n"); return 2; }
  }

  std::vector<float> obs(num_obs), exp_obs(num_obs);
  float last_action[4] = {0, 0, 0, 0};
  float exp_raw[4];
  double exp_cmd[4];
  DiffTracker dt;

  for (uint32_t s = 0; s < N; ++s) {
    Vec3 pos, vel, angvel;
    Vec4 quat;
    for (int i = 0; i < 3; ++i) pos[i] = read_f64(f);
    for (int i = 0; i < 3; ++i) vel[i] = read_f64(f);
    for (int i = 0; i < 4; ++i) quat[i] = read_f64(f);
    if (task == 0)
      for (int i = 0; i < 3; ++i) angvel[i] = read_f64(f);
    must_read(f, exp_obs.data(), sizeof(float) * num_obs);
    must_read(f, exp_raw, sizeof(float) * 4);
    must_read(f, exp_cmd, sizeof(double) * 4);

    if (task == 0)
      hover->build_obs(pos, vel, quat, angvel, last_action, obs.data());
    else
      gate->build_obs(pos, vel, quat, last_action, 1, obs.data());  // hist_steps=1 (28-D validator)

    for (int i = 0; i < num_obs; ++i)
      dt.track(dt.max_obs, dt.worst_obs_step,
               std::abs(static_cast<double>(obs[i]) - exp_obs[i]), s);

    const float* raw = policy.infer(obs.data());
    for (int i = 0; i < 4; ++i)
      dt.track(dt.max_raw, dt.worst_raw_step,
               std::abs(static_cast<double>(raw[i]) - exp_raw[i]), s);

    const RatesCommand cmd =
        task == 0 ? hover->scale_action(raw) : gate->scale_action(raw);
    const double got[4] = {cmd.roll, cmd.pitch, cmd.yaw, cmd.thrust_z};
    for (int i = 0; i < 4; ++i)
      dt.track(dt.max_cmd, dt.worst_cmd_step, std::abs(got[i] - exp_cmd[i]), s);

    for (int i = 0; i < 4; ++i)
      last_action[i] = std::min(1.0f, std::max(-1.0f, raw[i]));
  }
  std::fclose(f);

  const char* name = task == 0 ? "hover" : "gate";
  std::printf("%s replay: %u steps closed-loop\n", name, N);
  std::printf("  max |obs diff| = %.3e (step %d)\n", dt.max_obs,
              dt.worst_obs_step);
  std::printf("  max |raw diff| = %.3e (step %d)\n", dt.max_raw,
              dt.worst_raw_step);
  std::printf("  max |cmd diff| = %.3e (step %d)\n", dt.max_cmd,
              dt.worst_cmd_step);
  const bool pass = dt.max_obs < 1e-5 && dt.max_raw < 1e-4 && dt.max_cmd < 1e-4;
  std::printf("  %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int bench(const std::string& npw_path, int iters) {
  PolicyMlp policy;
  policy.load(npw_path);
  const int num_obs = static_cast<int>(policy.num_obs());

  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> obs_set(256, std::vector<float>(num_obs));
  for (auto& o : obs_set)
    for (auto& v : o) v = dist(rng);

  volatile float sink = 0;
  for (int i = 0; i < 100; ++i) sink += policy.infer(obs_set[i % 256].data())[0];

  std::vector<double> lat(iters);
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    sink += policy.infer(obs_set[i % 256].data())[0];
    lat[i] = std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - t0)
                 .count();
  }
  (void)sink;
  std::sort(lat.begin(), lat.end());
  double mean = 0;
  for (double v : lat) mean += v;
  mean /= iters;
  std::printf(
      "{\"backend\": \"cpp\", \"ckpt\": \"%s\", \"mean_us\": %.1f, "
      "\"p50_us\": %.1f, \"p99_us\": %.1f, \"max_us\": %.1f, \"iters\": %d}\n",
      npw_path.c_str(), mean, lat[iters / 2],
      lat[static_cast<int>(iters * 0.99)], lat[iters - 1], iters);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "replay")
    return replay(argv[2], argv[3]);
  if (argc >= 3 && std::string(argv[1]) == "bench")
    return bench(argv[2], argc > 3 ? std::atoi(argv[3]) : 5000);
  std::fprintf(stderr,
               "usage: rl_validate replay <vectors.bin> <policy.npw>\n"
               "       rl_validate bench <policy.npw> [iters]\n");
  return 2;
}
