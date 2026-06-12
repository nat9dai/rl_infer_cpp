// policy.hpp: Eigen MLP forward pass over .npw weights exported by
// rl_infer/scripts/export_policy_cpp.py (format spec lives there). Replicates
// the deployed NumPy backend op-for-op in float32: arch 0 = ELU MLP + clip
// (hover); arch 1 = Flax ELU+LayerNorm + tanh (gate). No heap in infer().
#pragma once

#include <sys/stat.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rl_infer_cpp {

class PolicyMlp {
 public:
  using RowVec = Eigen::RowVectorXf;
  using MatRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>;

  void load(const std::string& path) {
    check_not_stale(path);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("PolicyMlp: cannot open " + path);
    // RAII close: the throws below must not leak the handle (gate node
    // catches per-roll load failures and keeps running).
    struct FileCloser {
      void operator()(std::FILE* fp) const { std::fclose(fp); }
    };
    std::unique_ptr<std::FILE, FileCloser> guard(f);
    char magic[4];
    must_read(f, magic, 4, path);
    if (std::string(magic, 4) != "RLNW")
      throw std::runtime_error("PolicyMlp: bad magic in " + path);
    const uint32_t version = read_u32(f, path);
    if (version != 1)
      throw std::runtime_error("PolicyMlp: unsupported version in " + path);
    arch_ = read_u32(f, path);
    act_ = read_u32(f, path);
    if (arch_ > 1 || act_ > 1)
      throw std::runtime_error("PolicyMlp: bad arch/act field in " + path);
    num_obs_ = read_u32(f, path);
    num_act_ = read_u32(f, path);
    const uint32_t n_hidden = read_u32(f, path);
    has_norm_ = read_u32(f, path) != 0;
    if (has_norm_) {
      mean_ = read_vec(f, num_obs_, path);
      denom_ = read_vec(f, num_obs_, path);
    }
    W_.clear(); b_.clear(); ln_scale_.clear(); ln_bias_.clear();
    for (uint32_t i = 0; i < n_hidden; ++i) {
      const uint32_t in = read_u32(f, path), out = read_u32(f, path);
      W_.push_back(read_mat(f, in, out, path));
      b_.push_back(read_vec(f, out, path));
      if (arch_ == 1) {
        ln_scale_.push_back(read_vec(f, out, path));
        ln_bias_.push_back(read_vec(f, out, path));
      }
    }
    const uint32_t in = read_u32(f, path), out = read_u32(f, path);
    out_W_ = read_mat(f, in, out, path);
    out_b_ = read_vec(f, out, path);
    guard.reset();  // close now; nothing below reads the file
    if (out != num_act_)
      throw std::runtime_error("PolicyMlp: output dim mismatch in " + path);
    // Validate the layer dim chain: a corrupt dims field would otherwise
    // make infer() read past the scratch buffer (silent UB).
    uint32_t prev = num_obs_;
    for (const auto& w : W_) {
      if (static_cast<uint32_t>(w.rows()) != prev)
        throw std::runtime_error("PolicyMlp: layer dim chain mismatch in "
                                 + path);
      prev = static_cast<uint32_t>(w.cols());
    }
    if (static_cast<uint32_t>(out_W_.rows()) != prev)
      throw std::runtime_error("PolicyMlp: output in-dim mismatch in " + path);

    // Preallocate ping-pong scratch buffers sized to the widest layer.
    int widest = static_cast<int>(num_obs_);
    for (const auto& w : W_) widest = std::max<int>(widest, w.cols());
    scratch_a_.resize(widest);
    scratch_b_.resize(widest);
    out_buf_.resize(num_act_);
    path_ = path;
  }

  // obs: num_obs floats. Returns num_act floats valid until the next call.
  // Not thread-safe (single caller in the node's timer thread).
  const float* infer(const float* obs) {
    Eigen::Map<const RowVec> x_in(obs, num_obs_);
    RowVec* cur = &scratch_a_;
    RowVec* nxt = &scratch_b_;

    if (has_norm_) {
      cur->head(num_obs_) = (x_in - mean_).cwiseQuotient(denom_);
    } else {
      cur->head(num_obs_) = x_in;
    }
    int width = static_cast<int>(num_obs_);

    for (size_t i = 0; i < W_.size(); ++i) {
      const int out_w = static_cast<int>(W_[i].cols());
      nxt->head(out_w).noalias() = cur->head(width) * W_[i];
      nxt->head(out_w) += b_[i];
      // ELU (act_==0) or ReLU
      if (act_ == 0) {
        float* p = nxt->data();
        for (int j = 0; j < out_w; ++j)
          p[j] = p[j] > 0.0f ? p[j] : std::expm1f(p[j]);
      } else {
        nxt->head(out_w) = nxt->head(out_w).cwiseMax(0.0f);
      }
      if (arch_ == 1) {  // flax LayerNorm: var = max(E[x²]-E[x]², 0), eps 1e-6
        const auto seg = nxt->head(out_w);
        const float mu = seg.mean();
        const float ex2 = seg.cwiseProduct(seg).mean();
        const float var = std::max(ex2 - mu * mu, 0.0f);
        const float inv = 1.0f / std::sqrt(var + 1e-6f);
        nxt->head(out_w) =
            ((seg.array() - mu) * inv).matrix().cwiseProduct(ln_scale_[i])
            + ln_bias_[i];
      }
      std::swap(cur, nxt);
      width = out_w;
    }

    out_buf_.noalias() = cur->head(width) * out_W_;
    out_buf_ += out_b_;
    if (arch_ == 1) {
      for (int j = 0; j < out_buf_.size(); ++j)
        out_buf_[j] = std::tanh(out_buf_[j]);
    } else {
      out_buf_ = out_buf_.cwiseMax(-1.0f).cwiseMin(1.0f);
    }
    return out_buf_.data();
  }

  uint32_t num_obs() const { return num_obs_; }
  uint32_t num_act() const { return num_act_; }
  const std::string& path() const { return path_; }
  const char* arch_name() const {
    return arch_ == 0 ? "plain_mlp/clip" : "flax_layernorm/tanh";
  }

 private:
  // Refuse an .npw older than its sibling .pkl (retrained but never
  // re-exported). No .pkl sibling is fine (npw-only deployment).
  static void check_not_stale(const std::string& npw_path) {
    const std::string ext = ".npw";
    if (npw_path.size() < ext.size()
        || npw_path.compare(npw_path.size() - ext.size(), ext.size(), ext) != 0)
      return;
    std::string pkl_path = npw_path;
    pkl_path.replace(pkl_path.size() - ext.size(), ext.size(), ".pkl");
    struct stat pkl_st, npw_st;
    if (::stat(pkl_path.c_str(), &pkl_st) != 0) return;  // no .pkl sibling
    if (::stat(npw_path.c_str(), &npw_st) != 0) return;  // open() will report
    if (pkl_st.st_mtime > npw_st.st_mtime)
      throw std::runtime_error(
          "PolicyMlp: STALE weights: " + pkl_path
          + " is newer than its exported twin " + npw_path
          + ". Re-export before flying: python3 "
            "rl_infer/scripts/export_policy_cpp.py <pkl> <num_obs> <num_act> "
            "&& colcon build --packages-select rl_infer");
  }

  static void must_read(std::FILE* f, void* dst, size_t n,
                        const std::string& path) {
    if (std::fread(dst, 1, n, f) != n)
      throw std::runtime_error("PolicyMlp: truncated file " + path);
  }
  static uint32_t read_u32(std::FILE* f, const std::string& path) {
    uint32_t v;
    must_read(f, &v, sizeof(v), path);
    return v;
  }
  static RowVec read_vec(std::FILE* f, uint32_t n, const std::string& path) {
    RowVec v(n);
    must_read(f, v.data(), sizeof(float) * n, path);
    return v;
  }
  static MatRM read_mat(std::FILE* f, uint32_t rows, uint32_t cols,
                        const std::string& path) {
    MatRM m(rows, cols);
    must_read(f, m.data(), sizeof(float) * rows * cols, path);
    return m;
  }

  uint32_t arch_ = 0, act_ = 0, num_obs_ = 0, num_act_ = 0;
  bool has_norm_ = false;
  RowVec mean_, denom_;
  std::vector<MatRM> W_;
  std::vector<RowVec> b_, ln_scale_, ln_bias_;
  MatRM out_W_;
  RowVec out_b_;
  RowVec scratch_a_, scratch_b_, out_buf_;
  std::string path_;
};

}  // namespace rl_infer_cpp
