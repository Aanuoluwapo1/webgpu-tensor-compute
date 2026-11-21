#include "wgpu_compute/ops.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef WGPU_COMPUTE_SHADER_DIR
#define WGPU_COMPUTE_SHADER_DIR "shaders"
#endif

namespace {

int g_failures = 0;

void check_cond(bool condition, const std::string& test_name, const std::string& detail = "") {
    if (condition) {
        std::cout << "[PASS] " << test_name << "\n";
    } else {
        std::cerr << "[FAIL] " << test_name;
        if (!detail.empty()) {
            std::cerr << " (" << detail << ")";
        }
        std::cerr << "\n";
        ++g_failures;
    }
}

void test_buffer_alignment() {
    using namespace wgpu_compute;

    check_cond(align_up(1, 256) == 256, "align_up(1, 256) == 256");
    check_cond(align_up(256, 256) == 256, "align_up(256, 256) == 256");
    check_cond(align_up(257, 256) == 512, "align_up(257, 256) == 512");

    const auto info = compute_buffer_alignment(33, sizeof(float)); // 132 bytes raw
    check_cond(
        info.raw_byte_size == 132 &&
        info.copy_aligned_bytes == 132 &&
        info.binding_aligned_bytes == 256 &&
        info.padding_bytes == 124,
        "compute_buffer_alignment 256-byte uniform/storage offset calculation"
    );
}

void test_wgsl_shader_files() {
    using namespace wgpu_compute;
    const std::string dir = WGPU_COMPUTE_SHADER_DIR;
    const std::vector<std::string> shaders = {"rmsnorm.wgsl", "rope.wgsl", "q4_gemv.wgsl"};

    for (const auto& name : shaders) {
        const auto report = validate_wgsl_shader_file(name, dir + "/" + name);
        check_cond(
            report.exists && report.has_compute_entry && report.has_workgroup_size && report.byte_size > 100,
            "WGSL shader validation: " + name
        );
    }
}

void test_rmsnorm_parity_and_edges() {
    using namespace wgpu_compute;

    // 1. Non-power-of-two column dimension (e.g., 17 x 517)
    const std::size_t rows = 17;
    const std::size_t cols = 517;
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> x(rows * cols);
    std::vector<float> gamma(cols);
    for (auto& v : x) v = dist(rng);
    for (auto& g : gamma) g = dist(rng);

    std::vector<float> y_wgsl;
    std::vector<float> y_golden;
    rmsnorm_wgsl_f32(x, gamma, y_wgsl, rows, cols, 1e-5f);
    rmsnorm_golden_f64(x, gamma, y_golden, rows, cols, 1e-5f);

    const auto metrics = compute_parity_metrics(y_wgsl, y_golden, 1e-5f);
    check_cond(
        metrics.passed && metrics.cosine_similarity > 0.999999f,
        "RMSNorm WGSL f32 tree reduction vs FP64 golden (17x517)",
        "max_abs_err=" + std::to_string(metrics.max_abs_err)
    );

    // 2. Zero input tensor with epsilon guard (must not produce NaN or Inf)
    std::vector<float> zeros(4 * 64, 0.0f);
    std::vector<float> ones(64, 1.0f);
    std::vector<float> y_zero;
    rmsnorm_wgsl_f32(zeros, ones, y_zero, 4, 64, 1e-5f);
    bool all_zero_finite = true;
    for (float v : y_zero) {
        if (!std::isfinite(v) || std::abs(v) > 1e-7f) {
            all_zero_finite = false;
            break;
        }
    }
    check_cond(all_zero_finite, "RMSNorm zero-input epsilon stability guard");
}

void test_rope_parity_and_invariants() {
    using namespace wgpu_compute;

    const std::size_t seq_len = 19;
    const std::size_t num_heads = 6;
    const std::size_t head_dim = 64;
    const std::size_t pos_offset = 128;

    std::mt19937 rng(9876u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(seq_len * num_heads * head_dim);
    for (auto& v : x) v = dist(rng);

    std::vector<float> cos_table;
    std::vector<float> sin_table;
    rope_precompute_cos_sin(cos_table, sin_table, seq_len, head_dim, pos_offset, 10000.0f);

    std::vector<float> y_wgsl;
    std::vector<float> y_golden;
    rope_wgsl_f32(x, cos_table, sin_table, y_wgsl, seq_len, num_heads, head_dim);
    rope_golden_f64(x, y_golden, seq_len, num_heads, head_dim, pos_offset, 10000.0f);

    const auto metrics = compute_parity_metrics(y_wgsl, y_golden, 1e-5f);
    check_cond(
        metrics.passed && metrics.cosine_similarity > 0.999999f,
        "RoPE WGSL f32 vs FP64 golden with non-zero pos_offset (19x6x64)"
    );

    // Verify 2D rotation preserves L2 norm per pair
    bool norm_preserved = true;
    for (std::size_t i = 0; i < x.size(); i += 2) {
        const float n_in = std::hypot(x[i], x[i + 1]);
        const float n_out = std::hypot(y_wgsl[i], y_wgsl[i + 1]);
        if (std::abs(n_in - n_out) > 1e-5f) {
            norm_preserved = false;
            break;
        }
    }
    check_cond(norm_preserved, "RoPE orthogonal pair rotation L2-norm preservation");
}

void test_q4_0_and_int8_gemv() {
    using namespace wgpu_compute;

    // 1. Exact Q4_0 nibble packing and dequantization parity with non-multiple-of-32 columns
    const std::size_t rows = 23;
    const std::size_t cols = 100; // 3 full blocks of 32 + 1 partial block of 4 elements
    std::mt19937 rng(555u);
    std::uniform_real_distribution<float> dist(-1.4f, 1.4f);

    std::vector<float> weights_f32(rows * cols);
    std::vector<float> input_x(cols);
    for (auto& w : weights_f32) w = dist(rng);
    for (auto& x : input_x) x = dist(rng);

    const auto q4_mat = quantize_f32_to_q4_0(weights_f32, rows, cols);
    check_cond(q4_mat.blocks_per_row == 4, "Q4_0 partial block count for cols=100 is 4");

    std::vector<float> y_wgsl;
    std::vector<float> y_golden;
    q4_gemv_wgsl_f32(q4_mat, input_x, y_wgsl);
    q4_gemv_golden_f64(q4_mat, input_x, y_golden);

    const auto exact_dequant_parity = compute_parity_metrics(y_wgsl, y_golden, 1e-5f);
    check_cond(
        exact_dequant_parity.passed,
        "Q4_0 WGSL nibble-unpacking GEMV vs FP64 dequantized GEMV (23x100)"
    );

    // 2. Compare quantized Q4_0 and per-channel INT8 GEMV against uncompressed FP32 GEMV
    std::vector<float> y_fp32(rows, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            acc += static_cast<double>(weights_f32[r * cols + c]) * static_cast<double>(input_x[c]);
        }
        y_fp32[r] = static_cast<float>(acc);
    }

    const auto q4_vs_fp32 = compute_parity_metrics(y_wgsl, y_fp32, 1.5f);
    check_cond(
        q4_vs_fp32.cosine_similarity > 0.99f,
        "Q4_0 GEMV cosine similarity vs uncompressed FP32 (> 0.99)",
        "cosine=" + std::to_string(q4_vs_fp32.cosine_similarity)
    );

    const auto i8_mat = quantize_f32_to_int8_per_channel(weights_f32, rows, cols);
    std::vector<float> y_i8;
    int8_gemv_f32(i8_mat, input_x, y_i8);
    const auto i8_vs_fp32 = compute_parity_metrics(y_i8, y_fp32, 0.25f);
    check_cond(
        i8_vs_fp32.cosine_similarity > 0.9999f,
        "Per-channel INT8 GEMV cosine similarity vs uncompressed FP32 (> 0.9999)",
        "cosine=" + std::to_string(i8_vs_fp32.cosine_similarity)
    );

    // 3. Zero-scale edge case: all-zero weight matrix
    std::vector<float> zero_weights(16 * 64, 0.0f);
    std::vector<float> x_64(64, 1.0f);
    const auto q4_zero = quantize_f32_to_q4_0(zero_weights, 16, 64);
    std::vector<float> y_zero;
    q4_gemv_wgsl_f32(q4_zero, x_64, y_zero);
    bool all_zero = true;
    for (float v : y_zero) {
        if (v != 0.0f) {
            all_zero = false;
            break;
        }
    }
    check_cond(all_zero, "Q4_0 zero-scale matrix dequantizes and multiplies to exact 0.0f");
}

void test_input_validation_and_error_handling() {
    using namespace wgpu_compute;

    // 1. Odd head_dim in RoPE must throw std::invalid_argument
    bool threw_odd_head_dim = false;
    try {
        std::vector<float> c, s;
        rope_precompute_cos_sin(c, s, 4, 63);
    } catch (const std::invalid_argument&) {
        threw_odd_head_dim = true;
    }
    check_cond(threw_odd_head_dim, "RoPE rejects odd head_dim with std::invalid_argument");

    // 2. Non-positive epsilon in RMSNorm must throw std::invalid_argument
    bool threw_bad_eps = false;
    try {
        std::vector<float> x(16, 1.0f), g(16, 1.0f), y;
        rmsnorm_wgsl_f32(x, g, y, 1, 16, 0.0f);
    } catch (const std::invalid_argument&) {
        threw_bad_eps = true;
    }
    check_cond(threw_bad_eps, "RMSNorm rejects non-positive epsilon with std::invalid_argument");

    // 3. Corrupted Q4MatrixStorage buffer size must throw std::invalid_argument
    bool threw_corrupted_q4 = false;
    try {
        Q4MatrixStorage bad_q4;
        bad_q4.rows = 4;
        bad_q4.cols = 32;
        bad_q4.blocks_per_row = 1;
        bad_q4.block_scales.resize(2); // Should be 4
        bad_q4.packed_words.resize(16);
        std::vector<float> x(32, 1.0f), y;
        q4_gemv_wgsl_f32(bad_q4, x, y);
    } catch (const std::invalid_argument&) {
        threw_corrupted_q4 = true;
    }
    check_cond(threw_corrupted_q4, "Q4_0 GEMV rejects malformed storage buffers with std::invalid_argument");
}

void test_live_gpu_dispatch_parity() {
    using namespace wgpu_compute;
    const auto status = query_gpu_adapter_status();
    if (!status.available) {
        return;
    }
    const std::string dir = WGPU_COMPUTE_SHADER_DIR;

    // 1. Live GPU RMSNorm on non-multiple shape (17 x 517)
    {
        const std::size_t rows = 17, cols = 517;
        std::mt19937 rng(42u);
        std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
        std::vector<float> x(rows * cols), gamma(cols), y_gpu, y_golden;
        for (auto& v : x) v = dist(rng);
        for (auto& g : gamma) g = dist(rng);

        bool used_gpu = false;
        dispatch_rmsnorm_gpu(dir + "/rmsnorm.wgsl", x, gamma, y_gpu, rows, cols, 1e-5f, 1, &used_gpu);
        rmsnorm_golden_f64(x, gamma, y_golden, rows, cols, 1e-5f);
        const auto m = compute_parity_metrics(y_gpu, y_golden, 1e-5f);
        check_cond(used_gpu && m.passed, "Live GPU WebGPU rmsnorm.wgsl dispatch parity vs FP64 (" + status.adapter_name + ")");
    }

    // 2. Live GPU RoPE on (19 x 6 x 64)
    {
        const std::size_t S = 19, H = 6, D = 64;
        std::mt19937 rng(99u);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> x(S * H * D), cos_t, sin_t, y_gpu, y_golden;
        for (auto& v : x) v = dist(rng);
        rope_precompute_cos_sin(cos_t, sin_t, S, D, 0, 10000.0f);

        bool used_gpu = false;
        dispatch_rope_gpu(dir + "/rope.wgsl", x, cos_t, sin_t, y_gpu, S, H, D, 1, &used_gpu);
        rope_golden_f64(x, y_golden, S, H, D, 0, 10000.0f);
        const auto m = compute_parity_metrics(y_gpu, y_golden, 1e-5f);
        check_cond(used_gpu && m.passed, "Live GPU WebGPU rope.wgsl dispatch parity vs FP64 (" + status.adapter_name + ")");
    }

    // 3. Live GPU Q4_0 GEMV on (23 x 100)
    {
        const std::size_t rows = 23, cols = 100;
        std::mt19937 rng(777u);
        std::uniform_real_distribution<float> dist(-1.2f, 1.2f);
        std::vector<float> w_f32(rows * cols), x(cols), y_gpu, y_golden;
        for (auto& w : w_f32) w = dist(rng);
        for (auto& v : x) v = dist(rng);
        const auto q4 = quantize_f32_to_q4_0(w_f32, rows, cols);

        bool used_gpu = false;
        dispatch_q4_gemv_gpu(dir + "/q4_gemv.wgsl", q4, x, y_gpu, 1, &used_gpu);
        q4_gemv_golden_f64(q4, x, y_golden);
        const auto m = compute_parity_metrics(y_gpu, y_golden, 1e-4f);
        check_cond(used_gpu && m.passed, "Live GPU WebGPU q4_gemv.wgsl dispatch parity vs FP64 (" + status.adapter_name + ")");
    }
}

} // namespace

int main() {
    std::cout << "=== Running webgpu-tensor-compute Unit Test Suite ===\n";
    test_buffer_alignment();
    test_wgsl_shader_files();
    test_rmsnorm_parity_and_edges();
    test_rope_parity_and_invariants();
    test_q4_0_and_int8_gemv();
    test_input_validation_and_error_handling();
    test_live_gpu_dispatch_parity();

    if (g_failures == 0) {
        std::cout << "=== ALL TESTS PASSED (0 failures) ===\n";
        return 0;
    }
    std::cerr << "=== TEST SUITE FAILED (" << g_failures << " failures) ===\n";
    return 1;
}
