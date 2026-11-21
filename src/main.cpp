#include "wgpu_compute/ops.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef WGPU_COMPUTE_SHADER_DIR
#define WGPU_COMPUTE_SHADER_DIR "shaders"
#endif

namespace {

struct CliConfig {
    std::string op = "all";
    std::size_t rows = 64;
    std::size_t cols = 1024;
    std::size_t seq_len = 64;
    std::size_t num_heads = 8;
    std::size_t head_dim = 64;
    int iterations = 10;
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --op <rmsnorm|rope|q4_gemv|all>   Kernel suite to verify & benchmark (default: all)\n"
              << "  --rows <N>                        Row count for RMSNorm / Q4_GEMV (default: 64)\n"
              << "  --cols <N>                        Column dimension for RMSNorm / Q4_GEMV (default: 1024)\n"
              << "  --seq-len <N>                     Sequence length for RoPE (default: 64)\n"
              << "  --num-heads <N>                   Attention head count for RoPE (default: 8)\n"
              << "  --head-dim <N>                    Even head dimension for RoPE (default: 64)\n"
              << "  --iterations <N>                  Timing iterations per kernel (default: 10)\n"
              << "  --help                            Show this help message\n";
}

bool parse_positive_size(const std::string& flag, const std::string& val, std::size_t& out) {
    try {
        if (!val.empty() && val[0] == '-') {
            std::cerr << "Invalid negative value for " << flag << ": " << val << "\n";
            return false;
        }
        std::size_t idx = 0;
        const unsigned long parsed = std::stoul(val, &idx);
        if (idx != val.size() || parsed == 0) {
            std::cerr << "Invalid value for " << flag << ": '" << val << "' (must be a positive integer).\n";
            return false;
        }
        out = static_cast<std::size_t>(parsed);
        return true;
    } catch (const std::exception&) {
        std::cerr << "Failed to parse integer for " << flag << ": '" << val << "'.\n";
        return false;
    }
}

bool parse_args(int argc, char** argv, CliConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--op" && i + 1 < argc) {
            cfg.op = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            if (!parse_positive_size("--rows", argv[++i], cfg.rows)) return false;
        } else if (arg == "--cols" && i + 1 < argc) {
            if (!parse_positive_size("--cols", argv[++i], cfg.cols)) return false;
        } else if (arg == "--seq-len" && i + 1 < argc) {
            if (!parse_positive_size("--seq-len", argv[++i], cfg.seq_len)) return false;
        } else if (arg == "--num-heads" && i + 1 < argc) {
            if (!parse_positive_size("--num-heads", argv[++i], cfg.num_heads)) return false;
        } else if (arg == "--head-dim" && i + 1 < argc) {
            if (!parse_positive_size("--head-dim", argv[++i], cfg.head_dim)) return false;
        } else if (arg == "--iterations" && i + 1 < argc) {
            std::size_t iters = 0;
            if (!parse_positive_size("--iterations", argv[++i], iters)) return false;
            cfg.iterations = static_cast<int>(iters);
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    if (cfg.op != "rmsnorm" && cfg.op != "rope" && cfg.op != "q4_gemv" && cfg.op != "all") {
        std::cerr << "Invalid --op value: '" << cfg.op << "'. Must be rmsnorm, rope, q4_gemv, or all.\n";
        return false;
    }
    if (cfg.head_dim == 0 || cfg.head_dim % 2 != 0) {
        std::cerr << "Invalid --head-dim: must be a positive even integer.\n";
        return false;
    }
    return true;
}

std::vector<float> make_deterministic_tensor(std::size_t count, std::uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

struct KernelBenchRow {
    std::string kernel_name;
    std::string shape_desc;
    double avg_latency_ms;
    std::size_t aligned_buffer_bytes;
    wgpu_compute::ParityMetrics parity;
};

} // namespace

int main(int argc, char** argv) {
    CliConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    const auto gpu_status = wgpu_compute::query_gpu_adapter_status();

    std::cout << "====================================================================================\n";
    std::cout << "  webgpu-tensor-compute | Cross-Platform WGSL LLM Kernel & GPU/CPU Parity Harness\n";
    std::cout << "====================================================================================\n";
    if (gpu_status.available) {
        std::cout << "  Execution Backend : Live Hardware WebGPU Compute Pipeline (wgpu-native vs. FP64 Reference)\n";
        std::cout << "  Hardware Adapter  : " << gpu_status.adapter_name
                  << " (" << gpu_status.backend_name
                  << ", VendorID=0x" << std::hex << gpu_status.vendor_id
                  << ", DeviceID=0x" << gpu_status.device_id << std::dec << ")\n";
    } else {
        std::cout << "  Execution Backend : Deterministic CPU WGSL-Workgroup Simulator vs. FP64 Reference\n";
        std::cout << "  Hardware Adapter  : No live Dawn/wgpu-native GPU adapter attached (Zero fabricated telemetry)\n";
    }
    std::cout << "  Uniform Alignment : " << wgpu_compute::kWebGpuUniformBufferAlignment << " bytes | Q4_0 Block Size: "
              << wgpu_compute::kQ4BlockSize << "\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    // Validate WGSL shader files on disk
    const std::string shader_dir = WGPU_COMPUTE_SHADER_DIR;
    const std::vector<wgpu_compute::ShaderValidationReport> reports = {
        wgpu_compute::validate_wgsl_shader_file("rmsnorm.wgsl", shader_dir + "/rmsnorm.wgsl"),
        wgpu_compute::validate_wgsl_shader_file("rope.wgsl", shader_dir + "/rope.wgsl"),
        wgpu_compute::validate_wgsl_shader_file("q4_gemv.wgsl", shader_dir + "/q4_gemv.wgsl")
    };

    std::cout << "  WGSL Shader Source Verification:\n";
    for (const auto& rep : reports) {
        const bool ok = rep.exists && rep.has_compute_entry && rep.has_workgroup_size;
        std::cout << "    - " << std::left << std::setw(16) << rep.shader_name
                  << " | Status: " << (ok ? "VALID WGSL" : "MISSING/INVALID")
                  << " | Size: " << std::setw(5) << rep.byte_size << " B\n";
    }
    std::cout << "------------------------------------------------------------------------------------\n";

    std::vector<KernelBenchRow> results;
    bool all_passed = true;

    // 1. RMSNorm
    if (cfg.op == "rmsnorm" || cfg.op == "all") {
        const auto x = make_deterministic_tensor(cfg.rows * cfg.cols, 42u, 1.5f);
        const auto gamma = make_deterministic_tensor(cfg.cols, 1337u, 1.0f);
        std::vector<float> y_wgsl;
        std::vector<float> y_golden;

        wgpu_compute::rmsnorm_golden_f64(x, gamma, y_golden, cfg.rows, cfg.cols, 1e-5f);

        bool used_gpu = false;
        const double avg_ms = wgpu_compute::dispatch_rmsnorm_gpu(
            shader_dir + "/rmsnorm.wgsl", x, gamma, y_wgsl, cfg.rows, cfg.cols, 1e-5f, cfg.iterations, &used_gpu);

        const auto parity = wgpu_compute::compute_parity_metrics(y_wgsl, y_golden, 1e-5f);
        const auto align_in = wgpu_compute::compute_buffer_alignment(x.size(), sizeof(float));
        const auto align_w = wgpu_compute::compute_buffer_alignment(gamma.size(), sizeof(float));
        const auto align_out = wgpu_compute::compute_buffer_alignment(y_wgsl.size(), sizeof(float));

        results.push_back(KernelBenchRow{
            used_gpu ? "RMSNorm (GPU WGSL vs f64)" : "RMSNorm (WGSL f32 vs f64)",
            std::to_string(cfg.rows) + "x" + std::to_string(cfg.cols),
            avg_ms,
            align_in.binding_aligned_bytes + align_w.binding_aligned_bytes + align_out.binding_aligned_bytes,
            parity
        });
        all_passed = all_passed && parity.passed;
    }

    // 2. RoPE
    if (cfg.op == "rope" || cfg.op == "all") {
        const std::size_t total_elems = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        const auto x = make_deterministic_tensor(total_elems, 101u, 1.0f);
        std::vector<float> cos_table;
        std::vector<float> sin_table;
        std::vector<float> y_wgsl;
        std::vector<float> y_golden;

        wgpu_compute::rope_precompute_cos_sin(cos_table, sin_table, cfg.seq_len, cfg.head_dim, 0, 10000.0f);
        wgpu_compute::rope_golden_f64(x, y_golden, cfg.seq_len, cfg.num_heads, cfg.head_dim, 0, 10000.0f);

        bool used_gpu = false;
        const double avg_ms = wgpu_compute::dispatch_rope_gpu(
            shader_dir + "/rope.wgsl", x, cos_table, sin_table, y_wgsl,
            cfg.seq_len, cfg.num_heads, cfg.head_dim, cfg.iterations, &used_gpu);

        const auto parity = wgpu_compute::compute_parity_metrics(y_wgsl, y_golden, 1e-5f);
        const auto align_x = wgpu_compute::compute_buffer_alignment(x.size(), sizeof(float));
        const auto align_cs = wgpu_compute::compute_buffer_alignment(cos_table.size(), sizeof(float));

        results.push_back(KernelBenchRow{
            used_gpu ? "RoPE (GPU WGSL vs f64)" : "RoPE (WGSL f32 vs f64)",
            std::to_string(cfg.seq_len) + "x" + std::to_string(cfg.num_heads) + "x" + std::to_string(cfg.head_dim),
            avg_ms,
            align_x.binding_aligned_bytes * 2 + align_cs.binding_aligned_bytes * 2,
            parity
        });
        all_passed = all_passed && parity.passed;
    }

    // 3. Q4_0 Block GEMV
    if (cfg.op == "q4_gemv" || cfg.op == "all") {
        const auto weights_f32 = make_deterministic_tensor(cfg.rows * cfg.cols, 2024u, 0.8f);
        const auto input_x = make_deterministic_tensor(cfg.cols, 777u, 1.0f);
        const auto q4_mat = wgpu_compute::quantize_f32_to_q4_0(weights_f32, cfg.rows, cfg.cols);

        std::vector<float> y_wgsl;
        std::vector<float> y_dequant_f64;
        wgpu_compute::q4_gemv_golden_f64(q4_mat, input_x, y_dequant_f64);

        bool used_gpu = false;
        const double avg_ms = wgpu_compute::dispatch_q4_gemv_gpu(
            shader_dir + "/q4_gemv.wgsl", q4_mat, input_x, y_wgsl, cfg.iterations, &used_gpu);

        const auto parity = wgpu_compute::compute_parity_metrics(y_wgsl, y_dequant_f64, 1e-4f);
        const auto align_scales = wgpu_compute::compute_buffer_alignment(q4_mat.block_scales.size(), sizeof(float));
        const auto align_words = wgpu_compute::compute_buffer_alignment(q4_mat.packed_words.size(), sizeof(std::uint32_t));
        const auto align_x = wgpu_compute::compute_buffer_alignment(input_x.size(), sizeof(float));
        const auto align_y = wgpu_compute::compute_buffer_alignment(y_wgsl.size(), sizeof(float));

        results.push_back(KernelBenchRow{
            used_gpu ? "Q4_0 GEMV (GPU WGSL vs f64)" : "Q4_0 GEMV (WGSL vs Dequant)",
            std::to_string(cfg.rows) + "x" + std::to_string(cfg.cols),
            avg_ms,
            align_scales.binding_aligned_bytes + align_words.binding_aligned_bytes +
                align_x.binding_aligned_bytes + align_y.binding_aligned_bytes,
            parity
        });
        all_passed = all_passed && parity.passed;
    }

    // Print summary table
    std::cout << std::left
              << std::setw(28) << "Kernel Operation"
              << std::setw(14) << "Shape"
              << std::setw(12) << (gpu_status.available ? "GPU (ms)" : "CPU (ms)")
              << std::setw(14) << "MaxAbsErr"
              << std::setw(14) << "RMSE"
              << std::setw(12) << "CosineSim"
              << "Status\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (const auto& row : results) {
        std::cout << std::left << std::setw(28) << row.kernel_name
                  << std::setw(14) << row.shape_desc
                  << std::fixed << std::setprecision(4) << std::setw(12) << row.avg_latency_ms
                  << std::scientific << std::setprecision(3) << std::setw(14) << row.parity.max_abs_err
                  << std::setw(14) << row.parity.rmse
                  << std::fixed << std::setprecision(6) << std::setw(12) << row.parity.cosine_similarity
                  << (row.parity.passed ? "PASS" : "FAIL") << "\n";
    }
    std::cout << "====================================================================================\n";

    return all_passed ? 0 : 1;
}
