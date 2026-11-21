#ifndef WGPU_COMPUTE_OPS_HPP
#define WGPU_COMPUTE_OPS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace wgpu_compute {

// WebGPU standard constants
constexpr std::size_t kWebGpuUniformBufferAlignment = 256;
constexpr std::size_t kWebGpuStorageBufferOffsetAlignment = 256;
constexpr std::size_t kWebGpuCopyBufferAlignment = 4;
constexpr std::size_t kQ4BlockSize = 32;
constexpr std::size_t kQ4BytesPerBlock = 16;
constexpr std::size_t kQ4WordsPerBlock = 4;

/**
 * Buffer layout and WebGPU alignment descriptor.
 */
struct BufferAlignmentInfo {
    std::size_t raw_byte_size;
    std::size_t copy_aligned_bytes;      // Multiple of 4 bytes (WebGPU writeBuffer / copyBufferToBuffer)
    std::size_t binding_aligned_bytes;   // Multiple of 256 bytes (WebGPU dynamic uniform/storage binding offset)
    std::size_t padding_bytes;
};

/**
 * Rounds `byte_size` up to the next multiple of `alignment` (must be power of two).
 */
std::size_t align_up(std::size_t byte_size, std::size_t alignment) noexcept;

/**
 * Computes WebGPU alignment metadata for a buffer of `element_count` items of `element_size` bytes.
 */
BufferAlignmentInfo compute_buffer_alignment(std::size_t element_count, std::size_t element_size) noexcept;

/**
 * Numerical parity metrics between candidate and baseline tensors.
 */
struct ParityMetrics {
    float max_abs_err;
    float rmse;
    float cosine_similarity;
    bool passed;
};

/**
 * Computes numerical error metrics (Max Absolute Error, RMSE, Cosine Similarity)
 * between two float vectors of equal length.
 */
ParityMetrics compute_parity_metrics(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float abs_tol = 1e-5f
);

/**
 * Static WGSL shader integrity check result.
 */
struct ShaderValidationReport {
    std::string shader_name;
    std::string file_path;
    bool exists;
    bool has_compute_entry;
    bool has_workgroup_size;
    std::size_t byte_size;
};

/**
 * Validates that a WGSL shader file exists on disk and contains `@compute` and `@workgroup_size`.
 */
ShaderValidationReport validate_wgsl_shader_file(
    const std::string& shader_name,
    const std::string& file_path
);

// ============================================================================
// 1. RMSNorm Reference & Simulated WGSL Workgroup Execution
// ============================================================================

/**
 * Direct float64-accumulated golden reference for RMSNorm:
 *   y[r, c] = (x[r, c] / sqrt(mean(x[r, :]^2) + eps)) * weight[c]
 */
void rmsnorm_golden_f64(
    const std::vector<float>& input_x,
    const std::vector<float>& weight_gamma,
    std::vector<float>& output_y,
    std::size_t rows,
    std::size_t cols,
    float eps = 1e-5f
);

/**
 * Exact WGSL workgroup-faithful f32 implementation matching `shaders/rmsnorm.wgsl`
 * (256-lane strided accumulation + power-of-two shared-memory tree reduction).
 */
void rmsnorm_wgsl_f32(
    const std::vector<float>& input_x,
    const std::vector<float>& weight_gamma,
    std::vector<float>& output_y,
    std::size_t rows,
    std::size_t cols,
    float eps = 1e-5f
);

// ============================================================================
// 2. Rotary Position Embedding (RoPE)
// ============================================================================

/**
 * Precomputes RoPE cos and sin tables of shape [seq_len, head_dim / 2]
 * with frequency base theta (default 10000.0f) and starting position offset.
 */
void rope_precompute_cos_sin(
    std::vector<float>& cos_table,
    std::vector<float>& sin_table,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t pos_offset = 0,
    float theta_base = 10000.0f
);

/**
 * Direct float64 golden reference for RoPE over [seq_len, num_heads, head_dim].
 */
void rope_golden_f64(
    const std::vector<float>& input_x,
    std::vector<float>& output_y,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t pos_offset = 0,
    float theta_base = 10000.0f
);

/**
 * Exact WGSL-faithful f32 RoPE implementation matching `shaders/rope.wgsl`.
 */
void rope_wgsl_f32(
    const std::vector<float>& input_x,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    std::vector<float>& output_y,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t head_dim
);

// ============================================================================
// 3. Q4_0 Block Quantization (block_size = 32) & Per-Channel INT8 GEMV
// ============================================================================

/**
 * Host representation of packed Q4_0 matrix buffers bound to `shaders/q4_gemv.wgsl`.
 * Each block of 32 weights stores:
 *   - 1 float32 scale `d` in `block_scales`
 *   - 4 uint32 words (32 x 4-bit unsigned nibbles in [0, 15], zero-point = 8) in `packed_words`
 */
struct Q4MatrixStorage {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t blocks_per_row = 0;
    std::vector<float> block_scales;
    std::vector<std::uint32_t> packed_words;
};

/**
 * Quantizes a row-major float32 matrix [rows, cols] into Q4_0 blocks (block_size = 32).
 * Handles arbitrary `cols` by zero-padding partial trailing blocks.
 */
Q4MatrixStorage quantize_f32_to_q4_0(
    const std::vector<float>& weights_f32,
    std::size_t rows,
    std::size_t cols
);

/**
 * Dequantizes a `Q4MatrixStorage` back to a row-major float32 matrix [rows, cols].
 */
void dequantize_q4_0_to_f32(
    const Q4MatrixStorage& q4_mat,
    std::vector<float>& weights_out
);

/**
 * Executes Q4_0 matrix-vector multiplication matching `shaders/q4_gemv.wgsl`:
 *   y[row] = sum_{col} dequant(W_q4[row, col]) * x[col]
 */
void q4_gemv_wgsl_f32(
    const Q4MatrixStorage& q4_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
);

/**
 * Float64 golden reference for Q4_0 GEMV using explicitly dequantized weights.
 */
void q4_gemv_golden_f64(
    const Q4MatrixStorage& q4_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
);

/**
 * Per-channel symmetric INT8 matrix-vector multiply reference:
 *   W_f32[r, c] ~ int8_weights[r, c] * row_scales[r]
 */
struct Int8MatrixStorage {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> row_scales;
    std::vector<std::int8_t> weights_i8;
};

Int8MatrixStorage quantize_f32_to_int8_per_channel(
    const std::vector<float>& weights_f32,
    std::size_t rows,
    std::size_t cols
);

void int8_gemv_f32(
    const Int8MatrixStorage& i8_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
);

} // namespace wgpu_compute

#endif // WGPU_COMPUTE_OPS_HPP
