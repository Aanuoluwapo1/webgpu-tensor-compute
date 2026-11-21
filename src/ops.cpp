#include "wgpu_compute/ops.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wgpu_compute {

std::size_t align_up(std::size_t byte_size, std::size_t alignment) noexcept {
    if (alignment == 0) {
        return byte_size;
    }
    const std::size_t remainder = byte_size % alignment;
    return (remainder == 0) ? byte_size : (byte_size + alignment - remainder);
}

BufferAlignmentInfo compute_buffer_alignment(std::size_t element_count, std::size_t element_size) noexcept {
    if (element_count == 0 || element_size == 0) {
        return BufferAlignmentInfo{0, kWebGpuCopyBufferAlignment, kWebGpuUniformBufferAlignment, kWebGpuUniformBufferAlignment};
    }
    const std::size_t raw = element_count * element_size;
    const std::size_t copy_aligned = align_up(std::max<std::size_t>(raw, 4), kWebGpuCopyBufferAlignment);
    const std::size_t binding_aligned = align_up(std::max<std::size_t>(raw, 4), kWebGpuUniformBufferAlignment);
    return BufferAlignmentInfo{
        raw,
        copy_aligned,
        binding_aligned,
        binding_aligned - raw
    };
}

ParityMetrics compute_parity_metrics(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float abs_tol
) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("compute_parity_metrics: actual and expected vector sizes differ");
    }
    if (actual.empty()) {
        return ParityMetrics{0.0f, 0.0f, 1.0f, true};
    }

    double max_abs = 0.0;
    double sum_sq_err = 0.0;
    double dot_ab = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = static_cast<double>(actual[i]);
        const double b = static_cast<double>(expected[i]);
        const double err = std::abs(a - b);
        if (err > max_abs) {
            max_abs = err;
        }
        sum_sq_err += err * err;
        dot_ab += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    const double rmse = std::sqrt(sum_sq_err / static_cast<double>(actual.size()));
    double cosine = 1.0;
    if (norm_a > 0.0 && norm_b > 0.0) {
        cosine = dot_ab / (std::sqrt(norm_a) * std::sqrt(norm_b));
        cosine = std::clamp(cosine, -1.0, 1.0);
    } else if (norm_a == 0.0 && norm_b == 0.0) {
        cosine = 1.0;
    } else {
        cosine = 0.0;
    }

    const float max_abs_f = static_cast<float>(max_abs);
    const float rmse_f = static_cast<float>(rmse);
    const float cosine_f = static_cast<float>(cosine);
    const bool passed = (max_abs_f <= abs_tol) && !std::isnan(max_abs_f) && !std::isnan(rmse_f);

    return ParityMetrics{max_abs_f, rmse_f, cosine_f, passed};
}

ShaderValidationReport validate_wgsl_shader_file(
    const std::string& shader_name,
    const std::string& file_path
) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        return ShaderValidationReport{shader_name, file_path, false, false, false, 0};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string content = oss.str();

    const bool has_compute = content.find("@compute") != std::string::npos;
    const bool has_wg_size = content.find("@workgroup_size") != std::string::npos;

    return ShaderValidationReport{
        shader_name,
        file_path,
        true,
        has_compute,
        has_wg_size,
        content.size()
    };
}

// ============================================================================
// 1. RMSNorm Implementation
// ============================================================================

void rmsnorm_golden_f64(
    const std::vector<float>& input_x,
    const std::vector<float>& weight_gamma,
    std::vector<float>& output_y,
    std::size_t rows,
    std::size_t cols,
    float eps
) {
    if (eps <= 0.0f || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_golden_f64: eps must be positive and finite");
    }
    if (input_x.size() != rows * cols) {
        throw std::invalid_argument("rmsnorm_golden_f64: input_x size mismatch");
    }
    if (weight_gamma.size() != cols) {
        throw std::invalid_argument("rmsnorm_golden_f64: weight_gamma size mismatch");
    }
    output_y.resize(rows * cols);
    if (rows == 0 || cols == 0) {
        return;
    }

    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t row_offset = r * cols;
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = static_cast<double>(input_x[row_offset + c]);
            sum_sq += v * v;
        }
        const double mean_sq = sum_sq / static_cast<double>(cols);
        const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));

        for (std::size_t c = 0; c < cols; ++c) {
            const double x_val = static_cast<double>(input_x[row_offset + c]);
            const double w_val = static_cast<double>(weight_gamma[c]);
            output_y[row_offset + c] = static_cast<float>((x_val * inv_rms) * w_val);
        }
    }
}

void rmsnorm_wgsl_f32(
    const std::vector<float>& input_x,
    const std::vector<float>& weight_gamma,
    std::vector<float>& output_y,
    std::size_t rows,
    std::size_t cols,
    float eps
) {
    if (eps <= 0.0f || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm_wgsl_f32: eps must be positive and finite");
    }
    if (input_x.size() != rows * cols) {
        throw std::invalid_argument("rmsnorm_wgsl_f32: input_x size mismatch");
    }
    if (weight_gamma.size() != cols) {
        throw std::invalid_argument("rmsnorm_wgsl_f32: weight_gamma size mismatch");
    }
    output_y.resize(rows * cols);
    if (rows == 0 || cols == 0) {
        return;
    }

    constexpr std::size_t kWorkgroupSize = 256;
    float shared_sq_sum[kWorkgroupSize];

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t row_offset = row * cols;

        // Phase 1: 256-lane strided f32 accumulation
        for (std::size_t tid = 0; tid < kWorkgroupSize; ++tid) {
            float thread_sum = 0.0f;
            for (std::size_t col = tid; col < cols; col += kWorkgroupSize) {
                const float val = input_x[row_offset + col];
                thread_sum += val * val;
            }
            shared_sq_sum[tid] = thread_sum;
        }

        // Phase 2: Power-of-two tree reduction in shared memory
        for (std::size_t stride = kWorkgroupSize / 2; stride > 0; stride >>= 1) {
            for (std::size_t tid = 0; tid < stride; ++tid) {
                shared_sq_sum[tid] += shared_sq_sum[tid + stride];
            }
        }

        // Phase 3: Normalize using inverseSqrt
        const float mean_sq = shared_sq_sum[0] / static_cast<float>(cols);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (std::size_t tid = 0; tid < kWorkgroupSize; ++tid) {
            for (std::size_t col = tid; col < cols; col += kWorkgroupSize) {
                const float x_val = input_x[row_offset + col];
                const float w_val = weight_gamma[col];
                output_y[row_offset + col] = (x_val * inv_rms) * w_val;
            }
        }
    }
}

// ============================================================================
// 2. Rotary Position Embedding (RoPE) Implementation
// ============================================================================

void rope_precompute_cos_sin(
    std::vector<float>& cos_table,
    std::vector<float>& sin_table,
    std::size_t seq_len,
    std::size_t head_dim,
    std::size_t pos_offset,
    float theta_base
) {
    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("rope_precompute_cos_sin: head_dim must be a positive even integer");
    }
    if (theta_base <= 0.0f || !std::isfinite(theta_base)) {
        throw std::invalid_argument("rope_precompute_cos_sin: theta_base must be positive and finite");
    }
    const std::size_t half_dim = head_dim / 2;
    cos_table.resize(seq_len * half_dim);
    sin_table.resize(seq_len * half_dim);

    for (std::size_t seq = 0; seq < seq_len; ++seq) {
        const double pos = static_cast<double>(seq + pos_offset);
        for (std::size_t p = 0; p < half_dim; ++p) {
            const double exponent = -static_cast<double>(2 * p) / static_cast<double>(head_dim);
            const double inv_freq = std::pow(static_cast<double>(theta_base), exponent);
            const double angle = pos * inv_freq;
            cos_table[seq * half_dim + p] = static_cast<float>(std::cos(angle));
            sin_table[seq * half_dim + p] = static_cast<float>(std::sin(angle));
        }
    }
}

void rope_golden_f64(
    const std::vector<float>& input_x,
    std::vector<float>& output_y,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t pos_offset,
    float theta_base
) {
    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("rope_golden_f64: head_dim must be a positive even integer");
    }
    if (theta_base <= 0.0f || !std::isfinite(theta_base)) {
        throw std::invalid_argument("rope_golden_f64: theta_base must be positive and finite");
    }
    const std::size_t total = seq_len * num_heads * head_dim;
    if (input_x.size() != total) {
        throw std::invalid_argument("rope_golden_f64: input_x size mismatch");
    }
    output_y.resize(total);
    if (total == 0) {
        return;
    }
    const std::size_t half_dim = head_dim / 2;

    for (std::size_t seq = 0; seq < seq_len; ++seq) {
        const double pos = static_cast<double>(seq + pos_offset);
        for (std::size_t p = 0; p < half_dim; ++p) {
            const double exponent = -static_cast<double>(2 * p) / static_cast<double>(head_dim);
            const double inv_freq = std::pow(static_cast<double>(theta_base), exponent);
            const double angle = pos * inv_freq;
            const double c = std::cos(angle);
            const double s = std::sin(angle);

            for (std::size_t h = 0; h < num_heads; ++h) {
                const std::size_t base_idx = ((seq * num_heads + h) * head_dim) + (2 * p);
                const double x0 = static_cast<double>(input_x[base_idx]);
                const double x1 = static_cast<double>(input_x[base_idx + 1]);
                output_y[base_idx]     = static_cast<float>(x0 * c - x1 * s);
                output_y[base_idx + 1] = static_cast<float>(x0 * s + x1 * c);
            }
        }
    }
}

void rope_wgsl_f32(
    const std::vector<float>& input_x,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    std::vector<float>& output_y,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t head_dim
) {
    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("rope_wgsl_f32: head_dim must be a positive even integer");
    }
    const std::size_t total = seq_len * num_heads * head_dim;
    const std::size_t half_dim = head_dim / 2;
    if (input_x.size() != total) {
        throw std::invalid_argument("rope_wgsl_f32: input_x size mismatch");
    }
    if (cos_table.size() != seq_len * half_dim || sin_table.size() != seq_len * half_dim) {
        throw std::invalid_argument("rope_wgsl_f32: cos/sin table size mismatch");
    }

    output_y.resize(total);
    if (total == 0) {
        return;
    }
    const std::size_t total_pairs = seq_len * num_heads * half_dim;

    for (std::size_t pair_linear_idx = 0; pair_linear_idx < total_pairs; ++pair_linear_idx) {
        const std::size_t pair_idx = pair_linear_idx % half_dim;
        const std::size_t head_linear = pair_linear_idx / half_dim;
        const std::size_t head_idx = head_linear % num_heads;
        const std::size_t seq_idx = head_linear / num_heads;

        const std::size_t base_idx = ((seq_idx * num_heads + head_idx) * head_dim) + (2 * pair_idx);
        const std::size_t table_idx = (seq_idx * half_dim) + pair_idx;

        const float x0 = input_x[base_idx];
        const float x1 = input_x[base_idx + 1];
        const float c = cos_table[table_idx];
        const float s = sin_table[table_idx];

        output_y[base_idx]     = (x0 * c) - (x1 * s);
        output_y[base_idx + 1] = (x0 * s) + (x1 * c);
    }
}

// ============================================================================
// 3. Q4_0 Block Quantization & Per-Channel INT8 GEMV Implementation
// ============================================================================

Q4MatrixStorage quantize_f32_to_q4_0(
    const std::vector<float>& weights_f32,
    std::size_t rows,
    std::size_t cols
) {
    if (weights_f32.size() != rows * cols) {
        throw std::invalid_argument("quantize_f32_to_q4_0: weights_f32 size mismatch");
    }

    const std::size_t blocks_per_row = (cols + kQ4BlockSize - 1) / kQ4BlockSize;
    const std::size_t total_blocks = rows * blocks_per_row;

    Q4MatrixStorage storage;
    storage.rows = rows;
    storage.cols = cols;
    storage.blocks_per_row = blocks_per_row;
    storage.block_scales.assign(total_blocks, 0.0f);
    storage.packed_words.assign(total_blocks * kQ4WordsPerBlock, 0u);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::size_t block_idx = r * blocks_per_row + b;
            const std::size_t col_base = b * kQ4BlockSize;

            // Find max absolute value in block of 32 weights
            float max_abs = 0.0f;
            for (std::size_t i = 0; i < kQ4BlockSize; ++i) {
                const std::size_t c = col_base + i;
                if (c < cols) {
                    const float abs_v = std::abs(weights_f32[r * cols + c]);
                    if (abs_v > max_abs) {
                        max_abs = abs_v;
                    }
                }
            }

            // Symmetric Q4_0 scale: nibbles 0..15 shifted by -8 cover [-8*d, +7*d]
            // Use max_abs / 7.0f so both positive and negative peaks fit without clipping
            const float d = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.0f;
            const float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
            storage.block_scales[block_idx] = d;

            for (std::size_t w = 0; w < kQ4WordsPerBlock; ++w) {
                std::uint32_t packed_word = 0u;
                for (std::size_t nibble_idx = 0; nibble_idx < 8; ++nibble_idx) {
                    const std::size_t c = col_base + (w * 8) + nibble_idx;
                    std::uint32_t q_u32 = 8u; // zero-point = 8 represents 0.0f
                    if (c < cols && d > 0.0f) {
                        const float scaled = weights_f32[r * cols + c] * inv_d;
                        const int q_signed = static_cast<int>(std::nearbyint(scaled));
                        const int q_clamped = std::clamp(q_signed + 8, 0, 15);
                        q_u32 = static_cast<std::uint32_t>(q_clamped);
                    }
                    packed_word |= (q_u32 & 0xFu) << (nibble_idx * 4);
                }
                storage.packed_words[block_idx * kQ4WordsPerBlock + w] = packed_word;
            }
        }
    }

    return storage;
}

void dequantize_q4_0_to_f32(
    const Q4MatrixStorage& q4_mat,
    std::vector<float>& weights_out
) {
    const std::size_t expected_blocks_per_row = (q4_mat.cols + kQ4BlockSize - 1) / kQ4BlockSize;
    if (q4_mat.blocks_per_row != expected_blocks_per_row ||
        q4_mat.block_scales.size() != q4_mat.rows * q4_mat.blocks_per_row ||
        q4_mat.packed_words.size() != q4_mat.rows * q4_mat.blocks_per_row * kQ4WordsPerBlock) {
        throw std::invalid_argument("dequantize_q4_0_to_f32: malformed Q4MatrixStorage buffer sizes");
    }
    weights_out.assign(q4_mat.rows * q4_mat.cols, 0.0f);

    for (std::size_t r = 0; r < q4_mat.rows; ++r) {
        for (std::size_t b = 0; b < q4_mat.blocks_per_row; ++b) {
            const std::size_t block_idx = r * q4_mat.blocks_per_row + b;
            const float d = q4_mat.block_scales[block_idx];
            const std::size_t word_base = block_idx * kQ4WordsPerBlock;
            const std::size_t col_base = b * kQ4BlockSize;

            for (std::size_t w = 0; w < kQ4WordsPerBlock; ++w) {
                const std::uint32_t packed_word = q4_mat.packed_words[word_base + w];
                for (std::size_t nibble_idx = 0; nibble_idx < 8; ++nibble_idx) {
                    const std::size_t c = col_base + (w * 8) + nibble_idx;
                    if (c < q4_mat.cols) {
                        const std::uint32_t q_u32 = (packed_word >> (nibble_idx * 4)) & 0xFu;
                        weights_out[r * q4_mat.cols + c] = (static_cast<float>(q_u32) - 8.0f) * d;
                    }
                }
            }
        }
    }
}

void q4_gemv_wgsl_f32(
    const Q4MatrixStorage& q4_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
) {
    if (input_x.size() != q4_mat.cols) {
        throw std::invalid_argument("q4_gemv_wgsl_f32: input_x size mismatch");
    }
    const std::size_t expected_blocks_per_row = (q4_mat.cols + kQ4BlockSize - 1) / kQ4BlockSize;
    if (q4_mat.blocks_per_row != expected_blocks_per_row ||
        q4_mat.block_scales.size() != q4_mat.rows * q4_mat.blocks_per_row ||
        q4_mat.packed_words.size() != q4_mat.rows * q4_mat.blocks_per_row * kQ4WordsPerBlock) {
        throw std::invalid_argument("q4_gemv_wgsl_f32: malformed Q4MatrixStorage buffer sizes");
    }
    output_y.assign(q4_mat.rows, 0.0f);

    for (std::size_t row = 0; row < q4_mat.rows; ++row) {
        float acc = 0.0f;
        const std::size_t row_block_base = row * q4_mat.blocks_per_row;

        for (std::size_t b = 0; b < q4_mat.blocks_per_row; ++b) {
            const std::size_t block_idx = row_block_base + b;
            const float d = q4_mat.block_scales[block_idx];
            const std::size_t word_base = block_idx * kQ4WordsPerBlock;
            const std::size_t col_base = b * kQ4BlockSize;

            float block_dot = 0.0f;
            for (std::size_t w = 0; w < kQ4WordsPerBlock; ++w) {
                const std::uint32_t packed_word = q4_mat.packed_words[word_base + w];
                const std::size_t word_col_offset = col_base + (w * 8);

                for (std::size_t nibble_idx = 0; nibble_idx < 8; ++nibble_idx) {
                    const std::size_t col = word_col_offset + nibble_idx;
                    if (col < q4_mat.cols) {
                        const std::uint32_t q_u32 = (packed_word >> (nibble_idx * 4)) & 0xFu;
                        const float w_val = (static_cast<float>(q_u32) - 8.0f) * d;
                        block_dot += w_val * input_x[col];
                    }
                }
            }
            acc += block_dot;
        }
        output_y[row] = acc;
    }
}

void q4_gemv_golden_f64(
    const Q4MatrixStorage& q4_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
) {
    if (input_x.size() != q4_mat.cols) {
        throw std::invalid_argument("q4_gemv_golden_f64: input_x size mismatch");
    }
    std::vector<float> dequant_weights;
    dequantize_q4_0_to_f32(q4_mat, dequant_weights);

    output_y.assign(q4_mat.rows, 0.0f);
    for (std::size_t r = 0; r < q4_mat.rows; ++r) {
        double sum = 0.0;
        for (std::size_t c = 0; c < q4_mat.cols; ++c) {
            sum += static_cast<double>(dequant_weights[r * q4_mat.cols + c]) *
                   static_cast<double>(input_x[c]);
        }
        output_y[r] = static_cast<float>(sum);
    }
}

Int8MatrixStorage quantize_f32_to_int8_per_channel(
    const std::vector<float>& weights_f32,
    std::size_t rows,
    std::size_t cols
) {
    if (weights_f32.size() != rows * cols) {
        throw std::invalid_argument("quantize_f32_to_int8_per_channel: weights_f32 size mismatch");
    }

    Int8MatrixStorage storage;
    storage.rows = rows;
    storage.cols = cols;
    storage.row_scales.assign(rows, 0.0f);
    storage.weights_i8.assign(rows * cols, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        float max_abs = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) {
            const float v = std::abs(weights_f32[r * cols + c]);
            if (v > max_abs) {
                max_abs = v;
            }
        }
        const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 0.0f;
        const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        storage.row_scales[r] = scale;

        for (std::size_t c = 0; c < cols; ++c) {
            if (scale > 0.0f) {
                const int q = static_cast<int>(std::nearbyint(weights_f32[r * cols + c] * inv_scale));
                storage.weights_i8[r * cols + c] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
            }
        }
    }

    return storage;
}

void int8_gemv_f32(
    const Int8MatrixStorage& i8_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y
) {
    if (input_x.size() != i8_mat.cols) {
        throw std::invalid_argument("int8_gemv_f32: input_x size mismatch");
    }
    if (i8_mat.row_scales.size() != i8_mat.rows ||
        i8_mat.weights_i8.size() != i8_mat.rows * i8_mat.cols) {
        throw std::invalid_argument("int8_gemv_f32: malformed Int8MatrixStorage buffer sizes");
    }
    output_y.assign(i8_mat.rows, 0.0f);

    for (std::size_t r = 0; r < i8_mat.rows; ++r) {
        const float scale = i8_mat.row_scales[r];
        float dot = 0.0f;
        for (std::size_t c = 0; c < i8_mat.cols; ++c) {
            dot += static_cast<float>(i8_mat.weights_i8[r * i8_mat.cols + c]) * input_x[c];
        }
        output_y[r] = dot * scale;
    }
}

} // namespace wgpu_compute
