// ============================================================================
// WebGPU Compute Shader: Group-Quantized Q4_0 Matrix-Vector Multiply (GEMV)
// Target Specification: WGSL (WebGPU Shading Language)
// ============================================================================
// Computes y[row] = sum_{col=0}^{cols-1} dequant(W_q4[row, col]) * x[col]
//
// Q4_0 Block Format (block_size = 32 weights per block):
//   - scales[block_idx]: f32 block scale factor d
//   - packed_weights[block_idx * 4 + word_idx]: u32 holding 8 x 4-bit unsigned
//     nibbles (4 bytes per u32 word, 16 bytes = 32 nibbles per block).
//   - Each nibble q in [0, 15] is symmetrically dequantized with zero-point 8:
//       w_dequant = (f32(q) - 8.0) * d
//   - Layout inside each byte:
//       low nibble  (bits 0..3) -> element 2*k
//       high nibble (bits 4..7) -> element 2*k + 1
// ============================================================================

struct Q4GemvParams {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Q4GemvParams;
@group(0) @binding(1) var<storage, read> block_scales: array<f32>;
@group(0) @binding(2) var<storage, read> packed_q4_words: array<u32>;
@group(0) @binding(3) var<storage, read> input_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_y: array<f32>;

const BLOCK_SIZE: u32 = 32u;
const WORDS_PER_BLOCK: u32 = 4u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    var acc: f32 = 0.0;
    let row_block_base = row * params.blocks_per_row;

    for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
        let block_idx = row_block_base + b;
        let d = block_scales[block_idx];
        let word_base = block_idx * WORDS_PER_BLOCK;
        let col_base = b * BLOCK_SIZE;

        var block_dot: f32 = 0.0;

        for (var w: u32 = 0u; w < WORDS_PER_BLOCK; w = w + 1u) {
            let packed_word = packed_q4_words[word_base + w];
            let word_col_offset = col_base + (w * 8u);

            // Unpack 8 nibbles (4 bytes) from the 32-bit word
            for (var nibble_idx: u32 = 0u; nibble_idx < 8u; nibble_idx = nibble_idx + 1u) {
                let col = word_col_offset + nibble_idx;
                if (col < params.cols) {
                    let shift = nibble_idx * 4u;
                    let q_u32 = (packed_word >> shift) & 0xFu;
                    let w_val = (f32(q_u32) - 8.0) * d;
                    block_dot = block_dot + (w_val * input_x[col]);
                }
            }
        }

        acc = acc + block_dot;
    }

    output_y[row] = acc;
}
