// ============================================================================
// WebGPU Compute Shader: RoPE (Rotary Position Embedding)
// Target Specification: WGSL (WebGPU Shading Language)
// ============================================================================
// Applies rotary position embeddings to query/key activation tensors of shape
// [seq_len, num_heads, head_dim] using adjacent pair rotation:
//
//   For pair index p in [0, head_dim / 2):
//     freq_theta = (seq_idx + pos_offset) * inv_freq[p],
//     where inv_freq[p] = base^(-2*p / head_dim)
//     x0 = x[seq_idx, head, 2*p]
//     x1 = x[seq_idx, head, 2*p + 1]
//     y[seq_idx, head, 2*p]     = x0 * cos(freq_theta) - x1 * sin(freq_theta)
//     y[seq_idx, head, 2*p + 1] = x0 * sin(freq_theta) + x1 * cos(freq_theta)
//
// Host precomputes cos_table / sin_table of shape [seq_len, head_dim / 2]
// incorporating pos_offset, or binds full context tables offset by pos_offset.
// ============================================================================

struct RopeParams {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    pos_offset: u32,
};

@group(0) @binding(0) var<uniform> params: RopeParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read> sin_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.seq_len * params.num_heads * half_dim;
    let pair_linear_idx = gid.x;

    if (pair_linear_idx >= total_pairs || half_dim == 0u || params.num_heads == 0u) {
        return;
    }

    let pair_idx = pair_linear_idx % half_dim;
    let head_linear = pair_linear_idx / half_dim;
    let head_idx = head_linear % params.num_heads;
    let seq_idx = head_linear / params.num_heads;

    let base_idx = ((seq_idx * params.num_heads + head_idx) * params.head_dim) + (2u * pair_idx);
    let table_idx = (seq_idx * half_dim) + pair_idx;

    let x0 = input_x[base_idx];
    let x1 = input_x[base_idx + 1u];
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    output_y[base_idx]      = (x0 * c) - (x1 * s);
    output_y[base_idx + 1u] = (x0 * s) + (x1 * c);
}
