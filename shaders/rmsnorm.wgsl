// ============================================================================
// WebGPU Compute Shader: RMSNorm (Root Mean Square Layer Normalization)
// Target Specification: WGSL (WebGPU Shading Language)
// ============================================================================
// Computes per-row RMS normalization:
//   rms(x) = sqrt( (1 / cols) * sum_{i=0}^{cols-1} x[i]^2 + eps )
//   y[i]   = (x[i] / rms(x)) * weight[i]
//
// Dispatch geometry:
//   1 workgroup per token/row -> dispatchWorkgroups(rows, 1, 1)
//   Workgroup size: 256 threads performing strided accumulation + tree reduction
//   in workgroup shared memory.
//
// Alignment Notes:
//   Params uniform buffer is padded to 16-byte struct alignment (bound at
//   256-byte dynamic uniform offset boundary by host WebGPU pipeline).
// ============================================================================

struct RmsNormParams {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: RmsNormParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> weight_gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> shared_sq_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    let tid = local_id.x;
    let cols = params.cols;

    if (row >= params.rows || cols == 0u) {
        return;
    }

    let row_offset = row * cols;

    // Phase 1: Strided f32 sum-of-squares accumulation per thread
    var thread_sum: f32 = 0.0;
    var col: u32 = tid;
    loop {
        if (col >= cols) {
            break;
        }
        let val = input_x[row_offset + col];
        thread_sum = thread_sum + (val * val);
        col = col + WORKGROUP_SIZE;
    }

    shared_sq_sum[tid] = thread_sum;
    workgroupBarrier();

    // Phase 2: Unrolled power-of-two tree reduction in workgroup shared memory.
    // All workgroupBarrier() calls are placed in uniform workgroup control flow
    // to strictly satisfy the WGSL uniformity analysis specification.
    if (tid < 128u) { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u)  { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u)  { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u)  { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u)   { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u)   { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u)   { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u)   { shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + 1u]; }
    workgroupBarrier();

    // Phase 3: Compute inverse RMS with epsilon guard and normalize row
    let mean_sq = shared_sq_sum[0] / f32(cols);
    let inv_rms = inverseSqrt(mean_sq + params.eps);

    col = tid;
    loop {
        if (col >= cols) {
            break;
        }
        let x_val = input_x[row_offset + col];
        let w_val = weight_gamma[col];
        output_y[row_offset + col] = (x_val * inv_rms) * w_val;
        col = col + WORKGROUP_SIZE;
    }
}
