# `webgpu-tensor-compute`: Cross-Platform WebGPU (WGSL) Compute Shader Suite for LLM Inference Kernels

[![CI](https://github.com/Aanuoluwapo1/webgpu-tensor-compute/actions/workflows/ci.yml/badge.svg)](https://github.com/Aanuoluwapo1/webgpu-tensor-compute/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![WGSL](https://img.shields.io/badge/WebGPU-WGSL%201.0-ff6f00.svg)](https://www.w3.org/TR/WGSL/)
[![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**`webgpu-tensor-compute`** is a modular, dependency-free **WebGPU Shading Language (`WGSL`) Compute Shader Suite** and **C++17 Host Verification & Parity Engine** targeting core Transformer / LLM inference primitives:
- **`RMSNorm` (`shaders/rmsnorm.wgsl`)**: 256-thread workgroup strided `f32` accumulation + unrolled shared-memory power-of-two reduction tree with uniform `workgroupBarrier()` synchronization and `inverseSqrt` epsilon guard.
- **Rotary Position Embedding — `RoPE` (`shaders/rope.wgsl`)**: Adjacent-pair 2D orthogonal complex rotation across `[seq_len, num_heads, head_dim]` with KV-cache `pos_offset` support and precomputed `cos`/`sin` frequency tables.
- **Group-Quantized `Q4_0` & Per-Channel `INT8` GEMV (`shaders/q4_gemv.wgsl`)**: `llama.cpp` / GGUF-style 4-bit block quantization (`block_size = 32`, eight 4-bit unsigned nibbles bit-packed per 32-bit `u32` word with symmetric zero-point `8`) and fused nibble-unpacking matrix-vector multiply.
- **WebGPU Memory Alignment & Deterministic Parity Engine (`include/wgpu_compute/ops.hpp`, `src/ops.cpp`)**: 256-byte uniform/storage binding alignment helpers (`minUniformBufferOffsetAlignment = 256`), WGSL static source validator, workgroup-accurate single-precision (`f32`) CPU simulator, and double-precision (`f64`) golden reference verifier (`max_abs_err`, `rmse`, `cosine_similarity`).

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Shaders["WebGPU Compute Shader Suite (shaders/*.wgsl)"]
        RMS_WGSL["rmsnorm.wgsl (@workgroup_size 256,1,1 + var<workgroup> array<f32,256>)"]
        ROPE_WGSL["rope.wgsl (@workgroup_size 64,1,1 + Pairwise Complex Rotation)"]
        Q4_WGSL["q4_gemv.wgsl (@workgroup_size 64,1,1 + u32 Nibble Bit-Unpacking)"]
    end

    subgraph Host["C++17 Host & Alignment Engine (include/wgpu_compute/ops.hpp, src/ops.cpp)"]
        ALIGN["WebGPU Buffer Alignment (256B Uniform/Storage & 4B Copy Padding)"]
        VAL["WGSL Static Shader Validator (@compute & @workgroup_size Check)"]
        QUANT["Quantizers: Q4_0 (block_size=32, 4x u32/block) & Per-Channel INT8"]
    end

    subgraph Sim["Workgroup-Faithful f32 Simulator vs. FP64 Golden Reference"]
        SIM_F32["WGSL-Faithful f32 Kernels (256-Lane Tree Reduction & Bit-Shift Unpack)"]
        REF_F64["FP64 Golden Reference Kernels (64-Bit Accumulation)"]
        PARITY["Parity Verifier (MaxAbsErr, RMSE, Cosine Similarity)"]
        SIM_F32 & REF_F64 --> PARITY
    end

    Shaders --> VAL
    Host --> Sim
    VAL --> PARITY
```

---

## Kernel Formulations & WGSL Implementation Details

### 1. Root Mean Square Layer Normalization (`shaders/rmsnorm.wgsl`)

For an input activation matrix $X \in \mathbb{R}^{R \times C}$ and learned scale vector $\gamma \in \mathbb{R}^{C}$, `RMSNorm` normalizes each token row $r \in [0, R - 1]$:

$$\mathrm{RMS}(\mathbf{x}_r) = \sqrt{\frac{1}{C} \sum_{c=0}^{C-1} x_{r,c}^2 + \epsilon}, \qquad y_{r,c} = \frac{x_{r,c}}{\mathrm{RMS}(\mathbf{x}_r)} \cdot \gamma_c$$

- **Workgroup Geometry**: Dispatched with 1 workgroup per row (`dispatchWorkgroups(rows, 1, 1)`) and `@workgroup_size(256, 1, 1)`.
- **Two-Stage Reduction**:
  1. **Strided Thread Accumulation**: Each invocation `tid` in $[0, 255]$ accumulates $x_{r, \mathrm{tid} + 256k}^2$ in registers and writes its partial sum to `var<workgroup> shared_sq_sum: array<f32, 256>`.
  2. **Uniform Shared-Memory Tree Reduction**: An unrolled 8-step butterfly reduction (`stride = 128, 64, 32, 16, 8, 4, 2, 1`) reduces `shared_sq_sum[0]` with all `workgroupBarrier()` calls placed strictly in uniform control flow to satisfy the WGSL uniformity verifier.

### 2. Rotary Position Embeddings — `RoPE` (`shaders/rope.wgsl`)

Given query/key tensor $X \in \mathbb{R}^{S \times H \times D}$ (with even head dimension $D$), position offset $p_0$, and frequency base $\theta = 10000.0$, each feature pair $(2p, 2p+1)$ for $p \in [0, D/2 - 1]$ at sequence index $s \in [0, S - 1]$ is rotated by angle $\phi_{s,p} = (s + p_0)\,\theta^{-2p / D}$:

$$
\begin{bmatrix}
y_{s,h,2p} \\
y_{s,h,2p+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos \phi_{s,p} & -\sin \phi_{s,p} \\
\sin \phi_{s,p} & \cos \phi_{s,p}
\end{bmatrix}
\begin{bmatrix}
x_{s,h,2p} \\
x_{s,h,2p+1}
\end{bmatrix}
$$

- **Orthogonal Invariant**: Because the $2 \times 2$ rotation matrix is orthogonal ($\det R = \cos^2\phi + \sin^2\phi = 1$), the Euclidean L2 norm of every feature pair is strictly invariant: $\sqrt{y_{2p}^2 + y_{2p+1}^2} = \sqrt{x_{2p}^2 + x_{2p+1}^2}$.

### 3. Group-Quantized `Q4_0` Matrix-Vector Multiply (`shaders/q4_gemv.wgsl`)

For a weight matrix $W \in \mathbb{R}^{R \times C}$, each row is partitioned into blocks of $B = 32$ weights ($\lceil C / 32 \rceil$ blocks per row). Within block $b$:

1. **Symmetric Block Scale**:

$$
d_b = \frac{\max_{i \in [0, 31]} |w_{b,i}|}{7.0}, \qquad q_{b,i} = \mathrm{clip}\left(\mathrm{nearbyint}\left(\frac{w_{b,i}}{d_b}\right) + 8,\; 0,\; 15\right) \in \{0, \dots, 15\}
$$

2. **32-Bit `u32` Nibble Packing**: WebGPU storage buffers do not natively support 8-bit integer scalar types (`u8`) in standard WGSL without extensions; therefore, `webgpu-tensor-compute` packs **8 unsigned 4-bit nibbles** into each 32-bit `u32` word (`4 x u32` words = 16 bytes = 32 weights per block):

$$
\mathrm{word}_w = \sum_{k=0}^{7} q_{8w + k} \cdot 2^{4k}, \qquad \hat{w}_{8w+k} = \left(\mathrm{f32}\left((\mathrm{word}_w \gg 4k) \bmod 16\right) - 8.0\right) \cdot d_b
$$

- **Memory Compression**: Each block of 32 `FP32` weights (`128 bytes`) is compressed to `1 x f32` scale (`4 bytes`) + `4 x u32` packed words (`16 bytes`) = **`20 bytes` per 32 weights** (`4.5 bits/weight`, **6.4x compression** vs. `FP32`).

---

## WebGPU Workgroup & Memory Alignment Specification

| Kernel Shader | Workgroup Size | Dispatch Geometry | Shared Memory | WebGPU Buffer & Alignment Layout |
| :--- | :--- | :--- | :--- | :--- |
| **[`shaders/rmsnorm.wgsl`](shaders/rmsnorm.wgsl)** | `@workgroup_size(256, 1, 1)` | `(rows, 1, 1)` | `array<f32, 256>` (`1024 B`) | `RmsNormParams` padded to `16 B`; dynamic uniform/storage bindings aligned to **`256 B`** (`minUniformBufferOffsetAlignment`). |
| **[`shaders/rope.wgsl`](shaders/rope.wgsl)** | `@workgroup_size(64, 1, 1)` | `(ceil(total_pairs / 64), 1, 1)` | `0 B` (Register-only) | `RopeParams` (`16 B`); `cos_table` & `sin_table` of shape `[seq_len, head_dim / 2]`; buffer copies aligned to **`4 B`**. |
| **[`shaders/q4_gemv.wgsl`](shaders/q4_gemv.wgsl)** | `@workgroup_size(64, 1, 1)` | `(ceil(rows / 64), 1, 1)` | `0 B` (Register-only) | `Q4GemvParams` (`16 B`); `block_scales` (`f32[rows * blocks_per_row]`); `packed_q4_words` (`u32[rows * blocks_per_row * 4]`). |

---

## Quick Start & Build Instructions

### 1. Build with CMake (Linux, macOS, Windows MinGW / MSVC)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 2. Run the Automated Unit Test Suite (`test_wgpu_ops`)

```bash
ctest --test-dir build -C Release --output-on-failure
```

```text
=== Running webgpu-tensor-compute Unit Test Suite ===
[PASS] align_up(1, 256) == 256
[PASS] align_up(256, 256) == 256
[PASS] align_up(257, 256) == 512
[PASS] compute_buffer_alignment 256-byte uniform/storage offset calculation
[PASS] WGSL shader validation: rmsnorm.wgsl
[PASS] WGSL shader validation: rope.wgsl
[PASS] WGSL shader validation: q4_gemv.wgsl
[PASS] RMSNorm WGSL f32 tree reduction vs FP64 golden (17x517)
[PASS] RMSNorm zero-input epsilon stability guard
[PASS] RoPE WGSL f32 vs FP64 golden with non-zero pos_offset (19x6x64)
[PASS] RoPE orthogonal pair rotation L2-norm preservation
[PASS] Q4_0 partial block count for cols=100 is 4
[PASS] Q4_0 WGSL nibble-unpacking GEMV vs FP64 dequantized GEMV (23x100)
[PASS] Q4_0 GEMV cosine similarity vs uncompressed FP32 (> 0.99)
[PASS] Per-channel INT8 GEMV cosine similarity vs uncompressed FP32 (> 0.9999)
[PASS] Q4_0 zero-scale matrix dequantizes and multiplies to exact 0.0f
[PASS] RoPE rejects odd head_dim with std::invalid_argument
[PASS] RMSNorm rejects non-positive epsilon with std::invalid_argument
[PASS] Q4_0 GEMV rejects malformed storage buffers with std::invalid_argument
[PASS] Live GPU WebGPU rmsnorm.wgsl dispatch parity vs FP64 (NVIDIA GeForce GTX 1660 SUPER)
[PASS] Live GPU WebGPU rope.wgsl dispatch parity vs FP64 (NVIDIA GeForce GTX 1660 SUPER)
[PASS] Live GPU WebGPU q4_gemv.wgsl dispatch parity vs FP64 (NVIDIA GeForce GTX 1660 SUPER)
=== ALL TESTS PASSED (0 failures) ===
```

### 3. Run the CLI Benchmark & Live GPU Parity Harness (`wgpu_compute`)

```bash
./build/wgpu_compute --op all --rows 64 --cols 1024 --seq-len 64 --num-heads 8 --head-dim 64 --iterations 50
```

```text
====================================================================================
  webgpu-tensor-compute | Cross-Platform WGSL LLM Kernel & GPU/CPU Parity Harness
====================================================================================
  Execution Backend : Live Hardware WebGPU Compute Pipeline (wgpu-native vs. FP64 Reference)
  Hardware Adapter  : NVIDIA GeForce GTX 1660 SUPER (Vulkan, VendorID=0x10de, DeviceID=0x21c4)
  Uniform Alignment : 256 bytes | Q4_0 Block Size: 32
------------------------------------------------------------------------------------
  WGSL Shader Source Verification:
    - rmsnorm.wgsl     | Status: VALID WGSL | Size: 3653  B
    - rope.wgsl        | Status: VALID WGSL | Size: 2396  B
    - q4_gemv.wgsl     | Status: VALID WGSL | Size: 2755  B
------------------------------------------------------------------------------------
Kernel Operation            Shape         GPU (ms)    MaxAbsErr     RMSE          CosineSim   Status
------------------------------------------------------------------------------------
RMSNorm (GPU WGSL vs f64)   64x1024       0.0903      3.576e-07     3.721e-08     1.000000    PASS
RoPE (GPU WGSL vs f64)      64x8x64       0.0815      1.192e-07     2.356e-08     1.000000    PASS
Q4_0 GEMV (GPU WGSL vs f64) 64x1024       0.1158      2.861e-06     1.194e-06     1.000000    PASS
====================================================================================
```

---

## Programmatic C++17 API Usage

```cpp
#include "wgpu_compute/ops.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1. Compute 256-byte WebGPU uniform/storage alignment metadata
    auto align_info = wgpu_compute::compute_buffer_alignment(1024, sizeof(float));
    std::cout << "Raw bytes: " << align_info.raw_byte_size
              << " -> 256B Aligned: " << align_info.binding_aligned_bytes << "\n";

    // 2. Quantize a [64 x 1024] FP32 weight matrix into Q4_0 (block_size = 32)
    std::vector<float> weights_f32(64 * 1024, 0.25f);
    std::vector<float> input_x(1024, 1.0f);
    auto q4_mat = wgpu_compute::quantize_f32_to_q4_0(weights_f32, 64, 1024);

    // 3. Run WGSL-faithful Q4_0 GEMV and verify against FP64 dequantized reference
    std::vector<float> y_wgsl, y_golden;
    wgpu_compute::q4_gemv_wgsl_f32(q4_mat, input_x, y_wgsl);
    wgpu_compute::q4_gemv_golden_f64(q4_mat, input_x, y_golden);

    auto parity = wgpu_compute::compute_parity_metrics(y_wgsl, y_golden, 1e-4f);
    std::cout << "Q4_0 GEMV MaxAbsErr: " << parity.max_abs_err
              << " | CosineSim: " << parity.cosine_similarity << "\n";
    return 0;
}
```

---

## Project Layout

```text
webgpu-tensor-compute/
├── .github/workflows/
│   └── ci.yml                      # Cross-platform GitHub Actions CI (Ubuntu, Windows, macOS)
├── include/wgpu_compute/
│   └── ops.hpp                     # Public C++17 headers: alignment, WGSL validation, kernels, metrics
├── shaders/
│   ├── rmsnorm.wgsl                # 256-thread workgroup shared-memory reduction RMSNorm shader
│   ├── rope.wgsl                   # Rotary Position Embedding pairwise rotation shader
│   └── q4_gemv.wgsl                # Group-quantized Q4_0 (block_size=32) nibble-unpacking GEMV shader
├── src/
│   ├── ops.cpp                     # Workgroup-faithful f32 simulator & FP64 golden reference kernels
│   └── main.cpp                    # CLI verification & benchmark runner (--op rmsnorm|rope|q4_gemv|all)
├── tests/
│   └── test_wgpu_ops.cpp           # 19-check automated unit suite (edge shapes, invariants, error guards)
├── .gitignore
├── CMakeLists.txt
└── README.md
```
