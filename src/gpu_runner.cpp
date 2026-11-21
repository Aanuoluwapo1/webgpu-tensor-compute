#include "wgpu_compute/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND
#include "webgpu/webgpu.h"
#include "webgpu/wgpu.h"
#endif

namespace wgpu_compute {

#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND

namespace {

std::string backend_type_to_string(WGPUBackendType backend) {
    switch (backend) {
        case WGPUBackendType_D3D12: return "Direct3D 12";
        case WGPUBackendType_D3D11: return "Direct3D 11";
        case WGPUBackendType_Vulkan: return "Vulkan";
        case WGPUBackendType_Metal: return "Metal";
        case WGPUBackendType_OpenGL: return "OpenGL";
        case WGPUBackendType_OpenGLES: return "OpenGL ES";
        case WGPUBackendType_WebGPU: return "WebGPU";
        default: return "Native";
    }
}

struct NativeWgpuContext {
    bool initialized = false;
    bool available = false;
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    GpuAdapterStatus status{};
};

NativeWgpuContext& get_wgpu_context() {
    static NativeWgpuContext ctx;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    if (ctx.initialized) {
        return ctx;
    }
    ctx.initialized = true;

    ctx.instance = wgpuCreateInstance(nullptr);
    if (!ctx.instance) {
        return ctx;
    }

    WGPURequestAdapterOptions opts{};
    opts.featureLevel = WGPUFeatureLevel_Core;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPURequestAdapterCallbackInfo adapter_cb{};
    adapter_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    adapter_cb.callback = [](WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView, void* ud1, void*) {
        if (st == WGPURequestAdapterStatus_Success) {
            *static_cast<WGPUAdapter*>(ud1) = a;
        }
    };
    adapter_cb.userdata1 = &ctx.adapter;
    wgpuInstanceRequestAdapter(ctx.instance, &opts, adapter_cb);

    if (!ctx.adapter) {
        return ctx;
    }

    WGPUDeviceDescriptor dev_desc{};
    dev_desc.label = WGPUStringView{"wgpu_compute_device", WGPU_STRLEN};

    WGPURequestDeviceCallbackInfo dev_cb{};
    dev_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    dev_cb.callback = [](WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView, void* ud1, void*) {
        if (st == WGPURequestDeviceStatus_Success) {
            *static_cast<WGPUDevice*>(ud1) = d;
        }
    };
    dev_cb.userdata1 = &ctx.device;
    wgpuAdapterRequestDevice(ctx.adapter, &dev_desc, dev_cb);

    if (!ctx.device) {
        return ctx;
    }

    ctx.queue = wgpuDeviceGetQueue(ctx.device);
    if (!ctx.queue) {
        return ctx;
    }

    WGPUAdapterInfo info{};
    if (wgpuAdapterGetInfo(ctx.adapter, &info) == WGPUStatus_Success) {
        ctx.status.adapter_name = (info.device.data && info.device.length > 0)
            ? std::string(info.device.data, info.device.length)
            : "Unknown Hardware GPU";
        ctx.status.backend_name = backend_type_to_string(info.backendType);
        ctx.status.vendor_id = info.vendorID;
        ctx.status.device_id = info.deviceID;
        wgpuAdapterInfoFreeMembers(info);
    }
    ctx.status.available = true;
    ctx.available = true;
    return ctx;
}

std::string load_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open WGSL shader file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

WGPUShaderModule create_wgsl_module(WGPUDevice device, const std::string& wgsl_code, const char* label) {
    WGPUShaderSourceWGSL wgsl_src{};
    wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_src.code = WGPUStringView{wgsl_code.c_str(), wgsl_code.size()};

    WGPUShaderModuleDescriptor sm_desc{};
    sm_desc.nextInChain = &wgsl_src.chain;
    sm_desc.label = WGPUStringView{label, WGPU_STRLEN};

    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &sm_desc);
    if (!mod) {
        throw std::runtime_error(std::string("Failed to compile WGSL shader module: ") + label);
    }
    return mod;
}

WGPUComputePipeline create_compute_pipeline(WGPUDevice device, WGPUShaderModule mod, const char* label) {
    WGPUComputePipelineDescriptor cp_desc{};
    cp_desc.label = WGPUStringView{label, WGPU_STRLEN};
    cp_desc.layout = nullptr; // Auto layout
    cp_desc.compute.module = mod;
    cp_desc.compute.entryPoint = WGPUStringView{"main", WGPU_STRLEN};

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cp_desc);
    if (!pipeline) {
        throw std::runtime_error(std::string("Failed to create WebGPU compute pipeline: ") + label);
    }
    return pipeline;
}

WGPUBuffer create_gpu_buffer(WGPUDevice device, WGPUQueue queue, const void* init_data, std::size_t raw_bytes, WGPUBufferUsage usage, const char* label) {
    const std::size_t aligned_size = align_up(std::max<std::size_t>(raw_bytes, 4), 4);
    WGPUBufferDescriptor desc{};
    desc.label = WGPUStringView{label, WGPU_STRLEN};
    desc.size = aligned_size;
    desc.usage = usage;
    desc.mappedAtCreation = 0;

    WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
    if (!buf) {
        throw std::runtime_error(std::string("Failed to create WGPUBuffer: ") + label);
    }
    if (init_data && raw_bytes > 0) {
        if (raw_bytes == aligned_size) {
            wgpuQueueWriteBuffer(queue, buf, 0, init_data, aligned_size);
        } else {
            std::vector<std::uint8_t> padded(aligned_size, 0);
            std::memcpy(padded.data(), init_data, raw_bytes);
            wgpuQueueWriteBuffer(queue, buf, 0, padded.data(), aligned_size);
        }
    }
    return buf;
}

void readback_gpu_buffer(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUBuffer gpu_buf,
    void* dst_host,
    std::size_t raw_bytes
) {
    const std::size_t aligned_size = align_up(std::max<std::size_t>(raw_bytes, 4), 4);
    WGPUBuffer stage = create_gpu_buffer(
        device, queue, nullptr, aligned_size,
        WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "readback_staging");

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, gpu_buf, 0, stage, 0, aligned_size);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    WGPUSubmissionIndex sub_idx = wgpuQueueSubmitForIndex(queue, 1, &cmd);
    wgpuDevicePoll(device, 1, &sub_idx);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    bool mapped_ok = false;
    WGPUBufferMapCallbackInfo map_cb{};
    map_cb.mode = WGPUCallbackMode_AllowSpontaneous;
    map_cb.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* ud1, void*) {
        if (st == WGPUMapAsyncStatus_Success) {
            *static_cast<bool*>(ud1) = true;
        }
    };
    map_cb.userdata1 = &mapped_ok;

    wgpuBufferMapAsync(stage, WGPUMapMode_Read, 0, aligned_size, map_cb);
    wgpuDevicePoll(device, 1, nullptr);

    if (!mapped_ok) {
        wgpuBufferRelease(stage);
        throw std::runtime_error("wgpuBufferMapAsync failed during GPU readback");
    }

    const void* mapped_ptr = wgpuBufferGetConstMappedRange(stage, 0, aligned_size);
    if (!mapped_ptr) {
        wgpuBufferUnmap(stage);
        wgpuBufferRelease(stage);
        throw std::runtime_error("wgpuBufferGetConstMappedRange returned nullptr");
    }
    std::memcpy(dst_host, mapped_ptr, raw_bytes);
    wgpuBufferUnmap(stage);
    wgpuBufferRelease(stage);
}

} // namespace

#endif // WGPU_ENABLE_NATIVE_BACKEND

GpuAdapterStatus query_gpu_adapter_status() {
#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND
    return get_wgpu_context().status;
#else
    return GpuAdapterStatus{};
#endif
}

double dispatch_rmsnorm_gpu(
    const std::string& shader_path,
    const std::vector<float>& input_x,
    const std::vector<float>& weight_gamma,
    std::vector<float>& output_y,
    std::size_t rows,
    std::size_t cols,
    float eps,
    int iterations,
    bool* out_used_gpu
) {
#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND
    auto& ctx = get_wgpu_context();
    if (ctx.available) {
        if (out_used_gpu) *out_used_gpu = true;
        output_y.resize(rows * cols, 0.0f);

        struct RmsNormParams {
            std::uint32_t rows;
            std::uint32_t cols;
            float eps;
            std::uint32_t pad0;
        } params{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols), eps, 0u};

        if (rows > 65535) { throw std::invalid_argument("rmsnorm dispatch: rows exceeds maxComputeWorkgroupsPerDimension (65535)"); }
        const std::string code = load_text_file(shader_path);
        WGPUShaderModule mod = create_wgsl_module(ctx.device, code, "rmsnorm_wgsl");
        WGPUComputePipeline pipeline = create_compute_pipeline(ctx.device, mod, "rmsnorm_pipeline");
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        WGPUBuffer b_params = create_gpu_buffer(
            ctx.device, ctx.queue, &params, sizeof(params),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "rmsnorm_params");
        WGPUBuffer b_x = create_gpu_buffer(
            ctx.device, ctx.queue, input_x.data(), input_x.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "rmsnorm_x");
        WGPUBuffer b_gamma = create_gpu_buffer(
            ctx.device, ctx.queue, weight_gamma.data(), weight_gamma.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "rmsnorm_gamma");
        WGPUBuffer b_y = create_gpu_buffer(
            ctx.device, ctx.queue, nullptr, output_y.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "rmsnorm_y");

        WGPUBindGroupEntry entries[4]{};
        entries[0].binding = 0; entries[0].buffer = b_params; entries[0].size = sizeof(params);
        entries[1].binding = 1; entries[1].buffer = b_x;      entries[1].size = wgpuBufferGetSize(b_x);
        entries[2].binding = 2; entries[2].buffer = b_gamma;  entries[2].size = wgpuBufferGetSize(b_gamma);
        entries[3].binding = 3; entries[3].buffer = b_y;      entries[3].size = wgpuBufferGetSize(b_y);

        WGPUBindGroupDescriptor bg_desc{};
        bg_desc.layout = bgl;
        bg_desc.entryCount = 4;
        bg_desc.entries = entries;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx.device, &bg_desc);

        auto run_pass = [&]() {
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, static_cast<std::uint32_t>(rows), 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            WGPUSubmissionIndex idx = wgpuQueueSubmitForIndex(ctx.queue, 1, &cmd);
            wgpuDevicePoll(ctx.device, 1, &idx);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        };

        // Warmup dispatch on physical GPU
        run_pass();

        const int run_iters = std::max(1, iterations);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < run_iters; ++i) {
            run_pass();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;

        readback_gpu_buffer(ctx.device, ctx.queue, b_y, output_y.data(), output_y.size() * sizeof(float));

        wgpuBindGroupRelease(bg);
        wgpuBindGroupLayoutRelease(bgl);
        wgpuBufferRelease(b_params);
        wgpuBufferRelease(b_x);
        wgpuBufferRelease(b_gamma);
        wgpuBufferRelease(b_y);
        wgpuComputePipelineRelease(pipeline);
        wgpuShaderModuleRelease(mod);

        return avg_ms;
    }
#endif
    (void)shader_path;
    if (out_used_gpu) *out_used_gpu = false;
    const int run_iters = std::max(1, iterations);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < run_iters; ++i) {
        rmsnorm_wgsl_f32(input_x, weight_gamma, output_y, rows, cols, eps);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;
}

double dispatch_rope_gpu(
    const std::string& shader_path,
    const std::vector<float>& input_x,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    std::vector<float>& output_y,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t head_dim,
    int iterations,
    bool* out_used_gpu
) {
#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND
    auto& ctx = get_wgpu_context();
    if (ctx.available) {
        if (out_used_gpu) *out_used_gpu = true;
        const std::size_t total = seq_len * num_heads * head_dim;
        output_y.resize(total, 0.0f);

        struct RopeParams {
            std::uint32_t seq_len;
            std::uint32_t num_heads;
            std::uint32_t head_dim;
            std::uint32_t pos_offset;
        } params{
            static_cast<std::uint32_t>(seq_len),
            static_cast<std::uint32_t>(num_heads),
            static_cast<std::uint32_t>(head_dim),
            0u
        };

        const std::string code = load_text_file(shader_path);
        WGPUShaderModule mod = create_wgsl_module(ctx.device, code, "rope_wgsl");
        WGPUComputePipeline pipeline = create_compute_pipeline(ctx.device, mod, "rope_pipeline");
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        WGPUBuffer b_params = create_gpu_buffer(
            ctx.device, ctx.queue, &params, sizeof(params),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "rope_params");
        WGPUBuffer b_x = create_gpu_buffer(
            ctx.device, ctx.queue, input_x.data(), input_x.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "rope_x");
        WGPUBuffer b_cos = create_gpu_buffer(
            ctx.device, ctx.queue, cos_table.data(), cos_table.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "rope_cos");
        WGPUBuffer b_sin = create_gpu_buffer(
            ctx.device, ctx.queue, sin_table.data(), sin_table.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "rope_sin");
        WGPUBuffer b_y = create_gpu_buffer(
            ctx.device, ctx.queue, nullptr, output_y.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "rope_y");

        WGPUBindGroupEntry entries[5]{};
        entries[0].binding = 0; entries[0].buffer = b_params; entries[0].size = sizeof(params);
        entries[1].binding = 1; entries[1].buffer = b_x;      entries[1].size = wgpuBufferGetSize(b_x);
        entries[2].binding = 2; entries[2].buffer = b_cos;    entries[2].size = wgpuBufferGetSize(b_cos);
        entries[3].binding = 3; entries[3].buffer = b_sin;    entries[3].size = wgpuBufferGetSize(b_sin);
        entries[4].binding = 4; entries[4].buffer = b_y;      entries[4].size = wgpuBufferGetSize(b_y);

        WGPUBindGroupDescriptor bg_desc{};
        bg_desc.layout = bgl;
        bg_desc.entryCount = 5;
        bg_desc.entries = entries;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx.device, &bg_desc);

        const std::size_t total_pairs = seq_len * num_heads * (head_dim / 2);
        const std::uint32_t groups_x = static_cast<std::uint32_t>((total_pairs + 63) / 64);

        auto run_pass = [&]() {
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, groups_x, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            WGPUSubmissionIndex idx = wgpuQueueSubmitForIndex(ctx.queue, 1, &cmd);
            wgpuDevicePoll(ctx.device, 1, &idx);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        };

        run_pass();

        const int run_iters = std::max(1, iterations);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < run_iters; ++i) {
            run_pass();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;

        readback_gpu_buffer(ctx.device, ctx.queue, b_y, output_y.data(), output_y.size() * sizeof(float));

        wgpuBindGroupRelease(bg);
        wgpuBindGroupLayoutRelease(bgl);
        wgpuBufferRelease(b_params);
        wgpuBufferRelease(b_x);
        wgpuBufferRelease(b_cos);
        wgpuBufferRelease(b_sin);
        wgpuBufferRelease(b_y);
        wgpuComputePipelineRelease(pipeline);
        wgpuShaderModuleRelease(mod);

        return avg_ms;
    }
#endif
    (void)shader_path;
    if (out_used_gpu) *out_used_gpu = false;
    const int run_iters = std::max(1, iterations);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < run_iters; ++i) {
        rope_wgsl_f32(input_x, cos_table, sin_table, output_y, seq_len, num_heads, head_dim);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;
}

double dispatch_q4_gemv_gpu(
    const std::string& shader_path,
    const Q4MatrixStorage& q4_mat,
    const std::vector<float>& input_x,
    std::vector<float>& output_y,
    int iterations,
    bool* out_used_gpu
) {
#if defined(WGPU_ENABLE_NATIVE_BACKEND) && WGPU_ENABLE_NATIVE_BACKEND
    auto& ctx = get_wgpu_context();
    if (ctx.available) {
        if (out_used_gpu) *out_used_gpu = true;
        output_y.resize(q4_mat.rows, 0.0f);

        struct Q4GemvParams {
            std::uint32_t rows;
            std::uint32_t cols;
            std::uint32_t blocks_per_row;
            std::uint32_t pad0;
        } params{
            static_cast<std::uint32_t>(q4_mat.rows),
            static_cast<std::uint32_t>(q4_mat.cols),
            static_cast<std::uint32_t>(q4_mat.blocks_per_row),
            0u
        };

        const std::string code = load_text_file(shader_path);
        WGPUShaderModule mod = create_wgsl_module(ctx.device, code, "q4_gemv_wgsl");
        WGPUComputePipeline pipeline = create_compute_pipeline(ctx.device, mod, "q4_gemv_pipeline");
        WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

        WGPUBuffer b_params = create_gpu_buffer(
            ctx.device, ctx.queue, &params, sizeof(params),
            WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "q4_params");
        WGPUBuffer b_scales = create_gpu_buffer(
            ctx.device, ctx.queue, q4_mat.block_scales.data(), q4_mat.block_scales.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "q4_scales");
        WGPUBuffer b_words = create_gpu_buffer(
            ctx.device, ctx.queue, q4_mat.packed_words.data(), q4_mat.packed_words.size() * sizeof(std::uint32_t),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "q4_words");
        WGPUBuffer b_x = create_gpu_buffer(
            ctx.device, ctx.queue, input_x.data(), input_x.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "q4_x");
        WGPUBuffer b_y = create_gpu_buffer(
            ctx.device, ctx.queue, nullptr, output_y.size() * sizeof(float),
            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "q4_y");

        WGPUBindGroupEntry entries[5]{};
        entries[0].binding = 0; entries[0].buffer = b_params; entries[0].size = sizeof(params);
        entries[1].binding = 1; entries[1].buffer = b_scales; entries[1].size = wgpuBufferGetSize(b_scales);
        entries[2].binding = 2; entries[2].buffer = b_words;  entries[2].size = wgpuBufferGetSize(b_words);
        entries[3].binding = 3; entries[3].buffer = b_x;      entries[3].size = wgpuBufferGetSize(b_x);
        entries[4].binding = 4; entries[4].buffer = b_y;      entries[4].size = wgpuBufferGetSize(b_y);

        WGPUBindGroupDescriptor bg_desc{};
        bg_desc.layout = bgl;
        bg_desc.entryCount = 5;
        bg_desc.entries = entries;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx.device, &bg_desc);

        const std::uint32_t groups_x = static_cast<std::uint32_t>((q4_mat.rows + 63) / 64);

        auto run_pass = [&]() {
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, groups_x, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            WGPUSubmissionIndex idx = wgpuQueueSubmitForIndex(ctx.queue, 1, &cmd);
            wgpuDevicePoll(ctx.device, 1, &idx);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        };

        run_pass();

        const int run_iters = std::max(1, iterations);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < run_iters; ++i) {
            run_pass();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;

        readback_gpu_buffer(ctx.device, ctx.queue, b_y, output_y.data(), output_y.size() * sizeof(float));

        wgpuBindGroupRelease(bg);
        wgpuBindGroupLayoutRelease(bgl);
        wgpuBufferRelease(b_params);
        wgpuBufferRelease(b_scales);
        wgpuBufferRelease(b_words);
        wgpuBufferRelease(b_x);
        wgpuBufferRelease(b_y);
        wgpuComputePipelineRelease(pipeline);
        wgpuShaderModuleRelease(mod);

        return avg_ms;
    }
#endif
    (void)shader_path;
    if (out_used_gpu) *out_used_gpu = false;
    const int run_iters = std::max(1, iterations);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < run_iters; ++i) {
        q4_gemv_wgsl_f32(q4_mat, input_x, output_y);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / run_iters;
}

} // namespace wgpu_compute
