cmake_minimum_required(VERSION 3.16)
project(webgpu_tensor_compute VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_WGPU_NATIVE "Enable live hardware WebGPU execution via wgpu-native when present" ON)

if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
endif()

set(WGPU_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/wgpu-native")
set(WGPU_HAS_NATIVE FALSE)

if(USE_WGPU_NATIVE AND EXISTS "${WGPU_NATIVE_ROOT}/include/webgpu/webgpu.h")
    find_library(WGPU_NATIVE_LIB
        NAMES wgpu_native libwgpu_native.dll.a
        PATHS "${WGPU_NATIVE_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    if(WGPU_NATIVE_LIB)
        set(WGPU_HAS_NATIVE TRUE)
        message(STATUS "[wgpu_compute] Found wgpu-native: ${WGPU_NATIVE_LIB} (Live hardware GPU execution enabled)")
    endif()
endif()

if(NOT WGPU_HAS_NATIVE)
    message(STATUS "[wgpu_compute] Building with deterministic CPU WGSL simulator fallback (wgpu-native not linked)")
endif()

add_library(wgpu_ops STATIC
    src/ops.cpp
    src/gpu_runner.cpp
)
target_include_directories(wgpu_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(wgpu_ops PUBLIC
    WGPU_COMPUTE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

if(WGPU_HAS_NATIVE)
    target_include_directories(wgpu_ops PUBLIC "${WGPU_NATIVE_ROOT}/include")
    target_link_libraries(wgpu_ops PUBLIC "${WGPU_NATIVE_LIB}")
    target_compile_definitions(wgpu_ops PUBLIC WGPU_ENABLE_NATIVE_BACKEND=1)
else()
    target_compile_definitions(wgpu_ops PUBLIC WGPU_ENABLE_NATIVE_BACKEND=0)
endif()

add_executable(wgpu_compute
    src/main.cpp
)
target_link_libraries(wgpu_compute PRIVATE wgpu_ops)

enable_testing()
add_executable(test_wgpu_ops
    tests/test_wgpu_ops.cpp
)
target_link_libraries(test_wgpu_ops PRIVATE wgpu_ops)
add_test(NAME WgpuOpsUnitSuite COMMAND test_wgpu_ops)

if(WGPU_HAS_NATIVE AND WIN32 AND EXISTS "${WGPU_NATIVE_ROOT}/lib/wgpu_native.dll")
    add_custom_command(TARGET wgpu_compute POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WGPU_NATIVE_ROOT}/lib/wgpu_native.dll"
            "$<TARGET_FILE_DIR:wgpu_compute>/wgpu_native.dll"
    )
    add_custom_command(TARGET test_wgpu_ops POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WGPU_NATIVE_ROOT}/lib/wgpu_native.dll"
            "$<TARGET_FILE_DIR:test_wgpu_ops>/wgpu_native.dll"
    )
endif()
