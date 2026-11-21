cmake_minimum_required(VERSION 3.16)
project(webgpu_tensor_compute VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O3)
endif()

add_library(wgpu_ops STATIC
    src/ops.cpp
)
target_include_directories(wgpu_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(wgpu_ops PUBLIC
    WGPU_COMPUTE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

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
