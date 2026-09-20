cmake_minimum_required(VERSION 3.25)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/app/main.cpp)
target_include_directories(raytracer PRIVATE src)

if(MSVC)
    target_compile_options(raytracer PRIVATE /W4)
else()
    target_compile_options(raytracer PRIVATE -Wall -Wextra)
endif()

enable_testing()
add_executable(raytracer_tests tests/test_all.cpp)
target_include_directories(raytracer_tests PRIVATE src)
add_test(NAME unit_tests COMMAND raytracer_tests)

# GPU backend: headless Vulkan compute. CPU binary untouched.
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(GPU_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${GPU_SHADER_DIR})
foreach(shader grad normal path)
    add_custom_command(
        OUTPUT ${GPU_SHADER_DIR}/${shader}.spv
        COMMAND ${GLSLC_EXECUTABLE} -O -I${CMAKE_SOURCE_DIR}/src/gpu/shaders
                ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
                -o ${GPU_SHADER_DIR}/${shader}.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
                ${CMAKE_SOURCE_DIR}/src/gpu/shaders/common.glsl
        COMMENT "compiling ${shader}.comp")
    list(APPEND GPU_SHADERS_COMPILED ${GPU_SHADER_DIR}/${shader}.spv)
endforeach()
add_custom_target(gpu_shaders DEPENDS ${GPU_SHADERS_COMPILED})

add_executable(rt_gpu src/gpu/main.cpp)
target_include_directories(rt_gpu PRIVATE src)
target_link_libraries(rt_gpu PRIVATE Vulkan::Vulkan)
target_compile_definitions(rt_gpu PRIVATE SHADER_DIR="${GPU_SHADER_DIR}")
add_dependencies(rt_gpu gpu_shaders)
if(MSVC)
    target_compile_options(rt_gpu PRIVATE /W4)
endif()
