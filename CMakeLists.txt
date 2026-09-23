cmake_minimum_required(VERSION 3.25)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/app/main.cpp src/io/stb_image.cpp)
target_include_directories(raytracer PRIVATE src external)
if(MSVC)
    target_compile_options(raytracer PRIVATE /W4)
else()
    target_compile_options(raytracer PRIVATE -Wall -Wextra)
endif()

# Mesh/texture assets next to the binary (main also tries CWD root).
add_custom_command(TARGET raytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:raytracer>/assets
)

enable_testing()
add_executable(raytracer_tests tests/test_all.cpp tests/test_core.cpp tests/test_geometry.cpp
               tests/test_shading.cpp tests/test_scene.cpp src/io/stb_image.cpp)
target_include_directories(raytracer_tests PRIVATE src tests external)
add_test(NAME unit_tests COMMAND raytracer_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME parity COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/parity.py --fast WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# GPU backend: headless Vulkan compute. CPU binary untouched.
find_package(Vulkan REQUIRED)
# SDK layouts differ: Windows uses Bin/, Linux/macOS use bin/.
find_program(GLSLC_EXECUTABLE glslc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)

set(GPU_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${GPU_SHADER_DIR})

# Generate push_block.glsl directly from push_layout.inc (inlined from cmake/generate_push_block.cmake)
function(generate_push_block source_inc out_glsl)
    file(STRINGS "${source_inc}" PUSH_ENTRIES)
    set(GLSL_PC_MEMBERS "")
    foreach(ENTRY ${PUSH_ENTRIES})
        string(REGEX MATCH "X\\(([a-zA-Z0-9_]+),[ \t]*([a-zA-Z0-9_]+),[ \t]*([0-9]+)\\)" MATCHED "${ENTRY}")
        if(MATCHED)
            set(TYPE "${CMAKE_MATCH_1}")
            set(NAME "${CMAKE_MATCH_2}")
            string(APPEND GLSL_PC_MEMBERS "    ${TYPE} ${NAME};\n")
        endif()
    endforeach()
    file(WRITE "${out_glsl}" "// Auto-generated from push_layout.inc. Do not edit directly.\nlayout(push_constant) uniform PC {\n${GLSL_PC_MEMBERS}} pc;\n")
endfunction()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/push_layout.inc)
generate_push_block(${CMAKE_SOURCE_DIR}/src/gpu/push_layout.inc ${GPU_SHADER_DIR}/push_block.glsl)

foreach(shader grad normal path denoise joint)
    add_custom_command(
        OUTPUT ${GPU_SHADER_DIR}/${shader}.spv
        COMMAND ${GLSLC_EXECUTABLE} -O -I${CMAKE_SOURCE_DIR}/src/gpu/shaders -I${GPU_SHADER_DIR}
                ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
                -o ${GPU_SHADER_DIR}/${shader}.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
                ${CMAKE_SOURCE_DIR}/src/gpu/shaders/common.glsl
                ${GPU_SHADER_DIR}/push_block.glsl
        COMMENT "compiling ${shader}.comp")
    list(APPEND GPU_SHADERS_COMPILED ${GPU_SHADER_DIR}/${shader}.spv)
endforeach()
add_custom_target(gpu_shaders DEPENDS ${GPU_SHADERS_COMPILED})

add_executable(rt_gpu src/gpu/main.cpp src/io/stb_image.cpp)
target_include_directories(rt_gpu PRIVATE src external)
target_link_libraries(rt_gpu PRIVATE Vulkan::Vulkan)
target_compile_definitions(rt_gpu PRIVATE SHADER_DIR="${GPU_SHADER_DIR}")
add_dependencies(rt_gpu gpu_shaders)
if(MSVC)
    target_compile_options(rt_gpu PRIVATE /W4)
endif()

# Preview viewer: SDL3 via FetchContent (static, no system install).
# Offline configure skips the target with a notice instead of failing.
option(RAYTRACER_VIEWER "Build the SDL3 rt_view preview window" ON)
if(RAYTRACER_VIEWER)
    include(FetchContent)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(SDL3 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
                         GIT_TAG release-3.4.16 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(SDL3)
    add_executable(rt_view src/view/main.cpp)
    target_include_directories(rt_view PRIVATE src)
    target_link_libraries(rt_view PRIVATE SDL3::SDL3)
    if(MSVC)
        target_compile_options(rt_view PRIVATE /W4)
    endif()
endif()
