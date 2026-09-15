cmake_minimum_required(VERSION 3.25)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/main.cpp src/io/stb_image.cpp src/io/oidn_denoise.cpp)
target_include_directories(raytracer PRIVATE src external/stb)

# SIMD baseline: MSVC targets SSE2 unless told otherwise, which caps the
# auto-vectorizer at 128-bit. AVX2 (2013+) allows 256-bit vector code.
if(MSVC)
    target_compile_options(raytracer PRIVATE /arch:AVX2 /W4)
else()
    target_compile_options(raytracer PRIVATE -mavx2 -Wall -Wextra)
endif()

add_custom_command(TARGET raytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:raytracer>/assets
)

# Unit tests: dependency-free (see tests/test_all.cpp), run with ctest.
enable_testing()
add_executable(raytracer_tests tests/test_all.cpp src/io/stb_image.cpp)
target_include_directories(raytracer_tests PRIVATE src external/stb)
add_test(NAME unit_tests COMMAND raytracer_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Intel OIDN denoiser (for --oidn): downloaded once into the build tree, never
# committed. Disable with -DRAYTRACER_OIDN=OFF for fully offline configures.
option(RAYTRACER_OIDN "Download Intel OIDN for --oidn denoising" ON)
set(RAYTRACER_OIDN_AVAILABLE OFF)
if(RAYTRACER_OIDN)
    set(OIDN_VERSION 2.5.1)
    set(OIDN_URL https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.x64.windows.zip)
    set(OIDN_SHA256 f11f91bc072a5e3a564515724cb72ab8fcfbc445c84a84c197ff9b16cc01396f)
    set(OIDN_ROOT ${CMAKE_BINARY_DIR}/external/oidn/oidn-${OIDN_VERSION}.x64.windows)
    if(NOT EXISTS "${OIDN_ROOT}/include/OpenImageDenoise/oidn.h")
        message(STATUS "downloading OIDN ${OIDN_VERSION} (build tree only)")
        file(DOWNLOAD ${OIDN_URL} ${CMAKE_BINARY_DIR}/oidn.zip
             EXPECTED_HASH SHA256=${OIDN_SHA256} SHOW_PROGRESS
             STATUS OIDN_DL_STATUS)
        list(GET OIDN_DL_STATUS 0 OIDN_DL_CODE)
        if(OIDN_DL_CODE EQUAL 0)
            file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/oidn.zip DESTINATION ${CMAKE_BINARY_DIR}/external/oidn)
        else()
            message(WARNING "OIDN download failed (${OIDN_DL_STATUS}); --oidn will fall back to bilateral")
        endif()
    endif()
    if(EXISTS "${OIDN_ROOT}/include/OpenImageDenoise/oidn.h")
        set(RAYTRACER_OIDN_AVAILABLE ON)
    endif()
endif()
if(RAYTRACER_OIDN_AVAILABLE)
    target_include_directories(raytracer PRIVATE ${OIDN_ROOT}/include)
    target_link_libraries(raytracer PRIVATE ${OIDN_ROOT}/lib/OpenImageDenoise.lib)
    target_compile_definitions(raytracer PRIVATE RAYTRACER_OIDN=1)
    add_custom_command(TARGET raytracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${OIDN_ROOT}/bin $<TARGET_FILE_DIR:raytracer>)
endif()

# GPU scaffold: headless Vulkan compute (see src/gpu/main.cpp). CPU binary untouched.
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(GPU_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${GPU_SHADER_DIR})
foreach(shader fill normal path)
    add_custom_command(
        OUTPUT ${GPU_SHADER_DIR}/${shader}.spv
        COMMAND ${GLSLC_EXECUTABLE} -O ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
                -o ${GPU_SHADER_DIR}/${shader}.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/shaders/${shader}.comp
        COMMENT "compiling ${shader}.comp")
    list(APPEND GPU_SHADERS_COMPILED ${GPU_SHADER_DIR}/${shader}.spv)
endforeach()
add_custom_target(gpu_shaders DEPENDS ${GPU_SHADERS_COMPILED})

add_executable(rt_gpu src/gpu/main.cpp src/io/stb_image.cpp src/io/oidn_denoise.cpp)
target_include_directories(rt_gpu PRIVATE src external/stb)
target_link_libraries(rt_gpu PRIVATE Vulkan::Vulkan)
if(RAYTRACER_OIDN_AVAILABLE)
    target_include_directories(rt_gpu PRIVATE ${OIDN_ROOT}/include)
    target_link_libraries(rt_gpu PRIVATE ${OIDN_ROOT}/lib/OpenImageDenoise.lib)
    target_compile_definitions(rt_gpu PRIVATE RAYTRACER_OIDN=1)
    add_custom_command(TARGET rt_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${OIDN_ROOT}/bin $<TARGET_FILE_DIR:rt_gpu>)
endif()
add_dependencies(rt_gpu gpu_shaders)
add_custom_command(TARGET rt_gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${GPU_SHADER_DIR} $<TARGET_FILE_DIR:rt_gpu>/shaders)
if(MSVC)
    target_compile_options(rt_gpu PRIVATE /W4)
endif()