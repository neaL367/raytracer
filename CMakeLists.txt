cmake_minimum_required(VERSION 3.25)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/main.cpp)

# SIMD baseline: MSVC targets SSE2 unless told otherwise, which caps the
# auto-vectorizer at 128-bit. AVX2 (2013+) allows 256-bit vector code.
if(MSVC)
    target_compile_options(raytracer PRIVATE /arch:AVX2)
else()
    target_compile_options(raytracer PRIVATE -mavx2)
endif()

add_custom_command(TARGET raytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:raytracer>/assets
)