cmake_minimum_required(VERSION 3.25)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer src/main.cpp)

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
add_executable(raytracer_tests tests/test_all.cpp)
target_include_directories(raytracer_tests PRIVATE src)
add_test(NAME unit_tests COMMAND raytracer_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})