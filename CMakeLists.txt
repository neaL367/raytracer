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
