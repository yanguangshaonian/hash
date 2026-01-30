cmake_minimum_required(VERSION 3.22.1)
project(hash)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

include_directories(lib)
add_compile_options(-mwaitpkg)

add_executable(main src/main.cpp)
add_executable(l3_test src/l3_test.cpp)