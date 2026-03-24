cmake_minimum_required(VERSION 3.24)
project(apatheia LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


option(APATHEIA_BUILD_TESTS      "Build unit tests"        OFF)
option(APATHEIA_BUILD_BENCHMARKS "Build benchmarks"        OFF)
option(APATHEIA_BUILD_PYTHON     "Build pybind11 bindings" OFF)
option(APATHEIA_ENABLE_ASAN      "Enable AddressSanitizer" OFF)


set(APATHEIA_CXX_FLAGS
  -Wall -Wextra -Wpedantic
  -Wno-unused-parameter
)

set(APATHEIA_CUDA_FLAGS
  --use_fast_math
  --extended-lambda
  --expt-relaxed-constexpr
  -Xcompiler=-Wall
  -lineinfo
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND APATHEIA_CUDA_FLAGS -G -g)
else()
  list(APPEND APATHEIA_CUDA_FLAGS -O3)
endif()

if(APATHEIA_ENABLE_ASAN)
  list(APPEND APATHEIA_CXX_FLAGS -fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()



add_library(apatheia_core STATIC
  # src/core/world.cpp
  # src/core/sim_loop.cpp
  # src/core/constraint_solver.cpp
  # src/core/grad_engine.cpp
  # src/gpu/body.cu
  # src/gpu/collision.cu
  # src/gpu/contact.cu
  # src/gpu/grad.cu
)

target_include_directories(apatheia_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(apatheia_core
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${APATHEIA_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${APATHEIA_CUDA_FLAGS}>
)


set_target_properties(apatheia_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON
)



find_package(CUDAToolkit REQUIRED)
target_link_libraries(apatheia_core
  PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)


if(APATHEIA_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(apatheia_tests
    tests/test_math.cpp
    tests/test_quaternion.cpp
    tests/test_sim_loop.cpp
    tests/test_grad.cu
  )

  target_link_libraries(apatheia_tests
    PRIVATE
      apatheia_core
      GTest::gtest_main
  )

  target_compile_options(apatheia_tests
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${APATHEIA_CXX_FLAGS}>
      $<$<COMPILE_LANGUAGE:CUDA>:${APATHEIA_CUDA_FLAGS}>
  )

  set_target_properties(apatheia_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  include(GoogleTest)
  gtest_discover_tests(apatheia_tests)
endif()



if(APATHEIA_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(apatheia_benchmarks
    benchmarks/bench_step.cu
    benchmarks/bench_collision.cu
    benchmarks/bench_grad.cu
  )

  target_link_libraries(apatheia_benchmarks
    PRIVATE
      apatheia_core
      benchmark::benchmark_main
  )

  target_compile_options(apatheia_benchmarks
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${APATHEIA_CUDA_FLAGS}>
  )

  set_target_properties(apatheia_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()


if(APATHEIA_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(apatheia_py
    bindings/apatheia.cpp
  )

  target_link_libraries(apatheia_py
    PRIVATE
      apatheia_core
  )

  set_target_properties(apatheia_py PROPERTIES
    OUTPUT_NAME "apatheia"
  )
endif()


install(TARGETS apatheia_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)


message(STATUS "")
message(STATUS "APATHEIA build configuration")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs      : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Tests           : ${APATHEIA_BUILD_TESTS}")
message(STATUS "  Benchmarks      : ${APATHEIA_BUILD_BENCHMARKS}")
message(STATUS "  Python bindings : ${APATHEIA_BUILD_PYTHON}")
message(STATUS "  ASan            : ${APATHEIA_ENABLE_ASAN}")
message(STATUS "")
