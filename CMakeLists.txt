cmake_minimum_required(VERSION 3.18)
project(monotonic_rnnt LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

include_directories(include)

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" ON)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_KERNEL "output alpha beta" OFF)
option(WITH_GPU "compile monotonic-rnnt with cuda." OFF)
option(WITH_OMP "compile monotonic-rnnt with openmp." ON)

if (USE_NAIVE_KERNEL)
    add_definitions(-DUSE_NAIVE_KERNEL)
endif ()
if (DEBUG_TIME)
    add_definitions(-DDEBUG_TIME)
endif ()
if (DEBUG_KERNEL)
    add_definitions(-DDEBUG_KERNEL)
endif ()

if (NOT WITH_OMP)
    add_definitions(-DRNNT_DISABLE_OMP)
endif ()
if (WITH_OMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif ()

set(CMAKE_CXX_STANDARD 17)

#always skip for linux
set(CMAKE_SKIP_RPATH TRUE)

if (WITH_GPU)
    message(STATUS "Building GPU version")
    message(STATUS "Use CUDA version ${CMAKE_CUDA_COMPILER_VERSION}.")

    add_definitions(-DRNNT_ENABLE_GPU)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60 61 62)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 70)
    endif ()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
    endif ()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 86)
    endif ()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND CMAKE_CUDA_ARCHITECTURES 89 90)
    endif ()
    message(STATUS "Compiling with compute capabilities ${CMAKE_CUDA_ARCHITECTURES}")
    add_executable(test_time_gpu tests/test_time.cu tests/random.cpp src/status.cpp)
    add_executable(test_gpu tests/test_gpu.cu tests/random.cpp src/status.cpp)
else ()
    message(STATUS "Building CPU version")
    add_executable(test_time tests/test_time.cpp tests/random.cpp src/status.cpp)
    add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp src/status.cpp)
endif ()

find_package(Python3 COMPONENTS Interpreter)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
        RESULT_VARIABLE _tensorflow_include_res
        OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        RESULT_VARIABLE _tensorflow_lib_res
        OUTPUT_VARIABLE Tensorflow_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_library(
        Tensorflow_FRAMEWORK
        NAMES tensorflow_framework
        HINTS ${Tensorflow_LIB_DIR} /usr/local/lib/tensorflow
        PATH_SUFFIXES lib REQUIRED)

add_library(tf_operation SHARED tensorflow_binding/monotonic_rnnt_op.cc src/status.cpp
        src/cpu_rnnt.cpp)
target_include_directories(tf_operation PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(tf_operation PUBLIC ${Tensorflow_INCLUDE_DIR})
target_link_libraries(tf_operation PUBLIC ${Tensorflow_FRAMEWORK})
add_custom_target(
        tf_test
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:tf_operation>:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_binding/test.py
        DEPENDS tf_operation
        COMMENT "Running python test for tensorflow operation"
)

#INSTALL(TARGETS monotonic_rnnt_lib
#        RUNTIME DESTINATION "bin"
#        LIBRARY DESTINATION "lib"
#        ARCHIVE DESTINATION "lib")
#
#INSTALL(FILES include/rnnt_entrypoint.h DESTINATION "include")
