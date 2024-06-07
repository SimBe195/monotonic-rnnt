cmake_minimum_required(VERSION 3.18)
project(monotonic_rnnt LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})

option(USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(DEBUG_TIME "output kernel time" OFF)
option(DEBUG_SPACE "output workspace size" OFF)
option(DEBUG_LOG_SOFTMAX "output log softmax denominator" OFF)
option(DEBUG_FWDBWD "output alpha beta" OFF)
option(DEBUG_GRAD "output gradients" OFF)
option(WITH_GPU "compile monotonic-rnnt with cuda." ON)
option(WITH_OMP "compile monotonic-rnnt with openmp." ON)
option(INCLUDE_TF_OP "compile tensorflow operation" ON)

if (USE_NAIVE_KERNEL)
  add_definitions(-DUSE_NAIVE_KERNEL)
endif ()
if (DEBUG_TIME)
  add_definitions(-DDEBUG_TIME)
endif ()
if (DEBUG_SPACE)
  add_definitions(-DDEBUG_SPACE)
endif ()
if (DEBUG_LOG_SOFTMAX)
  add_definitions(-DDEBUG_LOG_SOFTMAX)
endif ()
if (DEBUG_FWDBWD)
  add_definitions(-DDEBUG_FWDBWD)
endif ()
if (DEBUG_GRAD)
  add_definitions(-DDEBUG_GRAD)
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

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${INCLUDE_DIR}")

  add_library(monotonic_rnnt SHARED src/rnnt_entrypoint.cu)

  add_executable(test_time_gpu tests/test_time.cu tests/random.cu)

  add_executable(test_gpu tests/test_gpu.cu tests/random.cu)
else ()
  message(STATUS "Building CPU version")

  add_library(monotonic_rnnt SHARED src/rnnt_entrypoint.cpp)
endif ()

add_executable(test_time_cpu tests/test_time.cpp tests/random.cpp)

add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp)

if (INCLUDE_TF_OP)
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

  # set(Tensorflow_FRAMEWORK ${Tensorflow_LIB_DIR}/libtensorflow.so.2)

  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
  find_library(
            Tensorflow_FRAMEWORK
            NAMES tensorflow_framework tensorflow_framework.so.2
            HINTS ${Tensorflow_LIB_DIR}
            PATH_SUFFIXES lib
            REQUIRED
    )

  add_library(monotonic_rnnt_tf_op SHARED tensorflow_binding/monotonic_rnnt_op.cu tensorflow_binding/monotonic_rnnt_op_align_restrict.cu)
  target_include_directories(monotonic_rnnt_tf_op PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(monotonic_rnnt_tf_op PUBLIC ${Tensorflow_INCLUDE_DIR})
  target_link_libraries(monotonic_rnnt_tf_op PUBLIC ${Tensorflow_FRAMEWORK})
endif()

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#
# add_library(monotonic_rnnt_torch_op SHARED pytorch_binding/monotonic_rnnt_op.cu)
# target_include_directories(monotonic_rnnt_torch_op PUBLIC ${TORCH_INCLUDE_DIRS})
# target_link_libraries(monotonic_rnnt_torch_op PUBLIC ${TORCH_LIBRARIES})

INSTALL(TARGETS monotonic_rnnt
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib")

INSTALL(FILES include/rnnt_entrypoint.h include/cpu_workspace_manager.h include/gpu_workspace_manager.h DESTINATION "include")
