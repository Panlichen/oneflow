set(BUILD_CUDA NO CACHE BOOL "")
set(BUILD_GIT_VERSION YES CACHE BOOL "")
set(BUILD_TESTING YES CACHE BOOL "")
set(WITH_ONEDNN YES CACHE BOOL "")
set(TREAT_WARNINGS_AS_ERRORS YES CACHE BOOL "")
set(THIRD_PARTY_MIRROR aliyun CACHE STRING "")
set(PIP_INDEX_MIRROR "https://pypi.tuna.tsinghua.edu.cn/simple" CACHE STRING "")
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
set(CMAKE_GENERATOR Ninja CACHE STRING "")
set(BUILD_CPP_API ON CACHE BOOL "")
set(WITH_MLIR ON CACHE BOOL "")
set(BUILD_FOR_CI ON CACHE BOOL "")
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CPU_THREADING_RUNTIME "SEQ" CACHE STRING "")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "")
set(ENABLE_ASAN ON CACHE BOOL "")
set(ENABLE_UBSAN OFF CACHE BOOL "")
