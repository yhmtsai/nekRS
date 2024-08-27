# find_package(Ginkgo 1.8.0 QUIET) 
# avoid add_sycl_to_target(TARGET) only in intel.
# it makes `-fsycl` public such that C files compilation are failed.
if(NOT Ginkgo_FOUND)
    message(STATUS "Fetching external Ginkgo")
    include(FetchContent)
    FetchContent_Declare(
        Ginkgo
        GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
        GIT_TAG        no_public_fsycl
    )
    set(GINKGO_BUILD_CUDA ${OCCA_CUDA_ENABLED} CACHE INTERNAL "")
    set(GINKGO_BUILD_HIP ${OCCA_HIP_ENABLED} CACHE INTERNAL "")
    set(GINKGO_BUILD_SYCL ${OCCA_DPCPP_ENABLED} CACHE INTERNAL "")
    set(GINKGO_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
    set(GINKGO_BUILD_TESTS OFF CACHE INTERNAL "")
    set(GINKGO_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    set(GINKGO_FORCE_GPU_AWARE_MPI ${NEKRS_GPU_MPI} CACHE INTERNAL "")
    FetchContent_MakeAvailable(Ginkgo)
endif()

find_package(nlohmann_json 3.9.1 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "Fetching external nlohmann_json")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.9.1
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
endif()