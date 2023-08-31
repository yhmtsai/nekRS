find_package(Ginkgo 1.5.0 QUIET) 

if(NOT Ginkgo_FOUND)
    message(STATUS "Fetching external Ginkgo")
    include(FetchContent)
    FetchContent_Declare(
        Ginkgo
        GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
        GIT_TAG        e831a097488e60196d3778f41a2da86676a51108
    )
    set(GINKGO_BUILD_CUDA ON CACHE INTERNAL "")
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
