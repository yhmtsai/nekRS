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
    FetchContent_MakeAvailable(Ginkgo)
endif()