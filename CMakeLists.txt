cmake_minimum_required(VERSION 3.16)

project(ReidBLAS VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(REIDBLAS_IMPL_SOURCES
    src/impl/reidblas_backend_ilp64.cpp
)

add_library(reidblas_impl STATIC
    ${REIDBLAS_IMPL_SOURCES}
)
set_target_properties(reidblas_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(reidblas_impl PUBLIC cxx_std_17)
target_include_directories(reidblas_impl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

add_library(reidblas SHARED
    src/reidblas.cpp
    src/reidblas_blas_ilp64_dispatch.cpp
    src/reidblas_blas_lp64_wrappers.cpp
)
set_target_properties(reidblas PROPERTIES
    OUTPUT_NAME "reidblas"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(reidblas PUBLIC cxx_std_17)
target_include_directories(reidblas PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/reidblas>
)
target_compile_definitions(reidblas PUBLIC REIDBLAS_USE_ILP64=0)
target_link_libraries(reidblas PRIVATE reidblas_impl)

add_library(reidblas_ilp64 SHARED
    src/reidblas.cpp
    src/reidblas_blas_ilp64_dispatch.cpp
)
set_target_properties(reidblas_ilp64 PROPERTIES
    OUTPUT_NAME "reidblas_ilp64"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(reidblas_ilp64 PUBLIC cxx_std_17)
target_include_directories(reidblas_ilp64 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/reidblas>
)
target_compile_definitions(reidblas_ilp64 PUBLIC REIDBLAS_USE_ILP64=1)
target_link_libraries(reidblas_ilp64 PRIVATE reidblas_impl)

add_library(ReidBLAS::reidblas ALIAS reidblas)
add_library(ReidBLAS::reidblas_ilp64 ALIAS reidblas_ilp64)

set(REIDBLAS_HEADERS
    src/reidblas.hpp
    src/reidblas_blas_lp64.h
    src/reidblas_blas_ilp64.h
    src/reidblas_blas_template.h
)

install(TARGETS reidblas reidblas_ilp64
    EXPORT ReidBLASTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/reidblas
)

install(FILES ${REIDBLAS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/reidblas
)

install(EXPORT ReidBLASTargets
    NAMESPACE ReidBLAS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReidBLAS
)

configure_package_config_file(
    cmake/ReidBLASConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ReidBLASConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReidBLAS
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ReidBLASConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ReidBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ReidBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReidBLAS
)

export(
    EXPORT ReidBLASTargets
    NAMESPACE ReidBLAS::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/ReidBLASTargets.cmake
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
    add_subdirectory(test)
endif()
