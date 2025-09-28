cmake_minimum_required(VERSION 3.16)

project(CompensatedBLAS VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(COMPENSATEDBLAS_IMPL_SOURCES
    src/impl/compensated_blas_backend_ilp64.cpp
    src/impl/runtime.cpp
    src/impl/naive_blas_backend.cpp
)

add_library(compensatedblas_impl STATIC
    ${COMPENSATEDBLAS_IMPL_SOURCES}
)
set_target_properties(compensatedblas_impl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(compensatedblas_impl PUBLIC cxx_std_17)
target_include_directories(compensatedblas_impl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

add_library(compensatedblas SHARED
    src/compensated_blas.cpp
    src/compensated_blas_ilp64_dispatch.cpp
    src/compensated_blas_lp64_wrappers.cpp
)
set_target_properties(compensatedblas PROPERTIES
    OUTPUT_NAME "CompensatedBLAS"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(compensatedblas PUBLIC cxx_std_17)
target_include_directories(compensatedblas PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/compensatedblas>
)
target_compile_definitions(compensatedblas PUBLIC COMPENSATEDBLAS_USE_ILP64=0)
target_link_libraries(compensatedblas PRIVATE compensatedblas_impl)

add_library(compensatedblas_ilp64 SHARED
    src/compensated_blas.cpp
    src/compensated_blas_ilp64_dispatch.cpp
)
set_target_properties(compensatedblas_ilp64 PROPERTIES
    OUTPUT_NAME "CompensatedBLAS_ilp64"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_compile_features(compensatedblas_ilp64 PUBLIC cxx_std_17)
target_include_directories(compensatedblas_ilp64 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/compensatedblas>
)
target_compile_definitions(compensatedblas_ilp64 PUBLIC COMPENSATEDBLAS_USE_ILP64=1)
target_link_libraries(compensatedblas_ilp64 PRIVATE compensatedblas_impl)

add_library(CompensatedBLAS::compensatedblas ALIAS compensatedblas)
add_library(CompensatedBLAS::compensatedblas_ilp64 ALIAS compensatedblas_ilp64)

set(COMPENSATEDBLAS_HEADERS
    src/compensated_blas.hpp
    src/compensated_blas_lp64.h
    src/compensated_blas_ilp64.h
    src/compensated_blas_template.h
)

install(TARGETS compensatedblas compensatedblas_ilp64
    EXPORT CompensatedBLASTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/compensatedblas
)

install(FILES ${COMPENSATEDBLAS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/compensatedblas
)

install(EXPORT CompensatedBLASTargets
    NAMESPACE CompensatedBLAS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CompensatedBLAS
)

configure_package_config_file(
    cmake/CompensatedBLASConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CompensatedBLASConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CompensatedBLAS
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CompensatedBLASConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CompensatedBLASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CompensatedBLASConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CompensatedBLAS
)

export(
    EXPORT CompensatedBLASTargets
    NAMESPACE CompensatedBLAS::
    FILE ${CMAKE_CURRENT_BINARY_DIR}/CompensatedBLASTargets.cmake
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(examples)
    enable_testing()
    add_subdirectory(test)
endif()
