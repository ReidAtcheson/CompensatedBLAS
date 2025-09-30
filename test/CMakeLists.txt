find_package(GTest REQUIRED)

add_executable(compensated_blas_tests
    compensated_blas_test.cpp
    kdiv_test.cpp
)

target_link_libraries(compensated_blas_tests
    PRIVATE
        compensatedblas
        compensatedblas_impl
        GTest::gtest_main
)

target_compile_features(compensated_blas_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(compensated_blas_tests)
