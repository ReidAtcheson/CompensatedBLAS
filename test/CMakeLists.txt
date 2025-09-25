find_package(GTest REQUIRED)

add_executable(reidblas_tests
    reidblas_test.cpp
)

target_link_libraries(reidblas_tests
    PRIVATE
        reidblas
        GTest::gtest_main
)

target_compile_features(reidblas_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(reidblas_tests)
