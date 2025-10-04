add_executable(conjugate_residual_solver
    conjugate_residual_solver.cpp
)
target_link_libraries(conjugate_residual_solver PRIVATE compensatedblas compensatedblas_impl)

add_executable(syrk_compensation_demo
    syrk_compensation_demo.cpp
)
target_link_libraries(syrk_compensation_demo PRIVATE compensatedblas compensatedblas_impl)

find_package(LAPACK)
if(LAPACK_FOUND)
    add_executable(eigenvalue_helper
        eigenvalue_helper.cpp
    )
    target_link_libraries(eigenvalue_helper PRIVATE LAPACK::LAPACK)
else()
    message(STATUS "LAPACK not found; eigenvalue_helper will not be built")
endif()
