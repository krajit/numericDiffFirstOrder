cmake_minimum_required (VERSION 3.8)
project(rosenbrock)

find_package(Ceres)
include_directories(${CERES_INCLUDE_DIRS})


# Make/files
# Add source to this project's executable.
add_executable (rosenbrock "rosenbrock.cc")

# target_link_libraries(ceresLaplacianOptimal ${LIBS})
target_link_libraries(rosenbrock ${CERES_LIBRARIES})