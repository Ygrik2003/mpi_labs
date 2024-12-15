cmake_minimum_required(VERSION 3.28.2)

project(MPI_labs CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(EnvModules REQUIRED)
find_package(
  Boost
  COMPONENTS Log CONFIG
  REQUIRED)
find_package(OpenMP REQUIRED)

# cmake-format: off
# For install in fedora MPI use:
# sudo dnf install mpich-devel
# sudo dnf install openmpi-devel
# cmake-format: on
env_module(load mpi/openmpi-x86_64)
find_package(MPI REQUIRED)

add_executable(mpi_1 mpi_1.cpp)
target_link_libraries(mpi_1 PUBLIC MPI::MPI_CXX Boost::log)

add_executable(mpi_2 mpi_2.cpp)
target_link_libraries(mpi_2 PUBLIC MPI::MPI_CXX Boost::log)

add_executable(mpi_3 mpi_3.cpp)
target_link_libraries(mpi_3 PUBLIC MPI::MPI_CXX Boost::log)

add_executable(mpi_4 mpi_4.cpp)
target_link_libraries(mpi_4 PUBLIC MPI::MPI_CXX Boost::log)

add_executable(mpi_5 mpi_5.cpp)
target_link_libraries(mpi_5 PUBLIC OpenMP::OpenMP_CXX Boost::log)
