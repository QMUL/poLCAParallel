cmake_minimum_required(VERSION 3.16)
project(poLCAParallel VERSION 1.2.4 HOMEPAGE_URL https://github.com/QMUL/poLCAParallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
include_directories(include)

file(GLOB ALL_SRC_FILES CONFIGURE_DEPENDS src/*.cc)
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cc)
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*_rcpp\\.cc$")

find_package(Armadillo REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_compile_definitions(ARMA_WARN_LEVEL=1)

add_executable(test_polca_parallel
  ${ALL_SRC_FILES}
  ${TEST_SOURCES}
)
target_link_libraries(test_polca_parallel PRIVATE armadillo LAPACK::LAPACK BLAS::BLAS Catch2::Catch2WithMain)
target_include_directories(test_polca_parallel PRIVATE tests/)
