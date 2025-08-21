cmake_minimum_required(VERSION 3.16)
project(poLCAParallel VERSION 1.2.3 HOMEPAGE_URL https://github.com/QMUL/poLCAParallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
include_directories(include)

file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS src/*.cc)
list(FILTER ALL_SRC_FILES EXCLUDE REGEX ".*_rcpp\\.cc$")

find_package(Armadillo REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_compile_definitions(ARMA_WARN_LEVEL=1)

set(TEST_UTIL tests/util_test.cc)

function(add_test_executable name source)
  add_executable(${name}
    ${ALL_SRC_FILES}
    ${source}
    ${TEST_UTIL}
  )
  target_link_libraries(${name} PRIVATE armadillo LAPACK::LAPACK BLAS::BLAS Catch2::Catch2WithMain)
  target_include_directories(${name} PRIVATE tests/)
endfunction()

add_test_executable(test_em_algorithm_array         tests/test_em_algorithm_array.cc)
add_test_executable(test_em_algorithm_array_serial  tests/test_em_algorithm_array_serial.cc)
add_test_executable(test_em_algorithm               tests/test_em_algorithm.cc)
add_test_executable(test_em_algorithm_regress       tests/test_em_algorithm_regress.cc)
add_test_executable(test_em_algorithm_nan           tests/test_em_algorithm_nan.cc)
add_test_executable(test_standard_error             tests/test_standard_error.cc)
add_test_executable(test_standard_error_regress     tests/test_standard_error_regress.cc)
add_test_executable(test_regularised_error          tests/test_regularised_error.cc)
add_test_executable(test_smoother                   tests/test_smoother.cc)
add_test_executable(test_goodness_fit               tests/test_goodness_fit.cc)
add_test_executable(test_blrt                       tests/test_blrt.cc)
