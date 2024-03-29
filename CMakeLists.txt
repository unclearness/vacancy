cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME vacancy)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "Vacancy: A Voxel Carving implementation in C++")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "/std:c++14 /W4 /EHsc /MP")


else()
# g++ option for *nix

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #Set this to ON if you want to use GNU++
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O2")


endif()

# third_party directries
if(NOT DEFINED STB_INSTALL_DIR)
  set(STB_INSTALL_DIR "third_party/stb" CACHE PATH "stb installed directory")
endif()
if(NOT DEFINED EIGEN_INSTALL_DIR)
  set(EIGEN_INSTALL_DIR "third_party/eigen" CACHE PATH "eigen installed directory")
message("EIGEN_INSTALL_DIR: ${EIGEN_INSTALL_DIR}") 
endif()

# switch for optional libraries
option(VACANCY_USE_STB "Use stb to enable image i/o" ON)
message("VACANCY_USE_STB: ${VACANCY_USE_STB}")
if(VACANCY_USE_STB)
  option(VACANCY_USE_STB_AS_STATIC_LIB "Use stb as static lib" OFF)
  message("VACANCY_USE_STB_AS_STATIC_LIB: ${VACANCY_USE_STB_AS_STATIC_LIB}")
  if(VACANCY_USE_STB_AS_STATIC_LIB)
    add_library(stb
      STATIC
      src/vacancy/stb.cc)
    set(STB_LIB_NAME stb)
    target_include_directories(${STB_LIB_NAME} PUBLIC third_party)
    target_compile_definitions(${STB_LIB_NAME} PUBLIC -DVACANCY_USE_STB)
  else()
    set(VACANCY_STB_IMPLEMENTATION_CC src/vacancy/stb.cc)
  endif()
endif()

option(VACANCY_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("VACANCY_USE_OPENMP: ${VACANCY_USE_OPENMP}")
if(VACANCY_USE_OPENMP)
  add_definitions(-DVACANCY_USE_OPENMP)
endif()

set(PUBLIC_LIB_NAME ${PROJECT_NAME})
add_library(${PUBLIC_LIB_NAME}
  STATIC
  include/vacancy/common.h
  include/vacancy/camera.h
  include/vacancy/mesh.h
  include/vacancy/image.h
  include/vacancy/log.h
  include/vacancy/voxel_carver.h

  src/vacancy/camera.cc
  src/vacancy/mesh.cc
  src/vacancy/image.cc
  src/vacancy/timer.h
  src/vacancy/log.cc
  src/vacancy/voxel_carver.cc
  src/vacancy/marching_cubes.h
  src/vacancy/marching_cubes.cc
  src/vacancy/marching_cubes_lut.h
  src/vacancy/marching_cubes_lut.cc
  src/vacancy/extract_voxel.h
  src/vacancy/extract_voxel.cc

  # implementations of header-only library
  ${VACANCY_STB_IMPLEMENTATION_CC}
)

if (VACANCY_USE_STB)
  target_compile_definitions(${PUBLIC_LIB_NAME} PUBLIC -DVACANCY_USE_STB)
endif()

# For OpenMP
if(VACANCY_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
      target_compile_options(${PUBLIC_LIB_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:${OPENMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:C>:${OPENMP_C_FLAGS}>)
  endif()
endif()

# set folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

SOURCE_GROUP("Public header files" FILES
  include/vacancy/common.h
  include/vacancy/camera.h
  include/vacancy/mesh.h
  include/vacancy/image.h
  include/vacancy/log.h
  include/vacancy/voxel_carver.h
)

SOURCE_GROUP("Private header files" FILES
  src/vacancy/timer.h
  src/vacancy/marching_cubes.h
  src/vacancy/marching_cubes_lut.h
  src/vacancy/extract_voxel.h
)

SOURCE_GROUP("Source files" FILES
  src/vacancy/camera.cc
  src/vacancy/mesh.cc
  src/vacancy/image.cc
  src/vacancy/log.cc
  src/vacancy/voxel_carver.cc
  src/vacancy/marching_cubes.cc
  src/vacancy/marching_cubes_lut.cc
  src/vacancy/extract_voxel.cc
)

target_include_directories(${PUBLIC_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${PUBLIC_LIB_NAME} PUBLIC third_party)
target_include_directories(${PUBLIC_LIB_NAME} PUBLIC ${EIGEN_INSTALL_DIR})
target_include_directories(${PUBLIC_LIB_NAME} PUBLIC ${STB_INSTALL_DIR})

set_target_properties(${PUBLIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set(EXAMPLES_EXE vacancy_examples)
add_executable(${EXAMPLES_EXE}
  examples.cc)
target_link_libraries(${EXAMPLES_EXE}
  ${PUBLIC_LIB_NAME}
  ${STB_LIB_NAME}
  )
if (WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXAMPLES_EXE})
  # suppress C2338 for eigen
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()
