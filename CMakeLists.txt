##============================================================================
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2014 Sandia Corporation.
##  Copyright 2014 UT-Battelle, LLC.
##  Copyright 2014. Los Alamos National Security
##
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
##  Laboratory (LANL), the U.S. Government retains certain rights in
##  this software.
##============================================================================
cmake_minimum_required(VERSION 3.0.0)

#--------------------------------------------------
# cmake policies
# see http://www.cmake.org/cmake/help/v2.8.12/cmake.html#section_Policies
#--------------------------------------------------
foreach(p
    CMP0003 CMP0017 CMP0023 CMP0042
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

#--------------------------------------------------
# Setup Output directories
#--------------------------------------------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
)
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
)
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
)

#--------------------------------------------------
# load script for checking out projects from git
#--------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/GitExternal.cmake)

#--------------------------------------------------
# Windows link problems with GLUT
#--------------------------------------------------
#set(BUILD_SHARED_LIBS ON)

#--------------------------------------------------
# Find HPX
#--------------------------------------------------
find_package(HPX)
include_directories(${HPX_INCLUDE_DIRS})

#--------------------------------------------------
# Find VTKm
#--------------------------------------------------
find_package(VTKm)

#--------------------------------------------------
# Find OpenGL
#--------------------------------------------------
find_package(OpenGL)
if (NOT OPENGL_GLU_FOUND)
  message(error "Did not find GLU")
endif()

#--------------------------------------------------
# Find Cuda
#--------------------------------------------------
find_package(Cuda)

#--------------------------------------------------
# Find Boost
#--------------------------------------------------
set(Boost_ADDITIONAL_VERSIONS
  ${Boost_ADDITIONAL_VERSIONS}
  "1.59.0" "1.59"
  "1.58.0" "1.58"
  "1.57.0" "1.57")

find_package(Boost
  1.55
  REQUIRED
  COMPONENTS
    atomic chrono date_time filesystem program_options regex system thread
)

#--------------------------------------------------
# OSX warnings
#--------------------------------------------------
# clang
add_definitions(-Wno-deprecated-declarations)
#gcc  -Wno-deprecated)

#--------------------------------------------------
# Find GLFW
#--------------------------------------------------
find_package(GLFW QUIET)
if(NOT GLFW_FOUND)
  option(GLFW_DOWNLOAD_AS_SUBPROJECT ON)
  if(GLFW_DOWNLOAD_AS_SUBPROJECT)
      #--------------------------------------------------
      # Checkout tip of GLFW repository
      #--------------------------------------------------
      Git_External(
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw
        https://github.com/glfw/glfw.git
        master
        VERBOSE
        SHALLOW
      )

      option(GLFW_BUILD_DOCS                  OFF "")
      option(GLFW_BUILD_EXAMPLES              OFF "")
      option(GLFW_BUILD_TESTS                 OFF "")

      add_subdirectory(glfw)
      set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)
    endif()
endif()

#--------------------------------------------------
# include dirs
#--------------------------------------------------
include_directories(
    ${Boost_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${VTKm_INCLUDE_DIRS}
)

#--------------------------------------------------
# libs
#--------------------------------------------------
set(TEST_LIBS
  glfw ${GLFW_LIBRARIES} ${OPENGL_glu_LIBRARY} 
)

#--------------------------------------------------
# tests to compile
#--------------------------------------------------
set(TESTS
  isosurface
  gaussian_splatter
)

#--------------------------------------------------
# normal, cuda, hpx versions
#--------------------------------------------------
foreach(test ${TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} ${TEST_LIBS})

  if (VTKm_ENABLE_CUDA)
    cuda_add_executable(cuda_${test} ${test}.cu)
    target_link_libraries(cuda_${test} ${TEST_LIBS})
  endif()

  if (VTKm_ENABLE_HPX)
    add_executable(hpx_${test} ${test}_hpx.cpp)
    target_link_libraries(hpx_${test}
      iostreams_component
      ${Boost_LIBRARIES}
      ${TEST_LIBS}
    )
    if (NOT Boost_USE_STATIC_LIBS)
      target_compile_definitions(hpx_${test} PUBLIC BOOST_ALL_DYN_LINK)
    endif()
    hpx_setup_target(hpx_${test})
  endif()

endforeach(test)
