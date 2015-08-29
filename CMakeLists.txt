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
find_package(Boost 1.55 REQUIRED)

# checkout master branch, but don't pull all the history
Git_External(
  ${CMAKE_CURRENT_SOURCE_DIR}/glfw
  https://github.com/glfw/glfw.git
  master
  VERBOSE
  SHALLOW
)

#--------------------------------------------------
# Find glfw
#--------------------------------------------------
#  option(GLUT_DOWNLOAD_AS_SUBPROJECT OFF)
#  if(GLUT_DOWNLOAD_AS_SUBPROJECT)
option(GLFW_BUILD_DOCS                  OFF "")
option(GLFW_BUILD_EXAMPLES              OFF "")
option(GLFW_BUILD_TESTS                 OFF "")

add_subdirectory(glfw)

#--------------------------------------------------
# include dirs
#--------------------------------------------------
include_directories(
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include
    ${OPENGL_INCLUDE_DIR}
    ${VTKm_INCLUDE_DIRS}
)
message("VTKM inc dir is ${VTKm_INCLUDE_DIRS}")
add_executable(isosurface isosurface.cpp)
target_include_directories(isosurface PRIVATE ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
target_link_libraries(isosurface glfw ${GLFW_LIBRARIES})

if (VTKm_ENABLE_CUDA)
  cuda_add_executable(cuda_isosurface isosurface.cu)
  target_link_libraries(cuda_isosurface glfw ${GLFW_LIBRARIES})
endif()

if (VTKm_ENABLE_HPX)
  add_definitions(-DBOOST_ALL_DYN_LINK)
  link_directories(${Boost_LIBRARY_DIRS})
  add_executable(hpx_isosurface isosurface_hpx.cpp)
  target_link_libraries(hpx_isosurface
    iostreams_component
    ${Boost_LIBRARIES}
    glfw
    ${GLFW_LIBRARIES}
  )
  hpx_setup_target(hpx_isosurface)

#  add_executable(hpx_compositor compositor_hpx.cpp)
#  target_link_libraries(hpx_compositor ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES})
#  hpx_setup_target(hpx_compositor)
#  target_link_libraries(hpx_compositor
#    iostreams_component
#    ${Boost_LIBRARIES}
#  )
endif()


