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
cmake_minimum_required(VERSION 2.8.11)
find_package(OpenGL)
find_package(GLUT)

find_package(Cuda)

include_directories("/Users/biddisco/src/vtkm/")
include_directories("/Users/biddisco/build/vtkm/")

add_executable(isosurface isosurface.cpp)
target_include_directories(isosurface PRIVATE ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
target_link_libraries(isosurface ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(cuda_isosurface isosurface.cu)
target_link_libraries(cuda_isosurface ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
