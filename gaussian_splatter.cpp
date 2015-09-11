//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014. Los Alamos National Security
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <random>

#ifndef VTKM_DEVICE_ADAPTER
# define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL
#endif

#define HPX_TIMING

#ifdef HPX_TIMING
# include <chrono>
  std::size_t os_threads;

// start timer
# define START_TIMER_BLOCK(name) \
  std::chrono::time_point<std::chrono::system_clock> start_##name, end_##name; \
  start_##name = std::chrono::system_clock::now();

// stop timer
# define END_TIMER_BLOCK(name) \
  end_##name = std::chrono::system_clock::now(); \
  std::chrono::duration<double> elapsed_##name = end_##name-start_##name; \
  std::cout << "CSVData " \
    << ", threads, "     << os_threads \
    << ", " #name "_time, " << elapsed_##name.count() << std::endl;

#else
# define START_TIMER_BLOCK(name)
# define END_TIMER_BLOCK(name)
#endif

#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/DynamicArrayHandle.h>
#include <vtkm/cont/Field.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/GaussianSplatter.h>
#include <vtkm/worklet/IsosurfaceUniformGrid.h>
#include <vtkm/Pair.h>

//now that the device adapter is included set a global typedef
//that is the chosen device tag
typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

#ifdef _WIN32
# include "windows.h""
#endif

#include "isosurface.h"

#if defined (__APPLE__)
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
#endif
//
#include "display.cpp"
//
typedef vtkm::FloatDefault FieldType;
typedef vtkm::Vec<FieldType, 3> floatVec;

//----------------------------------------------------------------------------
// Global variables
//----------------------------------------------------------------------------
vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3> > verticesArray, normalsArray;
vtkm::cont::ArrayHandle<vtkm::Float32> scalarsArray;

//----------------------------------------------------------------------------
// Empty Dataset
//----------------------------------------------------------------------------
vtkm::cont::DataSet MakeEmptyVolumeDataset(vtkm::Id3 dims, const floatVec &origin, const floatVec &spacing)
{
  vtkm::cont::DataSet dataSet;

  const vtkm::Id3 vdims(dims[0]+1, dims[1]+1, dims[2]+1);

  vtkm::cont::ArrayHandleUniformPointCoordinates coordinates(vdims, origin, spacing);
  dataSet.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coordinates", 1, coordinates));

  static const vtkm::IdComponent ndim = 3;
  vtkm::cont::CellSetStructured<ndim> cellSet("cells");
  cellSet.SetPointDimensions(vdims);
  dataSet.AddCellSet(cellSet);
  
  return dataSet;
}
//----------------------------------------------------------------------------
// Run our test
//----------------------------------------------------------------------------
int init_pipeline(int argc, char* argv[])
{
#if VTKM_DEVICE_ADAPTER == VTKM_DEVICE_ADAPTER_HPX
  os_threads = hpx::get_os_thread_count();
  std::cout << "Running HPX with threadcount " << os_threads << std::endl;
#endif
  // Abort if dimension and file name are not provided
  if (argc < 3)
  {
    std::cout << "Usage: isosurface {dimension} {isovalue} {optional-file-name} " << std::endl;
    return 0;
  }

  //
  // get command line values
  // NB. Cell dimension is dim, points are dim+1 in each dimension
  //
  int dim = atoi(argv[1]);
  float isovalue = atof(argv[2]);
  const char *fileName = argc>=4 ? argv[3] : nullptr;
  //
  const vtkm::Id3 dims(dim, dim, dim);
  const vtkm::Id3 vdims(dim+1, dim+1, dim+1);
  const floatVec origin(-dim / 2.0, -dim / 2.0, -dim / 2.0);
  const floatVec spacing(1.0, 1.0, 1.0);

  //
  // Create a volume, specify how many cells in each dim
  //
  vtkm::cont::DataSet dataSet = MakeEmptyVolumeDataset(dims, origin, spacing);

  START_TIMER_BLOCK(splatter)

  //
  // create a field by splatting particles into our volume
  //
  const int num_particles = 11;
  std::vector<double> xdata(num_particles), ydata(num_particles), zdata(num_particles);
  std::vector<float>  rdata(num_particles);
  std::random_device rd;
  std::mt19937 gen(rd());
  // give the particles real world space positions inside the volume
  std::uniform_real_distribution<> x_dis(origin[0], origin[0] + spacing[0]*dims[0]);
  std::uniform_real_distribution<> y_dis(origin[1], origin[1] + spacing[1]*dims[1]);
  std::uniform_real_distribution<> z_dis(origin[2], origin[2] + spacing[2]*dims[2]);
  // give each point a radius which is between 0 and 5 voxels wide for now
  std::uniform_real_distribution<> radius(0.0, spacing[0]*1.0);
  std::generate(xdata.begin(), xdata.end(), [&]{return x_dis(gen);});
  std::generate(ydata.begin(), ydata.end(), [&]{return y_dis(gen);});
  std::generate(zdata.begin(), zdata.end(), [&] {return z_dis(gen); });
  std::generate(rdata.begin(), rdata.end(), [&] {return radius(gen); });
//  xdata[0] = 0.5;
//  ydata[0] = 0.5;
//  zdata[0] = 0.5;
//  rdata[0] = 1.0;

  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> xValues;
  xValues = vtkm::cont::make_ArrayHandle(xdata);
  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> yValues;
  yValues = vtkm::cont::make_ArrayHandle(ydata);
  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> zValues;
  zValues = vtkm::cont::make_ArrayHandle(zdata);
  vtkm::cont::ArrayHandle<vtkm::Float32, VTKM_DEFAULT_STORAGE_TAG> rValues;
  rValues = vtkm::cont::make_ArrayHandle(rdata);

  OutputArrayDebug(xValues, "x Values");
  OutputArrayDebug(yValues, "y Values");
  OutputArrayDebug(zValues, "z Values");
  OutputArrayDebug(rValues, "Radii");

  vtkm::cont::ArrayHandle<vtkm::Float32> fieldArray;

  vtkm::worklet::GaussianSplatterFilterUniformGrid<DeviceAdapter> splatter(dims, origin, spacing, dataSet);

  splatter.run<VTKM_DEFAULT_STORAGE_TAG, VTKM_DEFAULT_STORAGE_TAG>
    (xValues, yValues, zValues, rValues, fieldArray);

  END_TIMER_BLOCK(splatter)

  OutputArrayDebug(fieldArray, "fieldArray");

  //
  // add field to the dataset
  //
  dataSet.AddField(vtkm::cont::Field("nodevar", 1, vtkm::cont::Field::ASSOC_POINTS, fieldArray));

  //
  // Create isosurface filter, use cell dimensions to initialize
  //
  vtkm::worklet::IsosurfaceFilterUniformGrid<vtkm::Float32, DeviceAdapter> isosurfaceFilter(dims, dataSet);

  START_TIMER_BLOCK(isosurface)

  //
  // and compute the isosurface
  //
  isosurfaceFilter.Run(isovalue,
                        dataSet.GetField("nodevar").GetData(),              
                        verticesArray,
                        normalsArray,
                        fieldArray);

  END_TIMER_BLOCK(isosurface)

  return 0;
}


#if VTKM_DEVICE_ADAPTER == VTKM_DEVICE_ADAPTER_HPX
int main(int argc, char **argv)
{
  {
    // setup all the pipeline stuff and wait till it's done
    hpx::future<int> init_p = hpx::async(init_pipeline, argc, argv);
    init_p.wait();

    // Get a reference to one of the main OS threads
    hpx::threads::executors::main_pool_executor scheduler;

    // create gui on OS thread
    hpx::future<GLFWwindow*> init_g = hpx::async(scheduler, init_glfw, 800, 800);
    GLFWwindow *window = init_g.get();

    typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > vertextype;
    typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > normaltype;

    std::function<void()> display_function = std::bind(&displayCall<vertextype, normaltype>,
                                                       verticesArray,
                                                       normalsArray);

    // run graphics loop on OS thread
    hpx::future<void> loop = hpx::async(scheduler, &run_graphics_loop, window, display_function);

    // can do something else while loop is executing in the background ...
    loop.wait();
  }
  return hpx::finalize();
}

#else
int main(int argc, char **argv)
{
  // setup all the pipeline stuff and wait till it's done
  init_pipeline(argc, argv);
  GLFWwindow *window = init_glfw(800, 800);
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > vertextype;
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > normaltype;

  std::function<void()> display_function = std::bind(&displayCall<vertextype, normaltype>,
                                                     verticesArray,
                                                     normalsArray);
  run_graphics_loop(window, display_function);
  return 0;
}
#endif

