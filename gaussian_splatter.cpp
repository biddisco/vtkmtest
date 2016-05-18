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

# ifndef START_TIMER_BLOCK
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
# endif
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
#include <vtkm/worklet/KernelSplatter.h>
#include <vtkm/filter/MarchingCubes.h>
#include <vtkm/Pair.h>

//now that the device adapter is included set a global typedef
//that is the chosen device tag
typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

#ifdef _WIN32
# include "windows.h""
#endif

#include <boost/bind.hpp>
//
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
typedef vtkm::Vec<vtkm::Float64, 3> doubleVec;

//----------------------------------------------------------------------------
// Global variables
//----------------------------------------------------------------------------
vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3> > verticesArray, normalsArray;
vtkm::cont::ArrayHandle<vtkm::Float32> scalarsArray;
floatVec view_origin(0.0, 0.0, 0.0);
float    view_dist = 10;
floatVec min_cube(0.0, 0.0, 0.0);
floatVec max_cube(1.0, 1.0, 1.0);

//----------------------------------------------------------------------------
// Empty Dataset
//----------------------------------------------------------------------------
vtkm::cont::DataSet MakeEmptyVolumeDataset(vtkm::Id3 dims, const floatVec &origin, const floatVec &spacing)
{
  vtkm::cont::DataSet dataSet;

  const vtkm::Id3 vdims(dims[0]+1, dims[1]+1, dims[2]+1);

  vtkm::cont::ArrayHandleUniformPointCoordinates coordinates(vdims, origin, spacing);
  dataSet.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coordinates", coordinates));

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
    std::cout << "Usage: splatter {dimension} {points} {radius} {isovalue} {optional-file-name} " << std::endl;
    return 0;
  }

  //
  // get command line values
  // NB. Cell dimension is dim, points are dim+1 in each dimension
  //
  int dim = atoi(argv[1]);
  int points = atoi(argv[2]);
  int radius = atof(argv[3]);
  float isovalue = atof(argv[4]);
  //
  const vtkm::Id3 dims(dim, dim, dim);
  const vtkm::Id3 vdims(dim+1, dim+1, dim+1);
  const doubleVec fdim(dims[0], dims[1], dims[2]);
  //
  // shift origin so that volume is centred on {0,0,0}
  const doubleVec spacing(1.0, 1.0, 1.0);
//  const floatVec origin(-spacing[0]*fdim[0]*0.5, -spacing[1]*fdim[1]*0.5, -spacing[2]*fdim[0]*0.5);
  const doubleVec origin(0.0, 0.0, 0.0);
  // for our GL window
  min_cube = origin;
  max_cube = /*origin + */(spacing*fdim);
  view_origin = static_cast<vtkm::FloatDefault>(0.5) * (fdim*spacing); // +
  // static_cast<vtkm::FloatDefault>(0.5) * (fdim*spacing);
  view_origin = origin;
  view_dist   = Magnitude(max_cube - min_cube)*10.0;
  //
  // Create a volume, specify how many cells in each dim
  //
  vtkm::cont::DataSet dataSet = MakeEmptyVolumeDataset(dims, origin, spacing);

  START_TIMER_BLOCK(splatter)

  //
  // create a field by splatting particles into our volume
  //
  const int num_particles = points;
  std::vector<double>  xdata(num_particles), ydata(num_particles), zdata(num_particles);
  std::vector<float>  hdata(num_particles);
  std::vector<float>  sdata(num_particles);
  std::random_device rd;
  std::mt19937 gen(rd());
  // give the particles real world space positions inside the volume
  std::uniform_real_distribution<> x_dis(origin[0], origin[0] + spacing[0]*fdim[0]);
  std::uniform_real_distribution<> y_dis(origin[1], origin[1] + spacing[1]*fdim[1]);
  std::uniform_real_distribution<> z_dis(origin[2], origin[2] + spacing[2]*fdim[2]);
  // give each point a smoothing length which is 1/5 of the cutoff for gaussian splats
  // we choose a value between 0 and 5 voxels wide for now
  double h_     = spacing[0]*fdim[0]/10.0;
  double norm_  = pow(M_PI, 1.5) * h_*h_*h_;

  std::uniform_real_distribution<>  H(spacing[0]*fdim[0]/10.0, spacing[0]*fdim[0]/10.0);
  std::uniform_real_distribution<>  scale(0.1, 5.0);
  std::generate(xdata.begin(), xdata.end(), [&]{return x_dis(gen);});
  std::generate(ydata.begin(), ydata.end(), [&]{return y_dis(gen);});
  std::generate(zdata.begin(), zdata.end(), [&]{return z_dis(gen); });
  std::generate(hdata.begin(), hdata.end(), [&]{return     H(gen); });
  std::generate(sdata.begin(), sdata.end(), [&]{return scale(gen); });
  //
  floatVec temp =  static_cast<vtkm::FloatDefault>(0.5) * (fdim*spacing);
  xdata[0] = temp[0];
  ydata[0] = temp[1];
  zdata[0] = temp[2];
  hdata[0] = spacing[0]*fdim[0]/10.0;
  sdata[0] = norm_;
  view_origin = origin;

  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> xValues;
  xValues = vtkm::cont::make_ArrayHandle(xdata);
  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> yValues;
  yValues = vtkm::cont::make_ArrayHandle(ydata);
  vtkm::cont::ArrayHandle<vtkm::Float64,VTKM_DEFAULT_STORAGE_TAG> zValues;
  zValues = vtkm::cont::make_ArrayHandle(zdata);
  vtkm::cont::ArrayHandle<vtkm::Float32, VTKM_DEFAULT_STORAGE_TAG> hValues;
  hValues = vtkm::cont::make_ArrayHandle(hdata);
  vtkm::cont::ArrayHandle<vtkm::Float32, VTKM_DEFAULT_STORAGE_TAG> sValues;
  sValues = vtkm::cont::make_ArrayHandle(sdata);

  vtkm::worklet::debug::OutputArrayDebug(xValues, "x Values");
  vtkm::worklet::debug::OutputArrayDebug(yValues, "y Values");
  vtkm::worklet::debug::OutputArrayDebug(zValues, "z Values");
  vtkm::worklet::debug::OutputArrayDebug(hValues, "Radii");
  vtkm::worklet::debug::OutputArrayDebug(sValues, "Scale");

  vtkm::cont::ArrayHandle<vtkm::Float32> fieldArray;

  // we are using variable smoothing lenght particles, so initialize
  // the kernel with default h=1.0
  vtkm::worklet::splatkernels::Gaussian<3> k(1.0);
//  vtkm::worklet::kernels::Spline3rdOrder<3> k(1.0);
//  vtkm::worklet::kernels::GaussianUnitHeight<3> k(1.0);

  vtkm::worklet::KernelSplatterFilterUniformGrid<vtkm::worklet::splatkernels::Gaussian<3>, DeviceAdapter>
    splatter(dims, origin, spacing, dataSet, k);

  splatter.run
    (xValues, yValues, zValues, hValues, sValues, fieldArray);

  END_TIMER_BLOCK(splatter)

  vtkm::worklet::debug::OutputArrayDebug(fieldArray, "fieldArray");

  //
  // add field to the dataset
  //
  dataSet.AddField(vtkm::cont::Field("pointvar", vtkm::cont::Field::ASSOC_POINTS, fieldArray));
  vtkm::cont::CellSetStructured<3> cellSet;
  dataSet.GetCellSet().CopyTo(cellSet);

  //
  // Create isosurface filter
  //
  vtkm::worklet::MarchingCubes<vtkm::Float32,DeviceAdapter> isosurfaceFilter;

  START_TIMER_BLOCK(isosurface)

  //
  // and compute the isosurface
  //
  isosurfaceFilter.Run(isovalue,
                       cellSet,
                       dataSet.GetCoordinateSystem(),
                       fieldArray,
                       verticesArray,
                       normalsArray);

  END_TIMER_BLOCK(isosurface)

  vtkm::worklet::debug::OutputArrayDebug(verticesArray, "verticesArray");

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
  float mc1[3] = {min_cube[0], min_cube[1], min_cube[2]};
  float mc2[3] = {max_cube[0], max_cube[1], max_cube[2]};
  set_viewpoint(view_origin[0],view_origin[1],view_origin[2], view_dist, mc1, mc2);
  GLFWwindow *window = init_glfw(800, 800);
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > vertextype;
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > normaltype;


  std::function<void()> display_function = boost::bind(&displayCall<vertextype, normaltype>,
                                                     verticesArray,
                                                     normalsArray);
  run_graphics_loop(window, display_function);
  return 0;
}
#endif

