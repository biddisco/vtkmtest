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
#include <vtkm/worklet/DispatcherMapField.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/filter/MarchingCubes.h>
#include <vtkm/Pair.h>

// set a global typedef that is the chosen device tag
typedef VTKM_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

#ifdef _WIN32
# include "windows.h""
#endif

#include <boost/bind.hpp>

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
// TangleField (from UnitTestIsosurfaceUniformGrid.cxx)
//----------------------------------------------------------------------------
class TangleField : public vtkm::worklet::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn<IdType> vertexId, FieldOut<Scalar> v);
  typedef void ExecutionSignature(_1, _2);
  typedef _1 InputDomain;

  const vtkm::Id xdim, ydim, zdim;
  const float xmin, ymin, zmin, xmax, ymax, zmax;
  const vtkm::Id cellsPerLayer;

  VTKM_CONT_EXPORT
    TangleField(const vtkm::Id3 dims, const float mins[3], const float maxs[3]) : xdim(dims[0]), ydim(dims[1]), zdim(dims[2]),
    xmin(mins[0]), ymin(mins[1]), zmin(mins[2]), xmax(maxs[0]), ymax(maxs[1]), zmax(maxs[2]), cellsPerLayer((xdim)* (ydim)) { };

  VTKM_EXEC_EXPORT
    void operator()(const vtkm::Id &vertexId, vtkm::Float32 &v) const
  {
    const vtkm::Id x = vertexId % (xdim);
    const vtkm::Id y = (vertexId / (xdim)) % (ydim);
    const vtkm::Id z = vertexId / cellsPerLayer;

    const float fx = static_cast<float>(x) / static_cast<float>(xdim - 1);
    const float fy = static_cast<float>(y) / static_cast<float>(xdim - 1);
    const float fz = static_cast<float>(z) / static_cast<float>(xdim - 1);

    const vtkm::Float32 xx = 3.0f*(xmin + (xmax - xmin)*(fx));
    const vtkm::Float32 yy = 3.0f*(ymin + (ymax - ymin)*(fy));
    const vtkm::Float32 zz = 3.0f*(zmin + (zmax - zmin)*(fz));

    v = (xx*xx*xx*xx - 5.0f*xx*xx + yy*yy*yy*yy - 5.0f*yy*yy + zz*zz*zz*zz - 5.0f*zz*zz + 11.8f) * 0.2f + 0.5f;
  }
};

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
  const floatVec origin(0.0, 0.0, 0.0);
  const floatVec spacing(1.0, 1.0, 1.0);

  // min and max for tangle field
  float mins[3] = { -1.0f, -1.0f, -1.0f };
  float maxs[3] = { 1.0f, 1.0f, 1.0f };

  //
  // Create a volume, specify how many cells in each dim
  //
  vtkm::cont::DataSet dataSet = MakeEmptyVolumeDataset(dims, origin, spacing);

  //
  // create a field for isosurfacing
  //
  vtkm::cont::ArrayHandle<FieldType> fieldArray;
  if (fileName != 0) {
    // Read the field from a file
    std::vector<FieldType> field;
    std::fstream in(fileName, std::ios::in);
    std::copy(std::istream_iterator<FieldType>(in), std::istream_iterator<FieldType>(), std::back_inserter(field));
    std::copy(field.begin(), field.end(), std::ostream_iterator<FieldType>(std::cout, " "));  std::cout << std::endl;
    fieldArray = vtkm::cont::make_ArrayHandle(&field[0], field.size());
  }
  else {
    START_TIMER_BLOCK(tangle)

    // Generate tangle field, N = num vertices for field evaluaition
    vtkm::cont::ArrayHandleCounting<vtkm::Id> vertexCountImplicitArray(0, 1, vdims[0] * vdims[1] * vdims[2]);
    vtkm::worklet::DispatcherMapField<TangleField> tangleFieldDispatcher(TangleField(vdims, mins, maxs));
    tangleFieldDispatcher.Invoke(vertexCountImplicitArray, fieldArray);

  }

  //
  // add field to the dataset
  //
  dataSet.AddField(vtkm::cont::Field("nodevar", vtkm::cont::Field::ASSOC_POINTS, fieldArray));
  vtkm::cont::CellSetStructured<3> cellSet;
  dataSet.GetCellSet().CopyTo(cellSet);

  //
  // Create isosurface filter, use cell dimensions to initialize
  //
  vtkm::worklet::MarchingCubes<vtkm::Float32,DeviceAdapter> isosurfaceFilter;

  START_TIMER_BLOCK(isosurface)

  //
  // and compute the isosurface
  //
  isosurfaceFilter.Run(isovalue,
                       cellSet,
                       dataSet.GetCoordinateSystem(),
                       scalarsArray,
                       verticesArray,
                       normalsArray);

  END_TIMER_BLOCK(isosurface)
  
  return 0;
}

#if VTKM_DEVICE_ADAPTER != VTKM_DEVICE_ADAPTER_HPX
//----------------------------------------------------------------------------
// standard int main() entry point
//----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  // setup all the pipeline stuff and wait till it's done
  init_pipeline(argc, argv);
  GLFWwindow *window = init_glfw(800,800);
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > vertextype;
  typedef vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32,3> > normaltype;

  std::function<void()> display_function = boost::bind(&displayCall<vertextype, normaltype>,
                                                     verticesArray,
                                                     normalsArray);

  run_graphics_loop(window, display_function);
  return 0;
}

#else
//----------------------------------------------------------------------------
// an int main for HPX which runs filters on hpx threads, GUI on OS thread
//----------------------------------------------------------------------------
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

#endif

