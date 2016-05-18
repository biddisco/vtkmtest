// tell vtkm that we want to use TBB backend
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_TBB

// if we have FreeGlut, change event handler
// #ifdef VTKM_USE_FREEGLUT
#undef VTKM_USE_FREEGLUT

#define TBB_TIMING

#include "isosurface.cpp"

