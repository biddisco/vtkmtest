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

#include <vtkm/cont/DeviceAdapter.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/DynamicArrayHandle.h>

#if defined (__APPLE__)
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
#endif
//
#include <GLFW/glfw3.h>
//
#include "quaternion.h"

/// Global variables
///
Quaternion qrot;
bool render_enabled = true;
double lastx, lasty;
//
float eye[3] = {0,0,4};
float center[3] = {0,0,0};
float up[3] = {0,1,0};
float zoom=45;
float aspect = 1.0;
float zNear = 1.0;
float zFar = 30.0;
float min_coord[3] = {0,0,0};
float max_coord[3] = {0.0001, 0.0001, 0.0001};

//----------------------------------------------------------------------------
// The cube has opposite corners at (0,0,0) and (1,1,1), which are black and
// white respectively.  The x-axis is the red gradient, the y-axis is the
// green gradient, and the z-axis is the blue gradient.  The cube's position
// and colors are fixed.
namespace Cube {

  const int NUM_VERTICES = 8;
  const int NUM_FACES = 6;

  void draw() {

    GLfloat vertices[NUM_VERTICES][3] = {
      {min_coord[0], min_coord[1], min_coord[2]}, {min_coord[0], min_coord[1], max_coord[2]},
      {min_coord[0], max_coord[1], min_coord[2]}, {min_coord[0], max_coord[1], max_coord[2]},

      {max_coord[0], min_coord[1], min_coord[2]}, {max_coord[0], min_coord[1], max_coord[2]},
      {max_coord[0], max_coord[1], min_coord[2]}, {max_coord[0], max_coord[1], max_coord[2]}};

    GLint faces[NUM_FACES][4] = {
      {1, 5, 7, 3}, {5, 4, 6, 7}, {4, 0, 2, 6},
      {3, 7, 6, 2}, {0, 1, 3, 2}, {0, 4, 5, 1}};

    GLfloat vertexColors[NUM_VERTICES][3] = {
      {1.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 1.0},
      {1.0, 1.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};

    glBegin(GL_QUADS);
    for (int i = 0; i < NUM_FACES; i++) {
      for (int j = 0; j < 4; j++) {
        glColor3fv((GLfloat*)&vertexColors[faces[i][j]]);
        glVertex3iv((GLint*)&vertices[faces[i][j]]);
      }
    }
    glEnd();
  }
}

//----------------------------------------------------------------------------
// Initialize OpenGL parameters, including lighting
//----------------------------------------------------------------------------
void initializeGL()
{
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_SMOOTH);

  float white[] = { 0.8, 0.8, 0.8, 1.0 };
  float black[] = { 0.0, 0.0, 0.0, 1.0 };
  float lightPos[] = { 10.0, 10.0, 10.5, 1.0 };

  glLightfv(GL_LIGHT0, GL_AMBIENT, white);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
  glLightfv(GL_LIGHT0, GL_SPECULAR, black);
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);
}

//----------------------------------------------------------------------------
// Render the computed triangles
//----------------------------------------------------------------------------
void set_viewpoint(float v0, float v1, float v2, float dist, float mins[3], float maxs[3])
{
  center[0] = v0;
  center[1] = v1;
  center[2] = v2;
  eye[0] = center[0];
  eye[1] = center[1];
  eye[2] = center[2] + dist;
  zNear = 0.1;
  zFar  = center[2] + dist*2;
  //
  min_coord[0] = mins[0];
  min_coord[1] = mins[1];
  min_coord[2] = mins[2];
  max_coord[0] = maxs[0];
  max_coord[1] = maxs[1];
  max_coord[2] = maxs[2];
}

//----------------------------------------------------------------------------
// Render the computed triangles
//----------------------------------------------------------------------------
template <typename vertextype, typename normaltype>
void displayCall(vertextype v_array, normaltype n_array)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective (50.0*zoom, aspect, zNear, zFar);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(eye[0],eye[1],eye[2],
            center[0],center[1],center[2],
            up[0],up[1],up[2]);

  glPushMatrix();


  float rotationMatrix[16];
  qrot.getRotMat(rotationMatrix);
  glMultMatrixf(rotationMatrix);

  Cube::draw();

  glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

  glColor3f(1.0f, 1.0f, 0.6f);

  if (render_enabled)
  {
    glBegin(GL_TRIANGLES);
    for (unsigned int i=0; i<v_array.GetPortalConstControl().GetNumberOfValues(); i++)
    {
      vtkm::Vec<vtkm::Float32, 3> curNormal = n_array.GetPortalConstControl().Get(i);
      vtkm::Vec<vtkm::Float32, 3> curVertex = v_array.GetPortalConstControl().Get(i);
      glNormal3f(curNormal[0], curNormal[1], curNormal[2]);
      glVertex3f(curVertex[0], curVertex[1], curVertex[2]);
    }
    glEnd();
  }

//  Cube::draw();

  glPopMatrix();
}

//----------------------------------------------------------------------------
// Resize function
//----------------------------------------------------------------------------
void reshape( GLFWwindow* window, int width, int height )
{
  GLfloat h = (GLfloat) height / (GLfloat) width;
  GLfloat xmax, znear, zfar;

  znear = 5.0f;
  zfar  = 30.0f;
  xmax  = znear * 0.5f;

  glViewport( 0, 0, (GLint) width, (GLint) height );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glFrustum( -xmax, xmax, -xmax*h, xmax*h, znear, zfar );
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();
  glTranslatef( 0.0, 0.0, -20.0 );
}

//----------------------------------------------------------------------------
// Handle mouse button pushes
//----------------------------------------------------------------------------
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (button != GLFW_MOUSE_BUTTON_LEFT)
    return;

  if (action == GLFW_PRESS)
  {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetCursorPos(window, &lastx, &lasty);
  }
  else
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

//----------------------------------------------------------------------------
// Handle mouse move
//----------------------------------------------------------------------------
void cursor_position_callback(GLFWwindow* window, double x, double y)
{
  double dx = x - lastx;
  double dy = y - lasty;
  if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
  {
    Quaternion newRotX;
    newRotX.setEulerAngles(-0.2*dx*M_PI/180.0, 0.0, 0.0);
    qrot.mul(newRotX);

    Quaternion newRotY;
    newRotY.setEulerAngles(0.0, 0.0, -0.2*dy*M_PI/180.0);
    qrot.mul(newRotY);
  }
  lastx = x;
  lasty = y;
}

//----------------------------------------------------------------------------
// Handle scroll events
//----------------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoff, double yoff)
{
  zoom += xoff/10.0;
}

//----------------------------------------------------------------------------
// Handle keyboard
//----------------------------------------------------------------------------
void key( GLFWwindow* window, int k, int s, int action, int mods )
{
  if( action != GLFW_PRESS ) return;

  switch (k) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GL_TRUE);
      break;
    default:
      return;
  }
}

//----------------------------------------------------------------------------
// create a glfw window, must be run on main OS thread AFAICT
//----------------------------------------------------------------------------
GLFWwindow *init_glfw(int w, int h)
{
  GLFWwindow* window;

  if( !glfwInit() )
  {
    fprintf( stderr, "Failed to initialize GLFW\n" );
    exit( EXIT_FAILURE );
  }

  glfwWindowHint(GLFW_DEPTH_BITS, 16);

  window = glfwCreateWindow( w, h, "vtkm-test", NULL, NULL );
  if (!window)
  {
    fprintf( stderr, "Failed to open GLFW window\n" );
    glfwTerminate();
    exit( EXIT_FAILURE );
  }

  // Set callback functions
  glfwSetFramebufferSizeCallback(window, reshape);
  glfwSetKeyCallback(window, key);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_position_callback);
  glfwSetScrollCallback(window, scroll_callback);

  glfwMakeContextCurrent(window);
  glfwSwapInterval( 1 );

  glfwGetFramebufferSize(window, &w, &h);
  reshape(window, w, h);
  initializeGL();

  return window;

}

//----------------------------------------------------------------------------
// interactive render loop, must be run on main OS thread AFAICT
//----------------------------------------------------------------------------
void run_graphics_loop(GLFWwindow* window, const std::function<void()> &displayFunction)
{
  // Main loop
  while( !glfwWindowShouldClose(window) )
  {
    // Draw scene
    displayFunction();

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Terminate GLFW
  glfwTerminate();
}

