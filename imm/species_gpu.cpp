#include <iostream>
#include <cmath>
#include <fstream>

#include <CL/cl.hpp>

#include "species.h"

void Species::push_gpu(const bool& accelerate)
{
  int kernel_reduce_method = 1;

  // create scheme params variable
  params scheme_params;
  scheme_params.nx = nx;
  scheme_params.lx = lx;
  scheme_params.dt = dt;
  scheme_params.mass = mass;
  scheme_params.charge = charge;
  scheme_params.weight = weight;
  scheme_params.accelerate = (int) accelerate;
  scheme_params.pic_tol = 0.0001;

  // get host float arrays of pos, vel and elec
  for (int p = 0; p < np; ++p)
  {
    pos_h[p] = (float) pos[p];
    vel_h[p] = (float) vel[p];
  }
  for (int i = 0; i < 2 * (nx + 1); ++i)
  {
    elec_h[i] = (float) elec[i];
  }

  // set avgmom expanded vector to zeros
  for (int i = 0; i < (nx + 1) * num_workgroups; ++i)
  {
    avgmom_expanded_h[i] = 0.0;
  }

  // write values to buffers
  queue.enqueueWriteBuffer(elec_d, CL_TRUE, 0, sizeof(float) * 2 * (nx + 1), elec_h);
  queue.enqueueWriteBuffer(pos_d, CL_TRUE, 0, sizeof(float) * np, pos_h);
  queue.enqueueWriteBuffer(vel_d, CL_TRUE, 0, sizeof(float) * np, vel_h);
  queue.enqueueWriteBuffer(avgmom_expanded_d, CL_TRUE, 0, sizeof(float) * (nx + 1) * num_workgroups, avgmom_expanded_h);

  // set kernel arguments
  // read:
  push_kernel.setArg(0, scheme_params);
  push_kernel.setArg(1, elec_d);
  push_kernel.setArg(2, pos_d);
  push_kernel.setArg(3, vel_d);
  // read/write:
  push_kernel.setArg(4, avgmom_expanded_d);
  // write:
  push_kernel.setArg(5, pos_new_d);
  push_kernel.setArg(6, vel_new_d);
  // local buffers for avgmom
  if (kernel_reduce_method == 1)
  {
    push_kernel.setArg(7, (nx + 1) * sizeof(float), NULL);
    push_kernel.setArg(8, (nx + 1) * workgroup_size * sizeof(float), NULL);
  }
  else if (kernel_reduce_method == 2)
  {
    push_kernel.setArg(7, (nx) * sizeof(float), NULL);
    push_kernel.setArg(8, (nx) * sizeof(float), NULL);
    push_kernel.setArg(9, workgroup_size * sizeof(int), NULL);
  }
  // enqueue kernel
  queue.enqueueNDRangeKernel(push_kernel, cl::NullRange, cl::NDRange(np), cl::NDRange(workgroup_size));

  // block host code until kernels have finished executing
  queue.finish();

  // read results from device
  queue.enqueueReadBuffer(pos_new_d, CL_TRUE, 0, sizeof(float) * np, pos_new_h);
  queue.enqueueReadBuffer(vel_new_d, CL_TRUE, 0, sizeof(float) * np, vel_new_h);
  queue.enqueueReadBuffer(avgmom_expanded_d, CL_TRUE, 0, sizeof(float) * (nx + 1) * num_workgroups, avgmom_expanded_h);

  // convert host float arrays to double arrays for pos, vel
  for (int p = 0; p < np; ++p)
  {
    pos_new[p] = (double) pos_new_h[p];
    vel_new[p] = (double) vel_new_h[p];
  }

  // reduce host float avgmom expanded to double avgmom
  for (int i = 0; i < nx + 1; ++i)
  {
    avgmom[i] = 0.0;
    for (int j = 0; j < num_workgroups; ++j)
    {
      avgmom[i] += (double) avgmom_expanded_h[j*(nx + 1) + i];
    }
  }

  // periodic boundary conditions
  avgmom[0] += avgmom[nx];
  avgmom[nx] = avgmom[0];

  // FOR TESTING
  /*
  double total = 0.0;
  for (int i = 0; i < nx + 1; ++i)
  {
    std::cout << avgmom[i] << std::endl;
    total += avgmom[i];
  }
  std::cout << "TOTAL: " << total << "; should be " << np << std::endl;
  while (true)
  {

  }
  */
}
