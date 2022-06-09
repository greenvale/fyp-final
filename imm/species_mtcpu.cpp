#include <iostream>
#include <cmath>
#include <fstream>

#include <omp.h>

#include "species.h"

void Species::push_mthread(const bool& accelerate)
{
  // set avgmom vector to zeros
  for (int i = 0; i < nx + 1; ++i)
  {
    avgmom[i] = 0.0;
  }

  omp_set_num_threads(target_num_threads);

  #pragma omp parallel
  {
    // initialise run-time variables
    bool    sub_flag;
    int     sub_index;
    int     method_flag;
    double  subt;
    double  subpos;
    double  subvel;
    int     subcell;
    double  subpos_new;
    double  subvel_new;
    int     subcell_new;
    double  accel;
    double  dsubt;
    double  dsubpos;
    double  dsubvel;
    int     dsubcell;
    double  dsubpos_lh;
    double  dsubpos_rh;
    double  shape_lhface;
    double  shape_rhface;

    int p;
    int id;
    int num_threads;
    int num_threads_t;

    // initialise local arrays
    double* avgmom_t = new double[nx + 1];
    double* elec_t   = new double[2 * (nx + 1)];

    for (int i = 0; i < nx + 1; ++i)
    {
      // set avgmom to zero for accumulation
      avgmom_t[i] = 0.0;

      // copy electric field for current and future timesteps
      elec_t[2*i + 0]   = elec[2*i + 0];
      elec_t[2*i + 1]   = elec[2*i + 1];
    }

    // get thread id and number of threads
    id = omp_get_thread_num();
    num_threads_t = omp_get_num_threads();

    // store the total number of threads in global variable
    #pragma omp single
    {
      num_threads = num_threads_t;
      std::cout << "Num threads: " << num_threads << std::endl;
    }

    // loop through particles with static scheduling
    #pragma omp for schedule(static, 10)
    for (p = 0; p < np; ++p)
    {
      sub_flag  = false;
      sub_index = -1;
      subt      = 0.0;

      // transfer current pos, vel and cell to sub-cycle variables
      subpos  = pos[p];
      subvel  = vel[p];
      subcell = (int) floor(pos[p] / dx);

      // sub-cycle loop
      while(sub_flag == false)
      {
        sub_index++;

        // LH and RH bounds on how far particle can move in this step
        dsubpos_lh = ((double) subcell * dx) - subpos;
        dsubpos_rh = dsubpos_lh + dx;

        /*
        method flag: indicates which method used to push particle
          => 0 : particle pushed within cell, using Picard to solve for dsubpos
          => 1 : particle pushed and forced to land on cell face, using Picard to solve for dsubt
          => 2 : particle pushed between cell faces using direct method
        */

        // check if accelerate block is being used and is needed; note method not valid for first sub-cycle iteration
        if ((accelerate == true) && (dsubcell != 0) && (sub_index > 0))
        {
          // direct push
          // if direct solution possible, method flag returns 2, else returns 0 meaning adaptive method needed
          direct_push(
            method_flag,    accel,
            dsubt,          dsubvel,
            dsubpos,        dsubcell,
            shape_lhface,   shape_rhface,
            subt,           subpos,
            subvel,         subcell,
            dsubpos_lh,     dsubpos_rh,
            elec_t
          );
        }
        else
        {
          // use adaptive push if accelerate block not used
          method_flag = 0;
        }

        // adaptive push
        if (method_flag == 0)
        {
          adaptive_push(
            method_flag,    accel,
            dsubt,          dsubvel,
            dsubpos,        dsubcell,
            shape_lhface,   shape_rhface,
            subt,           subpos,
            subvel,         subcell,
            dsubpos_lh,     dsubpos_rh,
            elec_t
          );
        }

        // check if subcycle finished
        if (subt + dsubt >= dt)
          sub_flag = true;

        // accumulate avgmom for this substep
        accumulate_avgmom(avgmom_t, dsubvel, shape_lhface, shape_rhface, subvel, subcell, dsubt);

        // step particle values
        substep(
          subpos_new,
          subvel_new,     subcell_new,
          subt,           subpos,
          subvel,         subcell,
          dsubt,          dsubpos,
          dsubvel,        dsubcell
        );
      }

      // transfer sub-cycle new values to global new values at end of particle evolution
      pos_new[p]  = subpos_new;
      vel_new[p]  = subvel_new;

      #pragma omp critical
      {
        //std::cout << "Thread " << id << ": " << "p = " << p << ", pos new: " << pos_new[p] << "\t";// << std::endl;
      }
    }

    // reduce avgmom components between threads
    #pragma omp critical
    {
      for (int i = 0; i < nx + 1; ++i)
      {
        avgmom[i] += avgmom_t[i];
      }
    }

    delete[] avgmom_t, elec_t;
  }

  // periodic boundary conditions
  avgmom[0] += avgmom[nx];
  avgmom[nx] = avgmom[0];
}
