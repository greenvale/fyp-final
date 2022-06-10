#include <vector>
#include <iostream>
#include <fstream>

#include <omp.h>
#include <CL/cl.hpp>

#include "species.h"
#include "plasma.h"

class Config
{
public:
  std::string accumulator_kernel_path = "./accumulator.cl";
  std::string push_kernel_path;
  std::string accumulator_kernel_name = "accumulate_moments_gpu";
  std::string push_kernel_name = "push_gpu";
  int nspecies, np, nx;
  std::vector<int> nv;
  double lx, dt, time_total;
  std::vector<double> species_dens_avg, species_dens_perturb, species_vel_avg,
    species_vel_range, species_vel_perturb, species_charge, species_mass, species_T;
  std::vector<std::string> species_vel_profile, species_dens_perturb_profile, species_vel_perturb_profile, species_name;
  Plasma* plasma_ptr;

  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Kernel accumulator_kernel;
  cl::Kernel push_kernel;

  // empty constructor - parameters are set outside of object
  Config()
  {

  }

  // create plasma object and associated species objects
  void create_plasma(std::string device_type)
  {
    // if device type is GPU then setup opencl params
    if (device_type == "gpu")
    {
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      // use first platform
      platform = platforms[0];
      std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
      // get vector of devices
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      if (devices.size() == 0)
      {
        std::cout << "No devices found on this platform!" << std::endl;
      }
      // use first device
      device = devices[0];
      std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

      // initialise context for device
      context = cl::Context({ device });

      // initialise queue for device
      queue = cl::CommandQueue(context, device);

      // create update moments kernel
      accumulator_kernel = create_kernel(accumulator_kernel_path, accumulator_kernel_name);
      // create evolve kernel
      push_kernel = create_kernel(push_kernel_path, push_kernel_name);
      std::cout << "Kernels have been compiled" << std::endl;
    }

    double dx = lx / nx;

    plasma_ptr = new Plasma(device_type, nspecies, nx, lx);

    // x face and x centre values used in calculation of distributions
    double* x_face = new double[nx + 1];
    double* x_centre = new double[nx];

    // calculate the x values of each cell face and centre
    x_face[0] = 0.0;
    for (int i = 0; i < nx; ++i)
    {
      x_face[i+1] = (i+1)*dx;
      x_centre[i] = 0.5 * (x_face[i] + x_face[i+1]);
    }

    for (int alfa = 0; alfa < nspecies; ++alfa)
    {
      double* v_centre = new double[nv[alfa]];
      double* fx = new double[nx];
      double* fv = new double[nx * nv[alfa]];
      double v_min = species_vel_avg[alfa] - 0.5 * species_vel_range[alfa];
      double v_max = species_vel_avg[alfa] + 0.5 * species_vel_range[alfa];
      double perturb_factor;

      // calculate v_centre for this species
      for (int i = 0; i < nv[alfa]; ++i)
      {
        v_centre[i] = v_min + i*(species_vel_range[alfa])/nv[alfa];
      }

        // calculate fx for this species
        for (int i = 0; i < nx; i++) {
            if (species_dens_perturb_profile[alfa] == "sin")
            {
                perturb_factor = sin(2.0 * M_PI * x_centre[i] / lx);
            }
            else if (species_dens_perturb_profile[alfa] ==  "cos")
            {
                perturb_factor = cos(2.0 * M_PI * x_centre[i] / lx);
            }
            else
            {
                perturb_factor = 0.0;
            }
            fx[i] = species_dens_avg[alfa] + (species_dens_perturb[alfa] * perturb_factor);
        }

      // calculate fv for this species
      if (species_vel_profile[alfa] == "boltzmann")
      {
        for (int i = 0; i < nx; ++i)
        {
          for (int j = 0; j < nv[alfa]; j++)
          {
      			fv[j + i*nv[alfa]] = (species_dens_avg[alfa] / sqrt(2.0 * M_PI * constants::Kb * species_T[alfa] / species_mass[alfa])) *
              exp(-species_mass[alfa] * (v_centre[j] - species_vel_avg[alfa])*(v_centre[j] - species_vel_avg[alfa])
              / (2.0 * constants::Kb * species_T[alfa]));

            //fv[j + i*nv[alfa]] = (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * (v_centre[j] - species_vel_avg[alfa]) * (v_centre[j] - species_vel_avg[alfa]));
      		}
        }
      }
      else if (species_vel_profile[alfa] == "two_beams")
      {
        for (int i = 0; i < nx; ++i)
        {
          for (int j = 0; j < nv[alfa]; j++)
          {
            if ((j == 0) || (j == nv[alfa] - 1))
            {
              fv[j + i*nv[alfa]] = 1.0;
            }
            else
            {
              fv[j + i*nv[alfa]] = 0.0;
            }
          }
        }
      }

      std::cout << "Species: " << species_name[alfa] << ": perturb: " << species_vel_perturb[alfa] << std::endl;

      // create species object in plasma
      plasma_ptr->add_species(context, queue, accumulator_kernel, push_kernel,
        np, nv[alfa], v_centre, species_name[alfa], species_charge[alfa], species_mass[alfa],
        v_min, v_max, fx, fv, species_dens_avg[alfa], species_vel_perturb[alfa], species_vel_perturb_profile[alfa]);

      delete[] v_centre, fx, fv;
    }

    delete[] x_face, x_centre;

    plasma_ptr->init_lo();
  }

  // run plasma evolution loop
  std::vector<double> run_plasma(bool accelerate, int skip, bool write_posvel)
  {
    //plasma_ptr->print_vals(0, skip, write_posvel);

    // Clear dt file
    std::ofstream dt_file("./output/dt.txt", std::ofstream::out | std::ofstream::trunc);

    double start_time = omp_get_wtime();
    double dt_evolve_target;
    double dt_evolve;
    double sim_time = 0.0;

    int k = -1;

    bool finish_flag = false;

    while (finish_flag == false)
    {
        if (sim_time + dt >= time_total)
        {
            dt_evolve_target = time_total - sim_time;
        }
        else
        {
            // set target evolve dt as prescribed dt
            dt_evolve_target = dt;
        }

        // prevent round off error dt
        if (dt_evolve_target > 0.00001)
        {
            k++;
            std::cout << "=========== SOLVING TIMESTEP " << k+1 << " ===========" << std::endl;
            dt_evolve = plasma_ptr->evolve(accelerate, dt_evolve_target);
            std::cout << "Finished outer Picard loop" << std::endl;

            // print future (k+1/2 or k+1) values before forward stepping system
            plasma_ptr->print_vals(k+1, skip, write_posvel);

            // forward step system
            plasma_ptr->step();

            // accumulate actual dt
            sim_time += dt_evolve;

            plasma_ptr->print_methods_tracker(0);

            // Print dt
            std::ofstream dt_file("./output/dt.txt", std::ofstream::out | std::ofstream::app);
            for (int i = 0; i < nx + 1; i++)
            {
                dt_file.width(15);
                dt_file << dt_evolve << "\t";
            }
            dt_file.close();
        }
        else
        {
            std::cout << "Prevented round off time: " << dt_evolve_target << std::endl;
            finish_flag = true;
        }

        if (sim_time >= time_total)
        {
            finish_flag = true;
        }
    }

    double end_time = omp_get_wtime();

    std::vector<double> runtime_data = {end_time - start_time, sim_time};

    return runtime_data;
  }

  // create kernel object from program source path
  cl::Kernel create_kernel(const std::string& kernel_path, const std::string& kernel_name)
  {
    std::ifstream file(kernel_path);
    std::string input;
    std::string kernel_source;
    while (file >> input) {
        input.append(" ");
        kernel_source.append(input);
    }

    // initialise program sources
    cl::Program::Sources sources;
    sources.push_back({ kernel_source.c_str(), kernel_source.length() });

    // build program
    cl::Program program(context, sources);

    // return any build errors
    if (program.build({ device }) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
    }

    // create kernel object
    return cl::Kernel(program, kernel_name.c_str());
  }

};
