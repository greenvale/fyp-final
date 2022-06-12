#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include <omp.h>
#include <CL/cl.hpp>

//#include "species.h"
//#include "plasma.h"
#include "config.cpp"

int main()
{

  //==================================================================
  // config objects for different benchmark tests

  // electron species; landau damping
  Config debug_config1;
  debug_config1.nspecies = 2;
  debug_config1.np = (int) 10000;
  debug_config1.nx = 30;
  debug_config1.nv = { 50 , 50 };
  debug_config1.lx = 4.0 * M_PI;
  debug_config1.dt = 0.1; //1.0 * 0.01 * 1.467;
  debug_config1.time_total = 1 * debug_config1.dt;
  debug_config1.species_name = { "electron", "ion" };
  debug_config1.species_dens_avg = { 1.0, 1.0 };
  debug_config1.species_dens_perturb = { 0.1, 0.0 };
  debug_config1.species_dens_perturb_profile = { "cos", "" };
  debug_config1.species_vel_profile = { "boltzmann", "two_beams" };
  debug_config1.species_vel_avg = { 0.0, 0.0 };
  debug_config1.species_vel_range = { 8.0 , 0.0 };
  debug_config1.species_vel_perturb = { 0.0, 0.0 };
  debug_config1.species_vel_perturb_profile = { "", "" };
  debug_config1.species_charge = { -1.0 , 1.0 };
  debug_config1.species_mass = { 1.0 , 2000.0 };
  debug_config1.species_T = { 1.0 , 1.0 };

  Config landau1;
  landau1.nspecies = 1;
  landau1.np = (int) 10000;
  landau1.nx = 30;
  landau1.nv = { 50 , 50 };
  landau1.lx = 4.0 * M_PI;
  landau1.dt = 1.0; //1.0 * 0.01 * 1.467;
  landau1.time_total = 2000;
  landau1.species_name = { "electron", "ion" };
  landau1.species_dens_avg = { 1.0, 1.0 };
  landau1.species_dens_perturb = { 0.1, 0.0 };
  landau1.species_dens_perturb_profile = { "cos", "" };
  landau1.species_vel_profile = { "boltzmann", "two_beams" };
  landau1.species_vel_avg = { 0.0, 0.0 };
  landau1.species_vel_range = { 8.0 , 0.0 };
  landau1.species_vel_perturb = { 0.0, 0.0 };
  landau1.species_vel_perturb_profile = { "", "" };
  landau1.species_charge = { -1.0 , 1.0 };
  landau1.species_mass = { 1.0 , 2000.0 };
  landau1.species_T = { 1.0 , 1.0 };

  Config twobeam1;
  twobeam1.nspecies = 1;
  twobeam1.np = 10000;
  twobeam1.nx = 30;
  twobeam1.nv = { 50 };
  twobeam1.lx = 1.0;
  twobeam1.dt = 1.0;
  twobeam1.time_total = 30.0;
  twobeam1.species_name = {"electron", "ion"};
  twobeam1.species_dens_avg = { 1.0, 1.0 };
  twobeam1.species_dens_perturb = { 0.1, 0.0 };
  twobeam1.species_dens_perturb_profile = {"cos", ""};
  twobeam1.species_vel_profile = {"two_beams", "boltzmann"};
  twobeam1.species_vel_avg = { 0.0, 0.0 };
  twobeam1.species_vel_range = { 0.2, 0.0 };
  twobeam1.species_vel_perturb = { 0.0, 0.0 };
  twobeam1.species_vel_perturb_profile = {"cos", ""};
  twobeam1.species_charge = { -1.0, 1.0 };
  twobeam1.species_mass = { 1.0, 2000.0 };
  twobeam1.species_T = { 1.0, 1.0 };

  Config shockwave1;
  shockwave1.nspecies = 2;
  shockwave1.np = 100000;
  shockwave1.nx = 30;
  shockwave1.nv = {50, 5000};
  shockwave1.lx = 30.0;
  shockwave1.dt = 1.0;
  shockwave1.time_total = 64;
  shockwave1.species_name = {"electron", "ion"};
  shockwave1.species_dens_avg = {1.0, 1.0};
  shockwave1.species_dens_perturb = {0.1, 0.4};
  shockwave1.species_dens_perturb_profile = {"sin", "sin"};
  shockwave1.species_vel_profile = {"boltzmann", "boltzmann"};
  shockwave1.species_vel_avg = { 0.0, 0.0223};
  shockwave1.species_vel_range = { 8.0, 0.2 };
  shockwave1.species_vel_perturb = { 0.0, 0.1 };
  shockwave1.species_vel_perturb_profile = {"sin", "sin"};
  shockwave1.species_charge = { -1.0, 1.0 };
  shockwave1.species_mass = { 1.0, 2000.0 };
  shockwave1.species_T = { 1.0, 0.0002 };

  std::vector<Config> configs = { debug_config1, landau1, twobeam1, shockwave1 };

  //==================================================================

  bool write_posvel = false;
  int skip = 1;
  int test_config_index = 1;

  // use direct method/fixed method accelerator?
  int accelerate = 0;

  // gpu kernel path
  configs[test_config_index].push_kernel_path =  "./test_kernel.cl";

  // create plasma object and associated species objects
  configs[test_config_index].create_plasma("cpu_sthread");

  std::vector<double>runtime_data = configs[test_config_index].run_plasma(accelerate, skip, write_posvel);

  std::cout << "PROGRAM COMPLETE" << std::endl;
  std::cout << "Executed in " << runtime_data[0] << " seconds" << std::endl;
  std::cout << "Simulation time: " << runtime_data[1] << " seconds" << std::endl;

  return 0;
}
