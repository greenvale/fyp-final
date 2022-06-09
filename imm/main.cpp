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
  debug_config1.nspecies = 1;
  debug_config1.nt = 1;
  debug_config1.np = 256*3;
  debug_config1.nx = 10;
  debug_config1.nv = {50};
  debug_config1.lx = 1.0;
  debug_config1.dt = pow(10.0, -6);
  debug_config1.species_name = {"electron", "ion"};
  debug_config1.species_dens_avg = { 1.0, 1.0 };
  debug_config1.species_dens_perturb = { 1.0, 0.0 };
  debug_config1.species_vel_profile = {"boltzmann"};
  debug_config1.species_vel_avg = {0.0};
  debug_config1.species_vel_range = {0.1};
  debug_config1.species_charge = { -1.0 };
  debug_config1.species_mass = { 1.0 };
  debug_config1.species_T = { 1.0 };

  Config landau1;
  landau1.nspecies = 2;
  landau1.nt = (int) 100;//(16 / (8.0 * 0.01));
  landau1.np = (int) 10000;
  landau1.nx = 30;
  landau1.nv = { 50 , 50 };
  landau1.lx = 4.0 * M_PI;
  landau1.dt = 0.2; //1.0 * 0.01 * 1.467;
  landau1.species_name = { "electron", "ion" };
  landau1.species_dens_avg = { 1.0, 1.0 };
  landau1.species_dens_perturb = { 0.1, 0.0 };
  landau1.species_dens_perturb_profile = { "cos", "" };
  landau1.species_vel_profile = { "boltzmann", "boltzmann" };
  landau1.species_vel_avg = { 0.0, 0.0 };
  landau1.species_vel_range = { 8.0 , 0.01 };
  landau1.species_vel_perturb = { 0.1, 0.0 };
  landau1.species_vel_perturb_profile = { "", "" };
  landau1.species_charge = { -1.0 , 1.0 };
  landau1.species_mass = { 1.0 , 2000.0 };
  landau1.species_T = { 1.0 , 1.0 };

  Config twobeam1;
  twobeam1.nspecies = 2;
  twobeam1.nt = 1000;
  twobeam1.np = 256*40;
  twobeam1.nx = 32;
  twobeam1.nv = { 50 };
  twobeam1.lx = 1.0;
  twobeam1.dt = 0.01;
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
  shockwave1.nt = 1000;
  shockwave1.np = 256*40;
  shockwave1.nx = 144;
  shockwave1.nv = {50, 5000};
  shockwave1.lx = 144.0;
  shockwave1.dt = 0.4;
  shockwave1.species_name = {"electron", "ion"};
  shockwave1.species_dens_avg = {1.0, 1.0};
  shockwave1.species_dens_perturb = {0.1, 0.4};
  shockwave1.species_dens_perturb_profile = {"sin", "sin"};
  shockwave1.species_vel_profile = {"boltzmann", "two_beams"};
  shockwave1.species_vel_avg = { 0.0, 0.0223};
  shockwave1.species_vel_range = { 8.0, 0.02 };
  shockwave1.species_vel_perturb = { 0.0, 0.01 };
  shockwave1.species_vel_perturb_profile = {"sin", "sin"};
  shockwave1.species_charge = { -1.0, 1.0 };
  shockwave1.species_mass = { 1.0, 2000.0 };
  shockwave1.species_T = { 1.0, 0.0002 };

  std::vector<Config> configs = { debug_config1, landau1, twobeam1, shockwave1 };

  //==================================================================

  bool write_posvel = true;
  int skip = 1;
  int test_config_index = 1;

  // use direct method accelerator?
  bool accelerate = true;

  // gpu kernel path
  configs[test_config_index].push_kernel_path =  "./push_kernel_1.cl";

  // create plasma object and associated species objects
  configs[test_config_index].create_plasma("cpu_sthread");

  std::vector<double>runtime_data = configs[test_config_index].run_plasma(accelerate, skip, write_posvel);

  std::cout << "PROGRAM COMPLETE" << std::endl;
  std::cout << "Executed in " << runtime_data[0] << " seconds" << std::endl;

  return 0;
}
