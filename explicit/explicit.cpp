#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

namespace constants
{
  const double e0 = 8.85418 * pow(10.0, -12);
	const double Kb = 1.3806 * pow(10.0, -23);
	const double T0 = 1.0;

	const double electron_mass = 9.1094 * pow(10.0, -31);
	const double electron_charge = -1.6022 * pow(10.0, -19);
  const double ion_mass = 9.1094 * pow(10.0, -31);
	const double ion_charge = 1.6022 * pow(10.0, -19);
}

struct TestConfig
{
  int ncells;
  int nt;
  double lx;
  double dt;
  int nspecies;
  int np;
  std::vector<int> nv;
  std::vector<std::string> names;
  std::vector<std::string> profiles;
  std::vector<std::string> perturb_profiles;
  std::vector<std::vector<double>> vel_ranges;
  std::vector<double> init_avgdens;
  std::vector<double> init_avgvels;
  std::vector<double> dens_perturb;
  std::vector<double> species_charges;
  std::vector<double> species_masses;
  std::vector<double> species_T;
};

// ACUMULATE MOMENTS : DENSITY & MOMENTUM
void accumulateMoments(
  const int& ncells,
  const int& nspecies,
  const int& np,
  double* pos,
  double* vel,
  int* cell,
  double* weight,
  double* dens,
  double* mom,
  double dx
)
{
  double shapeLH;
  double shapeRH;

  // set density and momentum vectors to zero
  for (int a = 0; a < nspecies; ++a)
  {
    for (int i = 0; i < ncells + 1; ++i)
    {
      // set moments to zero
      dens[a*(ncells + 1) + i] = 0.0;
      mom[a*(ncells + 1)+ i] = 0.0;
    }
  }

  //loop through species
  for (int a = 0; a < nspecies; ++a)
  {
    // loop through particles
    for (int p = 0; p < np; p++)
    {
      shapeLH = 1.0 - fabs(((double) cell[a*np + p]*dx) - pos[a*np + p])/dx;
      shapeRH = 1.0 - fabs(((double) (cell[a*np + p] + 1)*dx) - pos[a*np + p])/dx;

      dens[a*(ncells + 1) + cell[a*np + p]] += shapeLH * weight[a];
      dens[a*(ncells + 1) + cell[a*np + p] + 1] += shapeRH * weight[a];

      mom[a*(ncells + 1) + cell[a*np + p]] += shapeLH * weight[a] * vel[a*np +p];
      mom[a*(ncells + 1) + cell[a*np + p] + 1] += shapeRH * weight[a] * vel[a*np + p];
    }

    // apply periodic boundary condition to domain boundaries
    dens[a*(ncells + 1) + 0] += dens[a*(ncells + 1) + (ncells)];
    mom[a*(ncells + 1) + 0] += mom[a*(ncells + 1) + (ncells)];
    dens[a*(ncells + 1) + (ncells)] = dens[a*(ncells + 1) + 0];
    mom[a*(ncells + 1) + (ncells)] = mom[a*(ncells + 1) + 0];
  }
}

// SOLVE ELECTRIC FIELD USING EXPLICIT SCHEME
void solveField(
  const int& ncells,
  const int& nspecies,
  double* elec,
  double* elec_new,
  double* mom,
  double* charge,
  const double& dt
)
{
  double mom_term;
  double mom_term_sum = 0.0;

  for (int i = 0; i < ncells; ++i)
  {
    mom_term = 0.0;
    for (int a = 0; a < nspecies; ++a)
    {
      mom_term += charge[a] * mom[a*(ncells + 1) + i];
    }
    mom_term_sum += mom_term;
    elec_new[i] = elec[i] - (dt / constants::e0) * mom_term;
  }
  mom_term_sum /= ncells;
  for (int i = 0; i < ncells; ++i)
  {
    elec_new[i] += (dt / constants::e0) * mom_term_sum;
  }
  elec_new[ncells] = elec_new[0];
}

// PUSH PARTICLES USING EXPLICIT SCHEME
void pushParticles(
  const int& ncells,
  const int& nspecies,
  const int& np,
  double* pos,
  double* vel,
  double* elec,
  int* cell,
  double* charge,
  double* mass,
  const double& dt,
  const double& dx,
  const double& lx
)
{
  double accel;
  double dvel;
  double dpos;
  double shapeLH;
  double shapeRH;

  // loop through species
  for (int a = 0; a < nspecies; ++a)
  {
    // loop through particles
    for (int p = 0; p < np; ++p)
    {
      shapeLH = 1.0 - fabs(((double) cell[a*np + p]*dx) - pos[a*np + p])/dx;
      shapeRH = 1.0 - fabs(((double) (cell[a*np + p] + 1)*dx) - pos[a*np + p])/dx;

      accel = (charge[a] / mass[a]) * (elec[cell[a*np + p]] * shapeLH + elec[cell[a*np + p] + 1] * shapeRH);

      dvel = accel * dt;
      dpos = vel[a*np + p]*dt + 0.5*accel*dt*dt;

      pos[a*np + p] += dpos;
      vel[a*np + p] += dvel;

      // apply periodic boundary condition for position
      if (pos[a*np + p] < 0)
      {
        pos[a*np + p] += (floor(fabs(pos[a*np + p]) / lx) + 1.0) * lx;
      }
      else if (pos[a*np + p] > lx)
      {
        pos[a*np + p] -= floor(fabs(pos[a*np + p]) / lx) * lx;
      }

      cell[a*np + p] = (int) floor(pos[a*np + p] / dx);
    }
  }

}

// DISTRIBUTE PARTICLES ACCORDING TO PROFILE
void distributeParticles(
  const int& ncells,
  const int& nspecies,
  const int& np,
  std::vector<int> nv,
  double* pos,
  double* vel,
  int* cell,
  double* weight,
  std::vector<double> species_charges,
  std::vector<double> species_masses,
  std::vector<double> species_T,
  std::vector<std::string> profiles,
  std::vector<std::string> perturb_profiles,
  std::vector<double> dens_perturb,
  std::vector<std::vector<double>> vel_ranges,
  std::vector<double> init_avgdens,
  std::vector<double> init_avgvels,
  const double& lx,
  const double& dx
)
{
  double* x_centre = new double[ncells];
  for (int i = 0; i < ncells; ++i)
  {
    x_centre[i] = ((double) i + 0.5) * dx;
  }
  for (int a = 0; a < nspecies; ++a)
  {
    double* v_centre = new double[nv[a]];
    double* fx = new double[ncells];
    double* fv = new double[ncells * nv[a]];
    double* species_dens = new double[ncells + 1];
    double v_min = vel_ranges[a][0];
    double v_max = vel_ranges[a][1];
    double dv = (v_max - v_min) / (double) nv[a];
    double avg_dens = 0.0;
    double shapeLH;
    double shapeRH;

    // calculate v_centre for this species
    for (int i = 0; i < nv[a]; ++i)
    {
      v_centre[i] = init_avgvels[a] + v_min + i*(v_max - v_min)/nv[a];
    }

    // calculate fx for this species
    for (int i = 0; i < ncells; i++) {
      if (perturb_profiles[a] == "sin")
      {
        fx[i] = init_avgdens[a] + (0.5 * dens_perturb[a]) * sin(2.0 * M_PI * x_centre[i] / lx);
      }
      else if (perturb_profiles[a] == "cos")
      {
        fx[i] = init_avgdens[a] + (0.5 * dens_perturb[a]) * cos(2.0 * M_PI * x_centre[i] / lx);
      }
  	}

    // calculate fv for this species
    if (profiles[a] == "boltzmann")
    {
      for (int i = 0; i < ncells; ++i)
      {
        for (int j = 0; j < nv[a]; j++)
        {
    			fv[j + i*nv[a]] = (init_avgdens[a] / sqrt(2.0 * M_PI * constants::Kb * species_T[a] / species_masses[a])) *
            exp(-species_masses[a] * (v_centre[j] - init_avgvels[a])*(v_centre[j] - init_avgvels[a])
            / (2.0 * constants::Kb * species_T[a]));
    		}
      }
    }
    else if (profiles[a] == "two_beams")
    {
      for (int i = 0; i < ncells; ++i)
      {
        for (int j = 0; j < nv[a]; j++)
        {
          if ((j == 0) || (j == nv[a] - 1))
          {
            fv[j + i*nv[a]] = 1.0;
          }
          else
          {
            fv[j + i*nv[a]] = 0.0;
          }
        }
      }
    }

    int cell_xtot = 0;
    int cell_xvtot = 0;

    // initialise running totals for the fx and fv functions
    double fx_sum = 0.0;
    double fv_sum;

    int p_ind = 0;

    // calculate the sum of the fx function
    for (int i = 0; i < ncells; ++i)
    {
      fx_sum += fx[i];
    }

    // loop through each x point
    for (int i = 0; i < ncells; ++i)
    {
      // calculate sum of the values in fv function for this x value
      fv_sum = 0.0;
      for (int j = 0; j < nv[a]; ++j)
      {
        // fv is row-major
        fv_sum += fv[j + i*nv[a]];
      }

      // calculate the total number of particles across all xv cells for this x position
      cell_xtot = (int) round(fx[i] * np / fx_sum);

      for (int j = 0; j < nv[a]; ++j)
      {
        // calculate the cell totals for each velocity for this position
        cell_xvtot = (int) round(fv[j + i*nv[a]] * cell_xtot / fv_sum);

        // create a number of particles equal to the cell_vtot for this xv cell
        if (cell_xvtot > 0)
        {
          for (int k = 0; k < cell_xvtot; ++k)
          {
            if (p_ind < np)
            {
              pos[a*np + p_ind] = x_centre[i] + (((double) rand() / RAND_MAX) - 0.5) * dx;
              vel[a*np + p_ind] = v_centre[j] + (((double) rand() / RAND_MAX) - 0.5) * dv;
              cell[a*np + p_ind] = floor((pos[a*np + p_ind])/dx);
              //std::cout << "Created particle " << p_ind << " with pos: " << pos[a*np + p_ind] << ", vel: " << vel[a*np + p_ind] << ", cell: " << cell[a*np + p_ind] << std::endl;
              p_ind++;
            }
          }
        }
      }
    }

    // check if there is particle deficit and correct any such
    while (p_ind < np)
    {
      pos[a*np + p_ind] = ((double) rand() / RAND_MAX) * lx;
      vel[a*np + p_ind] = v_min + ((double) rand() / RAND_MAX) * (v_max - v_min);
      cell[a*np + p_ind] = floor((pos[a*np + p_ind])/dx);
      p_ind++;
    }

    // set particle weight initially to 1.0 to calibrate
    weight[a] = 1.0;

    // set density and momentum vectors to zero
    for (int i = 0; i < ncells + 1; ++i)
    {
      // set moments to zero
      species_dens[i] = 0.0;
    }

    // loop through particles
    for (int p = 0; p < np; p++)
    {
      shapeLH = 1.0 - fabs(((double) cell[a*np + p]*dx) - pos[a*np + p])/dx;
      shapeRH = 1.0 - fabs(((double) (cell[a*np + p] + 1)*dx) - pos[a*np + p])/dx;

      species_dens[cell[a*np + p]] += shapeLH * weight[a];
      species_dens[cell[a*np + p] + 1] += shapeRH * weight[a];
    }

    // apply periodic boundary condition to domain boundaries
    species_dens[0] += species_dens[ncells];
    species_dens[ncells] = species_dens[0];

    avg_dens = 0.0;
    // calculate avg density using this particle weight
    for (int i = 0; i < ncells + 1; ++i)
    {
      avg_dens += species_dens[i];
    }
    avg_dens /= (double) (ncells + 1);

    // scale particle weight so avg density matches target density
    weight[a] = init_avgdens[a] / avg_dens;

    std::cout << "Weight for species " << a << ": " << weight[a] << std::endl;

    delete[] v_centre, fx, fv, species_dens;
  }
}

// PRINT VALUES
void printValues(
  const int& k,
  const int& print_freq,
  const int& ncells,
  const int& nspecies,
  const int& np,
  const bool& print_posvel,
  double* pos,
  double* vel,
  double* elec,
  double* dens,
  double* mom,
  std::vector<std::string> names
)
{
  if (k % print_freq == 0)
  {
    if (print_posvel == true)
    {
      for (int a = 0; a < nspecies; ++a)
      {
        std::ofstream pos_file("./output/" + names[a] + "_pos.txt", std::ofstream::out | std::ofstream::app);
        for (int p = 0; p < np; p++)
        {
          pos_file.width(15);
          pos_file << pos[a*np + p] << "\t";
        }
        pos_file << std::endl;
        pos_file.close();

        std::ofstream vel_file("./output/" + names[a] + "_vel.txt", std::ofstream::out | std::ofstream::app);
        for (int p = 0; p < np; p++)
        {
          vel_file.width(15);
          vel_file << vel[a*np + p] << "\t";
        }
        vel_file << std::endl;
        vel_file.close();
      }
    }

    std::ofstream elec_file("./output/elec.txt", std::ofstream::out | std::ofstream::app);
    for (int i = 0; i < ncells + 1; i++)
    {
      elec_file.width(15);
      elec_file << elec[i] << "\t";
    }
    elec_file << std::endl;
    elec_file.close();

    for (int a = 0; a < nspecies; a++)
    {
      std::ofstream dens_file("./output/" + names[a] + "_dens.txt", std::ofstream::out | std::ofstream::app);
      std::ofstream mom_file("./output/" + names[a] + "_mom.txt", std::ofstream::out | std::ofstream::app);
      for (int i = 0; i < ncells + 1; i++)
      {
        dens_file.width(15);
        dens_file << dens[a*(ncells + 1) + i] << "\t";

        mom_file.width(15);
        mom_file << mom[a*(ncells + 1) + i] << "\t";
      }
      dens_file << std::endl;
      dens_file.close();

      mom_file << std::endl;
      mom_file.close();
    }
  }
}

// ====================================================

int main()
{
  // ====================================================
  // SAVED CONFIGS

  bool print_posvel = true;
  int print_freq = 10;
  int test_id = 2;

  // Bench mark test 1 - Landau damping
  TestConfig landau_config;
  landau_config.ncells = 40;
  landau_config.nt = 100;
  landau_config.lx = 1.0;
  landau_config.dt = 1.0 * pow(10.0, -6);
  landau_config.nspecies = 1;
  landau_config.np = 100000;
  landau_config.nv = { 50 };
  landau_config.names = { "electron" };
  landau_config.profiles = { "boltzmann",};
  landau_config.perturb_profiles = { "cos" };
  landau_config.vel_ranges = { {-15000.0, 15000.0}};
  landau_config.init_avgdens = { 1.0 };
  landau_config.init_avgvels = { 0.0 };
  landau_config.dens_perturb = { 0.3 };
  landau_config.species_charges = { constants::electron_charge };
  landau_config.species_masses = { constants::electron_mass };
  landau_config.species_T = { 1.0 };

  // Bench mark test 2 - Two stream instability
  TestConfig two_stream_config;
  two_stream_config.ncells = 40;
  two_stream_config.nt = 1000;
  two_stream_config.lx = 1.0;
  two_stream_config.dt = 3.0 * pow(10.0, -3);
  two_stream_config.nspecies = 1;
  two_stream_config.np = 100000;
  two_stream_config.nv = { 5 };
  two_stream_config.names = { "electron", "ion" };
  two_stream_config.profiles = { "two_beams" };
  two_stream_config.perturb_profiles = { "cos" };
  two_stream_config.vel_ranges = { {-0.1, 0.1} };
  two_stream_config.init_avgdens = { 1.0 };
  two_stream_config.init_avgvels = { 0.0 };
  two_stream_config.dens_perturb = { 0.3 };
  two_stream_config.species_charges = { constants::electron_charge };
  two_stream_config.species_masses = { constants::electron_mass };
  two_stream_config.species_T = { 1.0 };

  // Bench mark test 3 - Shockwave
  TestConfig shockwave_config;
  shockwave_config.ncells = 140;
  shockwave_config.nt = 10000;
  shockwave_config.lx = 1.0;
  shockwave_config.dt = pow(10.0, -6);
  shockwave_config.nspecies = 2;
  shockwave_config.np = 10000;
  shockwave_config.nv = { 50, 50 };
  shockwave_config.names = { "electron", "ion" };
  shockwave_config.profiles = { "boltzmann", "boltzmann" };
  shockwave_config.perturb_profiles = { "cos" , "cos" };
  shockwave_config.vel_ranges = { {-15000.0, 15000.0}, {-15000.0, 15000.0} };
  shockwave_config.init_avgdens = { 1.0, 1.0 };
  shockwave_config.init_avgvels = { 0.0, 50000.0 };
  shockwave_config.dens_perturb = { 0.3, 0.3 };
  shockwave_config.species_charges = { constants::electron_charge, constants::ion_charge };
  shockwave_config.species_masses = { constants::electron_mass, constants::ion_mass };
  shockwave_config.species_T = { 1.0, 2.0 * pow(10.0, -4) };

  std::vector<TestConfig> saved_tests = { landau_config, two_stream_config, shockwave_config };

  // ====================================================
  // SCHEME SETTINGS
  int ncells = saved_tests[test_id].ncells;
  int nspecies = saved_tests[test_id].nspecies;
  int np = saved_tests[test_id].np;
  int nt = saved_tests[test_id].nt;
  double dt = saved_tests[test_id].dt;
  double lx = saved_tests[test_id].lx;
  double dx = lx / (double) ncells;

  // ====================================================
  // ARRAYS

  double* elec      = new double[ncells + 1];                 // no. rows = ncells + 1 ; no. cols = 1 ;
  double* elec_new  = new double[ncells + 1];                 // no. rows = ncells + 1 ; no. cols = 1 ;
  double* dens      = new double[(ncells + 1) * nspecies];    // no. rows = ncells + 1 ; no. cols = nspecies ; COL-MAJOR
  double* mom       = new double[(ncells + 1) * nspecies];    // no. rows = ncells + 1 ; no. cols = nspecies ; COL-MAJOR

  double* pos       = new double[np * nspecies];              // no. rows = np ; no. cols = nspecies ; COL-MAJOR
  double* vel       = new double[np * nspecies];              // no. rows = np ; no. cols = nspecies ; COL-MAJOR
  int* cell         = new int[np * nspecies];                 // no. rows = np ; no. cols = nspecies ; COL-MAJOR

  double* charge    = new double[nspecies];                   // no. rows = nspecies ;
  double* mass      = new double[nspecies];                   // no. rows = nspecies ;
  double* weight    = new double[nspecies];                   // no. rows = nspecies ;

  for (int a = 0; a < nspecies; ++a)
  {
    charge[a] = saved_tests[test_id].species_charges[a];
    mass[a] = saved_tests[test_id].species_masses[a];
  }

  // ====================================================
  // CLEAN PRINT FILES

  for (int a = 0; a < nspecies; ++a)
  {
    std::ofstream pos_file("./output/" + saved_tests[test_id].names[a] + "_pos.txt", std::ofstream::out | std::ofstream::trunc);
    pos_file.close();

    std::ofstream vel_file("./output/" + saved_tests[test_id].names[a] + "_vel.txt", std::ofstream::out | std::ofstream::trunc);
    vel_file.close();
  }

  std::ofstream elec_file("./output/elec.txt", std::ofstream::out | std::ofstream::trunc);
  elec_file.close();

  for (int a = 0; a < nspecies; ++a)
  {
    std::ofstream dens_file("./output/" + saved_tests[test_id].names[a] + "_dens.txt", std::ofstream::out | std::ofstream::trunc);
    dens_file.close();

    std::ofstream mom_file("./output/" + saved_tests[test_id].names[a] + "_mom.txt", std::ofstream::out | std::ofstream::trunc);
    mom_file.close();
  }

  // ====================================================

  // initialise particle positions and velocities
  distributeParticles(
    ncells,
    saved_tests[test_id].nspecies,
    saved_tests[test_id].np,
    saved_tests[test_id].nv,
    pos,
    vel,
    cell,
    weight,
    saved_tests[test_id].species_charges,
    saved_tests[test_id].species_masses,
    saved_tests[test_id].species_T,
    saved_tests[test_id].profiles,
    saved_tests[test_id].perturb_profiles,
    saved_tests[test_id].dens_perturb,
    saved_tests[test_id].vel_ranges,
    saved_tests[test_id].init_avgdens,
    saved_tests[test_id].init_avgvels,
    lx,
    dx
  );

  // initialise the electric field with zeros
  for (int i = 0; i < ncells; ++i)
  {
    elec[i] = 0.0;
  }

  // evolution
  for (int k = 0; k < nt; k++)
  {
    // get momentum
    accumulateMoments(
      ncells,
      nspecies,
      np,
      pos,
      vel,
      cell,
      weight,
      dens,
      mom,
      dx
    );

    // print values for pevious timestep
    printValues(
      k,
      print_freq,
      ncells,
      nspecies,
      np,
      print_posvel,
      pos,
      vel,
      elec,
      dens,
      mom,
      saved_tests[test_id].names
    );

    // solve electric field
    solveField(
      ncells,
      nspecies,
      elec,
      elec_new,
      mom,
      charge,
      dt
    );

    // push particles
    pushParticles(
      ncells,
      nspecies,
      np,
      pos,
      vel,
      elec,
      cell,
      charge,
      mass,
      dt,
      dx,
      lx
    );

    // step electric field
    for (int i = 0; i < ncells + 1; ++i)
    {
      elec[i] = elec_new[i];
    }
  }

  std::cout << "PROGRAM COMPLETED" << std::endl;

  return 0;
}