#include <string>
#include <cmath>

#include <CL/cl.hpp>

#ifndef SPECIES_H
#define SPECIES_H

typedef struct params_tag
{
  cl_int nx;
  cl_float lx;
  cl_float dt;
  cl_float charge;
  cl_float mass;
  cl_float weight;
  cl_int accelerate;
  cl_float pic_tol;
} params;

class Species
{
private:
  // device type
  std::string device_type;

  // numerical scheme parameters
  int np;
  int nx;
  double lx;
  double dx;
  double dt;
  double* x_face;            // nx + 1
  double* x_centre;          // nx

  // particle quantities
  std::string name;
  double charge;
  double mass;
  double weight;
  double* pos;                // np
  double* vel;                // np
  double* pos_new;            // np
  double* vel_new;            // np

  // moment quantities
  double* dens;               // row-major | ((nx) ; 2) -> dens for each cell centre for (k, k+1) timesteps
  double* mom;                // row-major | ((nx + 1) ; 1) -> mom for each cell face for (k+1/2) timesteps
  double* avgmom;             // row-major | ((nx + 1) ; 1) -> avgmom for each cell face for (k+1/2) timesteps
  double* stress;             // row-major | ((nx) ; 2) -> stress for each cell centre for (k, k+1) timesteps
  double* nstress;            // row-major | ((nx) ; 1) -> normalised stress for each cell centre for k+1/2 timestep
  double* gamma;              // row-major | (nx ; 1) -> consistency parameter for each cell RH face

  // LO system quantities
  double* elec;               // row-major | (nx ; 2)

  // analysis variables - these are for validation/benchmarking/etc.
  int single_particle_ind = 0;
  double* dens_single;    // for continuity residual of 1 particle
  double* avgmom_single;  // for continuity residual of 1 particle
  double* continuity_res_single; // for continuity residual of 1 particle
  double* continuity_res; // for continuity residual of all particles

  // ====================================================

  // tolerances
  double pic_tol = pow(10.0,-5);
  int fixed_iter_max = 4;
  int max_picard = 5;

  // runtime data collection variables
  long int* methods_tracker;
  long int total_substeps;

  // openmp runtime variables
  int target_num_threads = 1;
  int chunksize = 10;

  // opencl runtime variables
  int workgroup_size = 256;
  int num_workgroups;

  cl::Context context;
  cl::CommandQueue queue;
  cl::Kernel accumulator_kernel;
  cl::Kernel push_kernel;

  cl::Buffer elec_d;
  cl::Buffer pos_d;
  cl::Buffer vel_d;
  cl::Buffer pos_new_d;
  cl::Buffer vel_new_d;
  cl::Buffer avgmom_expanded_d;

  float* elec_h;
  float* pos_h;
  float* vel_h;
  float* pos_new_h;
  float* vel_new_h;
  float* avgmom_expanded_h;

public:
// empty ctor
Species();

// initialisation ctor
Species(
  const cl::Context& _context,
  const cl::CommandQueue& _queue,
  const cl::Kernel& _accumulator_kernel,
  const cl::Kernel& _push_kernel,
  const std::string& _device_type,
  const int& _np,
  const int& _nx,
  const int& nv,
  const double& _lx,
  const double& _dt,
  double* _elec,
  double* _x_face,
  double* _x_centre,
  const double* v_centre,
  const std::string& _name,
  const double& _charge,
  const double& _mass,
  const double& v_min,
  const double& v_max,
  double* fx,
  double* fv,
  const double& target_dens,
  const double& vel_perturb,
  const std::string& vel_perturb_profile
);

// dtor
~Species();

void accumulate_moments(const double& dt_target);

// push subfunctions
void adaptive_push(
  // write:
  int&          _method_flag,
  double&       _accel,
  double&       _dsubt,
  double&       _dsubvel,
  double&       _dsubpos,
  int&          _dsubcell,
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const double& _subt,
  const double& _subpos,
  const double& _subvel,
  const int&    _subcell,
  const double& _dsubpos_lh,
  const double& _dsubpos_rh,
  double*       _elec
);

void direct_push(
  // write:
  int&          _method_flag,
  double&       _accel,
  double&       _dsubt,
  double&       _dsubvel,
  double&       _dsubpos,
  int&          _dsubcell,
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const double& _subt,
  const double& _subpos,
  const double& _subvel,
  const double& _subcell,
  const double& _dsubpos_lh,
  const double& _dsubpos_rh,
  double*       _elec
);

void fixed_iter_push(
  // write:
  int&          _method_flag,
  double&       _accel,
  double&       _dsubt,
  double&       _dsubvel,
  double&       _dsubpos,
  int&          _dsubcell,
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const double& _subt,
  const double& _subpos,
  const double& _subvel,
  const double& _subcell,
  const double& _dsubpos_lh,
  const double& _dsubpos_rh,
  double*       _elec
);

void get_face_shapes(
  // read/write:
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const int&    _subcell,
  const double& _pos
);

void get_accel(
  // write:
  double& _accel,
  // read:
  const double& _shape_lhface,
  const double& _shape_rhface,
  const int&    _subcell,
  double*       _elec
);

void solve_dsubpos(
  // write:
  double&       _shape_lhface,
  double&       _shape_rhface,
  double&       _accel,
  double&       _dsubvel,
  // read/write:
  double&       _dsubpos,
  // read:
  const double& _subpos,
  const double& _subvel,
  const int&    _subcell,
  const double& _dsubt,
  double*       _elec
);

void solve_dsubt(
  // read/write:
  double&       _dsubt,
  // read:
  const double& _dsubpos,
  const double& _accel,
  const double& _subvel
);

void substep(
  // write:
  double&       _subpos_new,
  double&       _subvel_new,
  int&          _subcell_new,
  // read/write:
  double&       _subt,
  double&       _subpos,
  double&       _subvel,
  int&          _subcell,
  // read:
  const double& _dsubt,
  const double& _dsubpos,
  const double& _dsubvel,
  const int&    _dsubcell
);

void accumulate_avgmom(
  // write:
  double*       _avgmom,
  double&       _dsubvel,
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const double& _subvel,
  const int&    _subcell,
  const double& _dsubt
);

void accumulate_avgmom_with_single(
  // write:
  double*       _avgmom,
  double*       _avgmom_single,
  double&       _dsubvel,
  double&       _shape_lhface,
  double&       _shape_rhface,
  // read:
  const double& _subvel,
  const int&    _subcell,
  const double& _dsubt,
  const int& pflag
);

// push functions
void push_sthread(const int& accelerate, const double& dt_new);
void push_mthread(const int& accelerate);
void push_gpu(const int& accelerate);

void step();
void step_mthread();

// getters
double* get_dens_ptr();
double* get_mom_ptr();
double* get_avgmom_ptr();
double* get_stress_ptr();
double* get_nstress_ptr();
double* get_gamma_ptr();

void clear_files();
void print_vals(const bool& particles_flag, const int& k);
void print_methods_tracker();

};

#endif
