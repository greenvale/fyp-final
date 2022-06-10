#include <iostream>
#include <cmath>
#include <fstream>

#include <CL/cl.hpp>

#include "species.h"

// sign function (imported)
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// empty ctor
Species::Species()
{
    std::cout << "Empty species constructor called" << std::endl;
}

// initialisation ctor
Species::Species(
    //=======================
    // openCL objects - will be nil if GPU isn't target device
    const cl::Context&        _context,
    const cl::CommandQueue&   _queue,
    const cl::Kernel&         _accumulator_kernel,
    const cl::Kernel&         _push_kernel,
    //=======================
    const std::string&  _device_type,
    const int&          _np,
    const int&          _nx,
    const int&           nv,
    const double&       _lx,
    const double&       _dt,
    double*             _elec,
    double*             _x_face,
    double*             _x_centre,
    const double*        v_centre,
    const std::string&  _name,
    const double&       _charge,
    const double&       _mass,
    const double&        v_min,
    const double&        v_max,
    double*              fx,
    double*              fv,
    const double&        target_dens,
    const double&        vel_perturb,
    const std::string&   vel_perturb_profile
)
{
    device_type = _device_type;

    np        = _np;
    nx        = _nx;
    lx        = _lx;
    dt        = _dt;
    dx        =  lx / nx;
    elec      = _elec;
    x_face    = _x_face;
    x_centre  = _x_centre;
    name      = _name;
    charge    = _charge;
    mass      = _mass;
    double dv =  (v_max - v_min) / nv;

    // check if np needs to be changed for gpu workgroup dimensions
    if (device_type == "gpu")
    {
        // calculate number of workgroups
        if (np % workgroup_size != 0)
        {
        np = floor(np / workgroup_size) * workgroup_size;
        std::cout << "!!! np changed to " << np << " (" << workgroup_size << " x " << np / workgroup_size << ") to be congruent with workgroup_size !!!" << std::endl;
        }
    }

    // allocate memory for arrays
    pos       =   new double[np];
    pos_new   =   new double[np];
    vel       =   new double[np];
    vel_new   =   new double[np];
    dens      =   new double[(nx) * 2];
    mom       =   new double[(nx + 1) * 2];
    avgmom    =   new double[nx + 1];
    stress    =   new double[(nx) * 2];
    nstress   =   new double[nx];
    gamma     =   new double[nx];
    methods_tracker = new long int[4];
    for (int i = 0; i < 4; ++i)
    {
        methods_tracker[i] = 0;
    }
    total_substeps = 0;

    // analysis variables
    dens_single = new double[(nx) * 2];
    avgmom_single = new double[nx + 1];
    continuity_res = new double[nx];
    continuity_res_single = new double[nx];

    // reset output files
    clear_files();

    // particle initialisation
    // initialise running totals for cell totals for all v cells of a certain x value
    int cell_xtot = 0;
    int cell_xvtot = 0;

    // initialise running totals for the fx and fv functions
    double fx_sum = 0.0;
    double fv_sum;

    int p_ind = 0;

    // calculate the sum of the fx function
    for (int i = 0; i < nx; ++i)
    {
        fx_sum += fx[i];
    }

    // loop through each x point
    for (int i = 0; i < nx; ++i)
    {
        // calculate sum of the values in fv function for this x value
        fv_sum = 0.0;
        for (int j = 0; j < nv; ++j)
        {
        // fv is row-major
        fv_sum += fv[j + i*nv];
        }

        // calculate the total number of particles across all xv cells for this x position
        cell_xtot = (int) round(fx[i] * np / fx_sum);

        double perturb_factor;

        for (int j = 0; j < nv; ++j)
        {
            // calculate the cell totals for each velocity for this position
            cell_xvtot = (int) round(fv[j + i*nv] * cell_xtot / fv_sum);

            // create a number of particles equal to the cell_vtot for this xv cell
            if (cell_xvtot > 0)
            {
                for (int k = 0; k < cell_xvtot; ++k)
                {
                if (p_ind < np)
                {
                    pos[p_ind] = x_centre[i] + (((double) rand() / RAND_MAX) - 0.5) * dx;
                    if (vel_perturb_profile == "sin")
                    {
                        perturb_factor = sin(2.0 * M_PI * pos[p_ind] / lx);
                    }
                    else if (vel_perturb_profile == "cos")
                    {
                        perturb_factor = cos(2.0 * M_PI * pos[p_ind] / lx);
                    }
                    else
                    {
                        perturb_factor = 0.0;
                    }
                    //std::cout << "vel perturb: " << vel_perturb << std::endl;
                    //std::cout << "vel shape: " << vel_perturb_profile << std::endl;
                    //std::cout << vel_perturb * perturb_factor << std::endl;
                    vel[p_ind] = v_centre[j] + ((((double) rand() / RAND_MAX) - 0.5) * dv) + (vel_perturb * perturb_factor);
                    //std::cout << "Created particle " << p_ind << " with pos: " << pos[p_ind] << ", vel: " << vel[p_ind] << std::endl;
                    p_ind++;
                }
                }
            }
        }
    }

    // check if there is particle deficit and correct any such
    while (p_ind < np)
    {
        pos[p_ind] = ((double) rand() / RAND_MAX) * lx;
        vel[p_ind] = v_min + ((double) rand() / RAND_MAX) * (v_max - v_min);
        p_ind++;
    }

    // calculate density
    //loop through particles
    double scaled_pos;
    int cell_lhface;
    int cell_rhface;
    int cell_rhcentre;
    int cell_midcentre;
    int cell_lhcentre;
    double d_lhcentre;
    double d_midcentre;
    double d_rhcentre;
    double shape_lhface;
    double shape_rhface;
    double shape_midcentre;
    double shape_lhcentre;
    double shape_rhcentre;

    // set dens and stress to zero ahead of accumulation
    for (int i = 0; i < nx; ++i)
    {
        dens[2*i + 0]   = 0.0;
        dens_single[2*i + 0] = 0.0;
        stress[2*i + 0] = 0.0;
    }
    for (int i = 0; i < nx + 1; ++i)
    {
        mom[2*i + 0] = 0.0;
    }

    for (int p = 0; p < np; ++p)
    {
        // calculated scaled positions which are used in calculating shapes
        scaled_pos = pos[p] / dx;

        // calculate LH and RH centre indices for current position
        cell_lhface = (int) floor(scaled_pos);
        cell_rhface = cell_lhface + 1;

        // calculate LH, middle and RH face indices for new position
        cell_midcentre = (int) floor(scaled_pos);
        cell_lhcentre = cell_midcentre - 1;
        cell_rhcentre = cell_midcentre + 1;

        // calculate first-order shapes for current position
        shape_lhface = 1.0 - fabs(scaled_pos - (double) cell_lhface);
        shape_rhface = 1.0 - fabs(scaled_pos - (double) cell_rhface);

        // calculate the second-order shapes for future new position
        d_lhcentre  = fabs(scaled_pos - ((double) cell_lhcentre   + 0.5));
        d_midcentre = fabs(scaled_pos - ((double) cell_midcentre  + 0.5));
        d_rhcentre  = fabs(scaled_pos - ((double) cell_rhcentre   + 0.5));

        // calculate second order shapes
        // for middle cell centre, always <= 0.5
        shape_midcentre = 0.75 - d_midcentre*d_midcentre;
        // for LH and RH neighbouring cell centres, always > 0.5
        shape_lhcentre = 0.5 * (1.5 - d_lhcentre) * (1.5 - d_lhcentre);
        shape_rhcentre = 0.5 * (1.5 - d_rhcentre) * (1.5 - d_rhcentre);

        // apply periodic boundary conditions to indices for accumulation
        // face indices
        if (cell_lhcentre < 0)
        {
        cell_lhcentre = nx - 1;
        }
        cell_rhcentre = ((cell_rhcentre) % (nx));

        // accumulate moment contributions (without weight / dx contribution)
        // initialise density @ t = 0
        dens[2*cell_lhcentre  + 0]    += shape_lhcentre;
        dens[2*cell_midcentre + 0]    += shape_midcentre;
        dens[2*cell_rhcentre  + 0]    += shape_rhcentre;
        if (p == single_particle_ind)
        {
        dens_single[2*cell_lhcentre  + 0] += shape_lhcentre;
        dens_single[2*cell_midcentre + 0] += shape_midcentre;
        dens_single[2*cell_rhcentre  + 0] += shape_rhcentre;
        }

        // initialise momentum @ t = 0
        mom[2*cell_lhface + 0] += shape_lhface * vel[p];
        mom[2*cell_rhface + 0] += shape_rhface * vel[p];

        // initialise stress @ t = 0
        stress[2*cell_lhcentre  + 0]  += shape_lhcentre   * vel[p] * vel[p];
        stress[2*cell_midcentre + 0]  += shape_midcentre  * vel[p] * vel[p];
        stress[2*cell_rhcentre  + 0]  += shape_rhcentre   * vel[p] * vel[p];
    }

    for (int i = 0; i < nx; ++i)
    {
        dens[2*i + 0] *= 1.0 / dx;
    }

    // calculate avg density using this particle weight
    double avg_dens = 0.0;
    for (int i = 0; i < nx; ++i)
    {
        avg_dens += dens[2*i + 0];
    }
    avg_dens /= (nx);

    // scale particle weight so avg density matches target density
    weight = target_dens / avg_dens;

    // initialise moments for current timestep (k=0) (only density, avgmom and stress need initialisation)
    for (int i = 0; i < nx; ++i)
    {
        dens[2*i + 0] *= weight;
        dens_single[2*i + 0] *= weight / dx;
        stress[2*i + 0] *= weight / dx;
    }
    for (int i = 0; i < nx + 1; ++i)
    {
        mom[2*i + 0] *= weight / dx;
    }

    // periodic boundary condition for momentum
    mom[2*0 + 0] += mom[2*nx + 0];
    mom[2*nx + 0] = mom[2*0 + 0];

    // set avgmom to zeros for mid timestep k+1/2
    for (int i = 0; i < nx + 1; ++i)
    {
        avgmom[i] = 0.0;
        avgmom_single[i] = 0.0;
    }

    // estimate future pos and vel values to calculate future density and stress
    for (int p = 0; p < np; ++p)
    {
        pos_new[p] = pos[p];
        vel_new[p] = vel[p];
    }

    // create opencl objects
    if (device_type == "gpu")
    {
        context             = _context;
        accumulator_kernel  = _accumulator_kernel;
        push_kernel       = _push_kernel;
        queue               = _queue;
        num_workgroups = np / workgroup_size;

        // initialise host float arrays
        elec_h    = new float[2 * (nx + 1)];
        pos_h     = new float[np];
        vel_h     = new float[np];
        pos_new_h = new float[np];
        vel_new_h = new float[np];
        avgmom_expanded_h = new float[(nx + 1) * num_workgroups];

        // initialise buffers
        elec_d = cl::Buffer(context, CL_MEM_READ_ONLY,   sizeof(float) * (nx + 1) * 2);
        pos_d  = cl::Buffer(context, CL_MEM_READ_ONLY,   sizeof(float) * np);
        vel_d  = cl::Buffer(context, CL_MEM_READ_ONLY,   sizeof(float) * np);
        pos_new_d = cl::Buffer(context, CL_MEM_WRITE_ONLY,  sizeof(float) * np);
        vel_new_d = cl::Buffer(context, CL_MEM_WRITE_ONLY,  sizeof(float) * np);
        avgmom_expanded_d = cl::Buffer(context, CL_MEM_READ_WRITE,  sizeof(float) * (nx + 1) * num_workgroups);
    }
}

//==============================================================================
//==============================================================================

/* update moments:
-> dens (time k+1),
-> mom (time k),
-> stress (time k+1),
-> nstress (time k+1/2),
-> gamma (time k+1/2)
*/
void Species::accumulate_moments(const double& dt_target)
{
    dt = dt_target;

    // initialise runtime variables
    double scaled_pos;
    double scaled_pos_new;

    int cell_lhface;
    int cell_rhface;
    int cell_lhcentre;
    int cell_midcentre;
    int cell_rhcentre;

    double shape_lhface;
    double shape_rhface;
    double shape_lhcentre;
    double shape_midcentre;
    double shape_rhcentre;
    double d_lhcentre;
    double d_midcentre;
    double d_rhcentre;

    // coefficient variables
    double wt_inv_dx = weight / dx;

    // set relevant vectors to zero ahead of accumulation
    for (int i = 0; i < nx; ++i)
    {
        // set future density to zero
        dens[2*i + 1] = 0.0;

        // set future stress to zero
        stress[2*i + 1] = 0.0;

        // set single particle density to zero
        dens_single[2*i + 1] = 0.0;
    }
    for (int i = 0; i < nx + 1; ++i)
    {
        // set current momentum to zero
        mom[2*i + 1] = 0.0;
    }

    //loop through particles
    for (int p = 0; p < np; ++p)
    {
        // calculated scaled positions which are used in calculating shapes
        //scaled_pos = pos[p] / dx;
        scaled_pos_new = pos_new[p] / dx;

        // calculate LH and RH centre indices for current position
        cell_lhface = (int) floor(scaled_pos_new);
        cell_rhface = cell_lhface + 1;

        // calculate LH, middle and RH face indices for new position
        cell_midcentre = (int) floor(scaled_pos_new);
        cell_lhcentre = cell_midcentre - 1;
        cell_rhcentre = cell_midcentre + 1;

        // calculate first-order shapes for current position
        shape_lhface = 1.0 - fabs(scaled_pos_new - (double) cell_lhface);
        shape_rhface = 1.0 - fabs(scaled_pos_new - (double) cell_rhface);

        // calculate the second-order shapes for future new position
        d_lhcentre = fabs(scaled_pos_new - ((double) cell_lhcentre + 0.5));
        d_midcentre = fabs(scaled_pos_new - ((double) cell_midcentre + 0.5));
        d_rhcentre = fabs(scaled_pos_new - ((double) cell_rhcentre + 0.5));

        // calculate second order shapes
        // for middle cell centre, always <= 0.5
        shape_midcentre = 0.75 - (d_midcentre * d_midcentre);
        // for LH and RH neighbouring cell centres, always > 0.5
        shape_lhcentre = 0.5 * (1.5 - d_lhcentre) * (1.5 - d_lhcentre);
        shape_rhcentre = 0.5 * (1.5 - d_rhcentre) * (1.5 - d_rhcentre);

        // apply periodic boundary conditions to indices for accumulation
        // face indices
        if (cell_lhcentre < 0)
        {
        cell_lhcentre = nx - 1;
        }
        cell_rhcentre = ((cell_rhcentre) % (nx));

        // accumulate moment contributions (without weight / dx contribution)
        // density for t = k+1
        dens[2*cell_lhcentre  + 1] += shape_lhcentre;
        dens[2*cell_midcentre + 1] += shape_midcentre;
        dens[2*cell_rhcentre  + 1] += shape_rhcentre;

        // density of single particle for t = k+1
        if (p == single_particle_ind)
        {
        dens_single[2*cell_lhcentre  + 1] += shape_lhcentre;
        dens_single[2*cell_midcentre + 1] += shape_midcentre;
        dens_single[2*cell_rhcentre  + 1] += shape_rhcentre;
        }

        // momentum for t = k+1
        mom[2*cell_lhface + 1] += vel_new[p] * shape_lhface;
        mom[2*cell_rhface + 1] += vel_new[p] * shape_rhface;

        // stress for t = k+1
        stress[2*cell_lhcentre  + 1]  += vel_new[p] * vel_new[p] * shape_lhcentre;
        stress[2*cell_midcentre + 1]  += vel_new[p] * vel_new[p] * shape_midcentre;
        stress[2*cell_rhcentre  + 1]  += vel_new[p] * vel_new[p] * shape_rhcentre;

        //std::cout << vel_new[p] << std::endl;
    }

    for (int i = 0; i < nx; ++i)
    {
        dens[2*i + 1]   *= wt_inv_dx;
        stress[2*i + 1] *= wt_inv_dx;

        dens_single[2*i + 1] *= wt_inv_dx;
    }
    for (int i = 0; i < nx + 1; ++i)
    {
        mom[2*i + 1] *= wt_inv_dx;
        //std::cout << "Stress: " << mom[2*i + 1] << std::endl;
    }

    // periodic boundary condition
    mom[2*0 + 1] += mom[2*nx + 1];
    mom[2*nx + 1] = mom[2*0 + 1];

    // calculate nstress at cell centres and gamma at cell RH faces
    int lh_centre;
    int rh_centre;
    for (int i = 0; i < nx; ++i)
    {
        // normalised stress for this cell
        nstress[i] = (stress[2*i + 0] + stress[2*i + 1]) / (dens[2*i + 0] + dens[2*i + 1]);

        // catch NaN error - this shouldn't have to be used unless there is bad distribution of particles
        if (dens[2*i + 0] + dens[2*i + 1] == 0)
        {
            nstress[i] = 1.0;
            std::cout << "CAUGHT NAN ERROR" << std::endl;
        }
    }
    for (int i = 0; i < nx; ++i)
    {
        // calculate necessary indices for RH centre cells for dens and nstress
        if (i < nx - 1)
        {
            lh_centre = i;
            rh_centre = i + 1;
        }
        else
        {
            lh_centre = nx - 1;
            rh_centre = 0;
        }
        // calculate gamma @ RH face
        // requires: avgmom + mom @ RH face of this cell
        gamma[i] = 2.0 * (avgmom[i+1] - mom[2*(i+1) + 0]) / dt;
        // requires: dens + nstress @ this + RH cells

        gamma[i] += (0.5 / dx) * ((dens[2*rh_centre + 0] + dens[2*rh_centre + 1]) * nstress[rh_centre]
        - (dens[2*lh_centre + 0] + dens[2*lh_centre + 1]) * nstress[lh_centre]);

        gamma[i] -= 0.125 * (charge / mass) * (dens[2*lh_centre + 0] + dens[2*rh_centre + 0] + dens[2*lh_centre + 1] + dens[2*rh_centre + 1]) * (elec[2*(i + 1) + 0] + elec[2*(i + 1) + 1]);

        gamma[i] /= 0.5 * (dens[2*lh_centre + 1] + dens[2*rh_centre + 1]);

        //std::cout << (dens[2*lh_centre + 1] + dens[2*rh_centre + 1]) << std::endl;

        // catch NaN error - this shouldn't have to be used unless there is bad distribution of particles
        if (dens[2*lh_centre + 1] + dens[2*rh_centre + 1] == 0.0)
        {
            gamma[i] = 1.0;
            std::cout << "CAUGHT NAN ERROR" << std::endl;
        }
    }
}

//==============================================================================
//==============================================================================
// EVOLVE FUNCTIONS

/*
get LH face and RH face shapes (first-order)
*/
void Species::get_face_shapes(
    // read/write:
    double&       _shape_lhface,
    double&       _shape_rhface,
    // read:
    const int&    _subcell,
    const double& _pos
)
{
    _shape_lhface = 1.0 - fabs(_pos - (double) (_subcell));
    _shape_rhface = 1.0 - fabs(_pos - (double) (_subcell + 1));

    // if shape is negative, set to zero
    if (_shape_lhface < 0.0)
        _shape_lhface = 0.0;

    if (_shape_rhface < 0.0)
        _shape_rhface = 0.0;
}

/*
get accel from electric field and first-order shapes
*/
void Species::get_accel(
// write:
double&       _accel,
// read:
const double& _shape_lhface,
const double& _shape_rhface,
const int&    _subcell,
double*       _elec
)
{
    _accel = (0.5 * charge / mass) * ((_elec[2*_subcell + 0] + _elec[2*_subcell + 1])*_shape_lhface
    + (_elec[2*_subcell + 2 + 0] + _elec[2*_subcell + 2 + 1])*_shape_rhface);
}

//==============================================================================
/*
solve for dsubpos using Picard method
*/
void Species::solve_dsubpos(
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
)
{
    // initialise running variables
    bool pic_flag = false;
    int pic_index = -1;
    double tmp;

    // indefinite Picard loop
    while (pic_flag == false)
    {
        pic_index++;

        // calculate first-order shapes for half position before periodic boundary conditions are applied to subcell centre indicies
        get_face_shapes(_shape_lhface, _shape_rhface, _subcell, (_subpos + 0.5 * _dsubpos) / dx);

        // calculate acceleration for half position
        get_accel(_accel, _shape_lhface, _shape_rhface, _subcell, _elec);

        // calculate dsubvel and dsubpos
        _dsubvel = _dsubt * _accel;
        tmp = _dsubt * (_subvel + 0.5 * _dsubvel);

        // check convergence or if Picard iterations have reached max
        if (fabs(_dsubpos - tmp) < pic_tol) {
        pic_flag = true;
        }

        //std::cout << "dsubpos: " << tmp << std::endl;

        // update dsubpos
        _dsubpos = tmp;
    }
}

/*
solve for dsubt for Picard method
*/
void Species::solve_dsubt(
    // read/write:
    double&       _dsubt,
    // read:
    const double& _dsubpos,
    const double& _accel,
    const double& _subvel
)
{
    // initialise running variables
    bool pic_flag = false;
    int pic_index = -1;
    double tmp;

    // Indefinite Picard loop
    while (pic_flag == false)
    {
        pic_index++;

        // calculate dsubvel and dsubpos
        tmp = _dsubpos / (_subvel + 0.5 * _dsubt * _accel);

        if (tmp < 0.0)
        {
        tmp = 0.0;
        }

        // check convergence
        if (fabs(_dsubt - tmp) < pic_tol) {
        pic_flag = true;
        }

        //std::cout << "Solved dsubt: " << tmp << std::endl;

        // update dsubt
        _dsubt = tmp;
    }
}

//==============================================================================
/*
adaptive push
*/
void Species::adaptive_push(
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
)
{
    _dsubcell = 0;

    // estimate sub-cycle timestep subt with first-order approx
    if (_subvel != 0.0)
    {
        _dsubt = dx / fabs(_subvel);
    }
    else
    {
        _dsubt = dt;
    }

    // truncate dsubt if necessary
    if (_dsubt + _subt > dt)
    {
        _dsubt = dt - _subt;
    }

    // initialise dsubpos
    _dsubpos = _dsubt * _subvel;

    // solve for dsubpos using Picard method
    solve_dsubpos(_shape_lhface, _shape_rhface, _accel, _dsubvel, _dsubpos,
        _subpos, _subvel, _subcell, _dsubt, _elec);

    // check if LH cell face has been crossed
    if (_dsubpos <= _dsubpos_lh)
    {
        _dsubpos = _dsubpos_lh;
        // set method flag to 1 meaning particle terminates at cell face
        _method_flag = 1;
        _dsubcell = -1;
    }
    // check if RH cell face has been crossed
    else if (_dsubpos >= _dsubpos_rh)
    {
        _dsubpos = _dsubpos_rh;
        // set method flag to 1 meaning particle terminates at cell face
        _method_flag = 1;
        _dsubcell = 1;
    }

    // cell face has been crossed - must recalculate based on new dsubpos given by cell face location
    if (_method_flag == 1)
    {
        //std::cout << "Particle forced to land on face" << std::endl;
        // calculate face shapes for new dsubpos
        get_face_shapes(_shape_lhface, _shape_rhface, _subcell, (_subpos + 0.5 * _dsubpos) / dx);

        // get accel from electric field given shape functions and cell index
        get_accel(_accel, _shape_lhface, _shape_rhface, _subcell, _elec);

        // solve for dsubt
        solve_dsubt(_dsubt, _dsubpos, _accel, _subvel);

        // update dsubvel
        _dsubvel = _dsubt * _accel;
    }
}

//==============================================================================
/*
direct particle push function
-
*/
void Species::direct_push(
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
)
{
    // initialise runtime variables
    double discrim;
    double dsubt_soln1;
    double dsubt_soln2;

    // calculate shapes manually as formulae not required
    _shape_lhface = 0.5 * (double)(1 + _dsubcell);
    _shape_rhface = 0.5 * (double)(1 - _dsubcell);

    // calculate dsubpos given previously direction
    _dsubpos = (double) _dsubcell * dx;

    // get accel from electric field given shape functions and cell index
    get_accel(_accel, _shape_lhface, _shape_rhface, _subcell, _elec);

    // calculate discriminant
    discrim = _subvel*_subvel + 2.0*_accel*_dsubpos;

    // check if discriminant is negative
    if (discrim < 0.0)
    {
        // set method flag to 3 to show invalidated
        _method_flag = 3;
        // set discrim to zero to allow other calculations to go ahead
        discrim = 0.0;
    }

    // Root method 1 (inactive)
    /*
    dsubt_soln1 = (-subvel - sqrt(discrim)) / accel;
    dsubt_soln2 = (-subvel + sqrt(discrim)) / accel;
    */

    // Root method 2 (active) (meant to be more accurate)
    dsubt_soln2 = -0.5 * (_subvel + sgn(_subvel) * sqrt(discrim));
    dsubt_soln1 = dsubt_soln2 / (0.5 * _accel);
    dsubt_soln2 = (-_dsubpos) / dsubt_soln2;

    // if dsubt_soln1 positive and not invalidated, assume it is soln
    if ((dsubt_soln1 > 0.0) && (_method_flag != 3))
    {
        _dsubt = dsubt_soln1;
        // use temporary method flag to indicate viable solution found
        _method_flag = 2;
    }

    // if dsubt_soln2 positive, not invalidated and less than dsubt_soln1 then it replaces dsubt_soln1
    if ((dsubt_soln2 > 0.0) && (_method_flag != 3) && (dsubt_soln2 < dsubt_soln1))
    {
        // check if no solution found already or this solution is a smaller positive value
        _dsubt = dsubt_soln2;
        // use temporary method flag to indicate viable solution found
        _method_flag = 2;
    }

    // if positive soln not requiring truncation then use direct method
    // else use adaptive algorithm
    if ((_method_flag == 2) && (_dsubt + _subt <= dt))
    {
        // calculate dsubvel
        _dsubvel = _accel * _dsubt;
    }
    else
    {
        // use adaptive algorithm
        _method_flag = 0;
    }
}

//==============================================================================
/*
substep :
- step forward sub-cycle values
*/
void Species::substep(
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
)
{
    // dubspos, dsubvel and dsubt now final values
    // calculate new subpos, subvel, subcell
    _subpos_new   = _subpos   + _dsubpos;
    _subvel_new   = _subvel   + _dsubvel;
    _subcell_new  = _subcell  + _dsubcell; // GET RID OF THIS!!!

    // apply periodic boundary conditions to new subcell and subpos
    _subcell_new = _subcell + _dsubcell;
    if (_subcell_new < 0)
    {
        _subcell_new  += nx;
        _subpos_new   += lx;
    }
    else if (_subcell_new > (nx - 1))
    {
        _subcell_new  -= nx;
        _subpos_new   -= lx;
    }

    // step subcycle values
    _subt    += _dsubt;
    _subpos   = _subpos_new;
    _subvel   = _subvel_new;
    _subcell  = _subcell_new;
}

/*
accumulate avgmom contributions for sub-cycle step
*/
void Species::accumulate_avgmom_with_single(
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
    const int& _pflag
)
{
    // calculate avgmom contribution
    _avgmom[_subcell]     += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_lhface * _dsubt;
    _avgmom[_subcell + 1] += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_rhface * _dsubt;

    // calculate avgmom contribution for single particle
    if (_pflag == 1)
    {
        _avgmom_single[_subcell]     += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_lhface * _dsubt;
        _avgmom_single[_subcell + 1] += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_rhface * _dsubt;
    }
}

// analytical version
void Species::accumulate_avgmom(
    // write:
    double*       _avgmom,
    double&       _dsubvel,
    double&       _shape_lhface,
    double&       _shape_rhface,
    // read:
    const double& _subvel,
    const int&    _subcell,
    const double& _dsubt
)
{
    // calculate avgmom contribution
    _avgmom[_subcell]     += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_lhface * _dsubt;
    _avgmom[_subcell + 1] += (1.0 / (dx * dt)) * weight * (_subvel + 0.5 * _dsubvel) * _shape_rhface * _dsubt;
}

//==============================================================================
/*
push particles with single thread
*/
void Species::push_sthread(const bool& accelerate, const double& dt_new)
{
    // initialise run-time variables
    bool   sub_flag;
    int    sub_index;
    int    method_flag;
    double subt;
    double subpos;
    double subvel;
    int    subcell;
    double subpos_new;
    double subvel_new;
    int    subcell_new;
    double accel;
    double dsubt;
    double dsubpos;
    double dsubvel;
    int    dsubcell = 0;
    double dsubpos_lh;
    double dsubpos_rh;
    double shape_lhface;
    double shape_rhface;

    dt = dt_new;

    // set avgmom vector to zeros
    for (int i = 0; i < nx + 1; ++i)
    {
        avgmom[i] = 0.0;

        // set avgmom single to zeros
        avgmom_single[i] = 0.0;
    }

    // loop through particles
    for (int p = 0; p < np; ++p)
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
            //std::cout << "Trying accelerator" << std::endl;
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
            elec
            );
        }
        else
        {
            // use adaptive push if accelerate block not used
            method_flag = 0;
        }

        // adaptive push (maybe direct push wasn't successful)
        if (method_flag == 0)
        {
            //std::cout << "Particle " << p << " adaptive push " << " subt: " << subt << std::endl;

            adaptive_push(
            method_flag,    accel,
            dsubt,          dsubvel,
            dsubpos,        dsubcell,
            shape_lhface,   shape_rhface,
            subt,           subpos,
            subvel,         subcell,
            dsubpos_lh,     dsubpos_rh,
            elec
            );

            //std::cout << "subvel: " << subvel << ", accel: " << accel << std::endl;
            //std::cout << "subpos: " << subpos << ", dsubpos: " << dsubpos << std::endl;
        }

        // check if subcycle finished
        if ((subt + dsubt >= dt) || (sub_index > 10000))
            sub_flag = true;

        int pflag = 0;
        if (p == single_particle_ind)
        {
            pflag = 1;
        }

        if (p == single_particle_ind)
        {
            /*
            std::cout << "push index " << sub_index <<  std::endl;
            std::cout << "push method " << method_flag << std::endl;
            std::cout << "dsubpos: " << dsubpos << std::endl;
            std::cout << "new pos: " << subpos + dsubpos << std::endl;
            std::cout << "pos in cell % " << (subpos)/dx - floor((subpos)/dx) << std::endl;
            std::cout << "pos in cell new % " << (subpos + dsubpos)/dx - floor((subpos + dsubpos)/dx) << std::endl;*/
        }

        // accumulate avgmom for this substep
        accumulate_avgmom(avgmom, dsubvel, shape_lhface, shape_rhface, subvel, subcell, dsubt);
        //accumulate_avgmom_with_single(avgmom, avgmom_single, dsubvel, shape_lhface, shape_rhface, subvel, subcell, dsubt, pflag);

        // step particle values
        substep(
            subpos_new,
            subvel_new,     subcell_new,
            subt,           subpos,
            subvel,         subcell,
            dsubt,          dsubpos,
            dsubvel,        dsubcell);

        // record method used to push particle
        methods_tracker[method_flag] += 1;
        }

        // record total sub-cycle steps
        total_substeps += sub_index + 1;

        // transfer sub-cycle new values to global new values at end of particle evolution
        pos_new[p]  = subpos_new;
        vel_new[p]  = subvel_new;

        //std::cout << "Particle " << p << "iteration: " << sub_index << ", subt: " << subt << std::endl;
    }

    // periodic boundary conditions
    avgmom[0] += avgmom[nx];
    avgmom[nx] = avgmom[0];

    avgmom_single[0] += avgmom_single[nx];
    avgmom_single[nx] = avgmom_single[0];
}

//==============================================================================
//==============================================================================


/*
step HO values - pos, vel, cell, dens and stress -> shift future values to current position and then extrapolate new current values to new future position
*/
void Species::step()
{
    for (int p = 0; p < np; ++p)
    {
        // transfer future values to new current values for pos, vel and cell
        pos[p]  = pos_new[p];
        vel[p]  = vel_new[p];

        // project new current values to new future values for pos, vel and cell
        pos_new[p]  = pos[p];
        vel_new[p]  = vel[p];

        // project future values to new current values for density and stress
        for (int i = 0; i < nx; ++i)
        {
            dens[2*i + 0]   = dens[2*i + 1];
            stress[2*i + 0] = stress[2*i + 1];

            dens_single[2*i + 0]   = dens_single[2*i + 1];
        }
        for (int i = 0; i < nx + 1; ++i)
        {
            mom[2*i + 0]    = mom[2*i + 1];
        }
    }
}

// getters
double* Species::get_dens_ptr()
{
return dens;
}

double* Species::get_mom_ptr()
{
return mom;
}

double* Species::get_avgmom_ptr()
{
return avgmom;
}

double* Species::get_nstress_ptr()
{
return nstress;
}

double* Species::get_gamma_ptr()
{
return gamma;
}

void Species::clear_files()
{
std::ofstream cont_file("./output/ho_" + name +"_continuity.txt", std::ofstream::out | std::ofstream::trunc);
cont_file.close();
std::ofstream dens_file("./output/ho_" + name +"_dens.txt", std::ofstream::out | std::ofstream::trunc);
dens_file.close();
std::ofstream mom_file("./output/ho_" + name +"_mom.txt", std::ofstream::out | std::ofstream::trunc);
mom_file.close();
std::ofstream avgmom_file("./output/ho_" + name +"_avgmom.txt", std::ofstream::out | std::ofstream::trunc);
avgmom_file.close();
std::ofstream stress_file("./output/ho_" + name +"_stress.txt", std::ofstream::out | std::ofstream::trunc);
stress_file.close();
std::ofstream gamma_file("./output/ho_" + name +"_gamma.txt", std::ofstream::out | std::ofstream::trunc);
gamma_file.close();
std::ofstream pos_file("./output/ho_" + name +"_pos.txt", std::ofstream::out | std::ofstream::trunc);
pos_file.close();
std::ofstream vel_file("./output/ho_" + name +"_vel.txt", std::ofstream::out | std::ofstream::trunc);
vel_file.close();
}

void Species::print_vals(const bool& particles_flag, const int& k) {
// do not calculate continuity for initial timestep as there is no valid avgmom
// analysis data
for (int i = 0; i < nx; ++i)
{
    continuity_res_single[i] = 0.0;
}

for (int i = 0; i < nx; ++i)
{
    continuity_res_single[i] = (dens_single[2*i + 1] - dens_single[2*i + 0])/dt;
    continuity_res_single[i] += (avgmom_single[i + 1] - avgmom_single[i])/dx;
}

// continuity
std::ofstream cont_file("./output/ho_" + name +"_continuity.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx; i++) {
    cont_file.width(15);
    cont_file << continuity_res_single[i] << "\t";
}
cont_file << std::endl;
cont_file.close();

// density
std::ofstream dens_file("./output/ho_" + name +"_dens.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx; i++) {
    dens_file.width(15);
    dens_file << dens[2*i + 1] << "\t";
}
dens_file << std::endl;
dens_file.close();

// momentum
std::ofstream mom_file("./output/ho_" + name +"_mom.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx + 1; i++) {
    mom_file.width(15);
    mom_file << mom[2*i + 1] << "\t";
}
mom_file << std::endl;
mom_file.close();

// avg momentum
std::ofstream avgmom_file("./output/ho_" + name +"_avgmom.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx + 1; i++) {
    avgmom_file.width(15);
    avgmom_file << avgmom[i] << "\t";
}
avgmom_file << std::endl;
avgmom_file.close();

// stress
std::ofstream stress_file("./output/ho_" + name +"_stress.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx; i++) {
    stress_file.width(15);
    stress_file << stress[2*i + 1] << "\t";
}
stress_file << std::endl;
stress_file.close();

// gamma
std::ofstream gamma_file("./output/ho_" + name +"_gamma.txt", std::ofstream::out | std::ofstream::app);
for (int i = 0; i < nx; i++) {
    gamma_file.width(15);
    gamma_file << gamma[i] << "\t";
}
gamma_file << std::endl;
gamma_file.close();

if (particles_flag == true)
{
    // pos
    std::ofstream pos_file("./output/ho_" + name +"_pos.txt", std::ofstream::out | std::ofstream::app);
    for (int p = 0; p < np; p++) {
    pos_file.width(15);
    pos_file << pos_new[p] << "\t";
    }
    pos_file << std::endl;
    pos_file.close();

    // vel
    std::ofstream vel_file("./output/ho_" + name +"_vel.txt", std::ofstream::out | std::ofstream::app);
    for (int p = 0; p < np; p++) {
    vel_file.width(15);
    vel_file << vel_new[p] << "\t";
    }
    vel_file << std::endl;
    vel_file.close();
}
}

void Species::print_methods_tracker()
{
for (int i = 0; i < 4; ++i)
{
    std::cout << "Method " << i << ": " << methods_tracker[i] << std::endl;
}
std::cout << "Avg no. substeps: " << total_substeps / (long int)(np * 100) << std::endl;
}