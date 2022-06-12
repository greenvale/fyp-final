typedef struct params_tag
{
  int nx;
  float lx;
  float dt;
  float charge;
  float mass;
  float weight;
  int accelerate;
  float pic_tol;
} params;


kernel
void push_gpu
(
       const params par,
global const float* elec_d,
global const float* pos_d,
global const float* vel_d,
global       float* avgmom_expanded_d,
global       float* pos_new_d,
global       float* vel_new_d,
local        float* avgmom_local,
local        float* avgmom_local_expanded
)
{
  /* work-item variables */
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);

  int global_size = get_global_size(0);
  int local_size = get_local_size(0);
  int num_groups = get_num_groups(0);

  /* runtime variables */
  int   sub_flag;
  int   sub_index;
  int   method_flag;
  int   pic_flag;
  int   pic_index;

  float subt;
  float subpos;
  float subvel;
  int   subcell;
  float subpos_new;
  float subvel_new;
  int   subcell_new;

  float accel;
  float dsubt;
  float dsubpos;
  float dsubvel;
  int   dsubcell = 0;
  float dsubpos_lh;
  float dsubpos_rh;

  float shape_lhface;
  float shape_rhface;
  float davgmom_lh;
  float davgmom_rh;
  float davgmom_lh_tmp;
  float davgmom_rh_tmp;

  float discrim;
  float dsubt_soln1;
  float dsubt_soln2;
  float tmp;
  int subcell_new_tmp;

  float dx = par.lx / (float) par.nx;

  /* ========================================================= */

  for (int i = 0; i < par.nx + 1; ++i)
  {
    avgmom_local_expanded[local_id * (par.nx + 1) + i] = 0.0;
  }

  sub_flag = 0;
  sub_index = -1;

  subt    = 0.0;
  subpos  = pos_d[global_id];
  subvel  = vel_d[global_id];
  subcell = (int) floor(subpos / dx);

  while (sub_flag == 0)
  {
    sub_index++;

    dsubpos_lh = ((float) subcell * dx) - subpos;
    dsubpos_rh = dsubpos_lh + dx;

    /* ========================================================= */
    /* ACCELERATOR */
    if ((par.accelerate > 0) && (dsubcell != 0) && (sub_index > 0))
    {
      shape_lhface = 0.5;
      shape_rhface = 0.5;
      dsubpos = (float) dsubcell * dx;
      accel = (elec_d[2*subcell + 0] + elec_d[2*subcell + 1]) * shape_lhface;
      accel += (elec_d[2*subcell + 2 + 0] + elec_d[2*subcell + 2 + 1]) * shape_rhface;
      accel *= 0.5 * par.charge / par.mass;

      /* ========================================================= */
      /* DIRECT METHOD */
      if (par.accelerate == 1)
      {
        discrim = subvel*subvel + 2.0*accel*dsubpos;
        if (discrim < 0.0)
        {
          method_flag = 3;
          discrim = 0.0;
        }
        /*dsubt_soln1 = (-subvel - sqrt(discrim)) / accel;
        dsubt_soln2 = (-subvel + sqrt(discrim)) / accel;*/

        tmp = (subvel >= 0.0) - (subvel < 0.0);
        dsubt_soln2 = -0.5 * (subvel + tmp * sqrt(discrim));
        dsubt_soln1 = dsubt_soln2 / (0.5 * accel);
        dsubt_soln2 = (-dsubpos) / dsubt_soln2;

        if ((dsubt_soln1 > 0.0) && (method_flag != 3))
        {
          dsubt = dsubt_soln1;
          method_flag = 2;
        }
        if ((dsubt_soln2 > 0.0) && (method_flag != 3) && (dsubt_soln2 < dsubt_soln1))
        {
          dsubt = dsubt_soln2;
          method_flag = 2;
        }
        if ((method_flag == 2) && (dsubt + subt <= par.dt))
        {
          dsubvel = accel * dsubt;
        }
        else
        {
          method_flag = 0;
        }
      }
      /* ========================================================= */
      /* ADAPTIVE METHOD */
      else if (par.accelerate == 2)
      {
        tmp = 0.0;

        for (int i = 0; i < 5; ++i)
        {
          dsubt = tmp;
          tmp = dsubpos / (subvel + 0.5*dsubt*accel);
        }

        method_flag = 2*((tmp > 0.0) && (fabs(dsubt - tmp) < par.pic_tol) && (dsubt + subt <= par.dt));
        dsubt = tmp;
        dsubvel = accel * dsubt;
      }
    }
    else
    {
      method_flag = 0;
    }

    /* ========================================================= */
    /* ADAPTIVE PUSH */

    if (method_flag == 0)
    {
      dsubcell = 0;

      dsubt = (subvel != 0.0)*(dx / fabs(subvel)) + (subvel == 0.0)*par.dt;

      dsubt = ((dsubt + subt) <= par.dt)*(dsubt) + ((dsubt + subt) > par.dt)*(par.dt - subt);

      dsubpos = dsubt * subvel;

      /* solve dsubpos with Picard loop */
      pic_flag = 0;
      for (int i = 0; i < 10; ++i)
      {
        shape_lhface = 1.0 - fabs((subpos + 0.5 * dsubpos)/dx - (float)(subcell));
        shape_rhface = 1.0 - fabs((subpos + 0.5 * dsubpos)/dx - (float)(subcell + 1));
        shape_lhface *= (shape_lhface >= 0.0);
        shape_rhface *= (shape_rhface >= 0.0);

        accel = (elec_d[2*subcell + 0] + elec_d[2*subcell + 1]) * shape_lhface;
        accel += (elec_d[2*subcell + 2 + 0] + elec_d[2*subcell + 2 + 1]) * shape_rhface;
        accel *= 0.5 * par.charge / par.mass;
        dsubvel = dsubt * accel;
        tmp = dsubt * (subvel + 0.5 * dsubvel);

        pic_flag = (fabs(dsubpos - tmp) < par.pic_tol);
        pic_flag += (pic_flag == 0)*(pic_index > 10);

        dsubpos = tmp;
      }

      /* check if cell face crossed */
      dsubpos = dsubpos*((dsubpos > dsubpos_lh) && (dsubpos < dsubpos_rh)) + dsubpos_lh*(dsubpos <= dsubpos_lh) + dsubpos_rh*(dsubpos >= dsubpos_rh);
      method_flag = (dsubpos <= dsubpos_lh) + (dsubpos >= dsubpos_rh);
      dsubcell = -(dsubpos <= dsubpos_lh) + (dsubpos >= dsubpos_rh);

      /* make particle land on cell face */
      if (method_flag == 1)
      {
        shape_lhface = 1.0 - fabs((subpos + 0.5 * dsubpos)/dx - (float)(subcell));
        shape_rhface = 1.0 - fabs((subpos + 0.5 * dsubpos)/dx - (float)(subcell + 1));
        shape_lhface *= (shape_lhface >= 0.0);
        shape_rhface *= (shape_rhface >= 0.0);

        accel  = (elec_d[2*subcell + 0] + elec_d[2*subcell + 1]) * shape_lhface;
        accel += (elec_d[2*subcell + 2 + 0] + elec_d[2*subcell + 2 + 1]) * shape_rhface;
        accel *= 0.5 * par.charge / par.mass;

        /* solve dsubt with Picard loop */
        pic_flag = 0;
        for (int i = 0; i < 10; ++i)
        {
          tmp = dsubpos / (subvel + 0.5 * subt * accel);
          tmp *= (tmp > 0.0);
          tmp = tmp*(tmp + subt <= par.dt) + (par.dt - subt)*(tmp + subt > par.dt);

          pic_flag = (fabs(dsubt - tmp) < par.pic_tol);
          pic_flag += (pic_flag == 0)*(pic_index > 10);

          dsubt = tmp;
        }
        dsubvel = dsubt * accel;
      }
    }

    /* check if sub-cycle is finished */
    sub_flag = (subt + dsubt >= par.dt);

    /* davgmom contribution */
    davgmom_lh = (1.0 / (dx * par.dt)) * par.weight * (subvel + 0.5 * dsubvel) * shape_lhface * dsubt;
    davgmom_rh = (1.0 / (dx * par.dt)) * par.weight * (subvel + 0.5 * dsubvel) * shape_rhface * dsubt;

    avgmom_local_expanded[local_id * (par.nx + 1) + subcell]      += davgmom_lh;
    avgmom_local_expanded[local_id * (par.nx + 1) + subcell + 1]  += davgmom_rh;

    /* sub-step */
    subpos_new = subpos + dsubpos;
    subvel_new = subvel + dsubvel;
    subcell_new = subcell + dsubcell;

    subcell_new_tmp = subcell_new;
    subcell_new += ((subcell_new_tmp < 0) - (subcell_new_tmp > par.nx-1))*par.nx;
    subpos_new += ((subcell_new_tmp < 0) - (subcell_new_tmp > par.nx-1))*par.lx;

    subt += dsubt;
    subpos = subpos_new;
    subvel = subvel_new;
    subcell = subcell_new;
  }

  pos_new_d[global_id] = subpos_new;
  vel_new_d[global_id] = subvel_new;

  /* reduction of avgmom contribution */
  for (int i = 0; i < par.nx + 1; ++i)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    avgmom_local[i] = work_group_reduce_add(
      avgmom_local_expanded[local_id * (par.nx + 1) + i]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* transfer avgmom to expanded global */
  if (local_id == 0)
  {
    for (int i = 0; i < par.nx + 1; ++i)
    {
      avgmom_expanded_d[group_id * (par.nx + 1) + i] = avgmom_local[i];
    }
  }
}
