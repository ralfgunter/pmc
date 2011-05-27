/********************************************************************************
*          Monte-Carlo Simulation for Light Transport in 3D Volumes             *
*********************************************************************************
*                                                                               *
* Copyright (C) 2002-2008,  David Boas    (dboas <at> nmr.mgh.harvard.edu)      *
*               2008        Jay Dubb      (jdubb <at> nmr.mgh.harvard.edu)      *
*               2008        Qianqian Fang (fangq <at> nmr.mgh.harvard.edu)      *
*               2011        Ralf Gunter   (ralfgunter <at> gmail.com)           *
*                                                                               *
* License:  4-clause BSD License, see LICENSE for details                       *
*                                                                               *
* Example:                                                                      *
*         tMCimg input.inp                                                      *
*                                                                               *
* Please find more details in README and doc/HELP                               *
********************************************************************************/

#include "main.h"
#include "logistic_rand_kernel.h"
#include "bitset2d_kernel.h"

// TODO: do away with the first argument.
__device__ void henyey_greenstein(Real *t, Real *tg, float *momTiss, char tissueIndex, float3 *d)
{
    float3 d0;

    Real rand;
    Real foo;
    Real theta, stheta, ctheta;
    Real phi, sphi, cphi;
    Real gg;

    gg = tg[tissueIndex];

    // TODO: study more closely the random functions.
    rand = rand_next_aangle(t);
    phi = 2.0 * PI * rand;
    sincosf(phi, &sphi, &cphi);

    rand = rand_next_zangle(t);

    if(gg > EPS) {
        foo = (1.0 - gg * gg) / (1.0 - gg + 2.0 * gg * rand);
        foo *= foo;
        foo = (1.0 + gg * gg - foo) / (2.0 * gg);
        theta = acos(foo);
        stheta = sin(theta);
        ctheta = foo;
    } else {  // If g is exactly zero, then use isotropic scattering angle
        theta = 2.0 * PI * rand;
        sincosf(theta, &stheta, &ctheta);
    }

    if(theta > 0)
        momTiss[LIN3D(threadIdx.x,blockIdx.x,tissueIndex,blockDim.x,gridDim.x)] += 1 - ctheta;

    d0.x = d->x;
    d0.y = d->y;
    d0.z = d->z;
    if( d->z < 1.0 && d->z > -1.0 ) {
        d->x = stheta * (d0.x*d0.z*cphi - d0.y*sphi) / sqrt(1 - d0.z*d0.z) + d0.x * ctheta;
        d->y = stheta * (d0.y*d0.z*cphi + d0.x*sphi) / sqrt(1 - d0.z*d0.z) + d0.y * ctheta;
        d->z = -stheta * cphi * sqrt(1 - d0.z*d0.z) + d0.z * ctheta;
    } else {
        d->x  = stheta * cphi;
        d->y  = stheta * sphi;
        d->z *= ctheta;
    }
}

__global__ void run_simulation(GPUMemory g, Simulation s)
{
    // Loop index
    int i;

    // Random number generation
    Real t[RAND_BUF_LEN], tnew[RAND_BUF_LEN];

    char tissueIndex;   // tissue type of the current voxel
    int time;           // time elapsed since the photon was launched
    Real step;
    Real musr;
    int photonIndex = LIN2D(threadIdx.x, blockIdx.x, blockDim.x);

    // Set the photon weight to 1 and initialize photon length parameters
    Real P2pt = 1.0;   // photon weight
    Real dist = 0.0;   // distance traveled so far by the photon 
    Real Lnext = s.grid.minstepsize;
    Real Lresid = 0.0;

    // Initialize the RNG
    gpu_rng_init(t, tnew, g.seed, photonIndex);

    // Direction cosines of the photon
    float3 d = s.src.d;

    // Photon position (euclidean)
    float3 r = s.src.r;

    // Photon position (grid)
    int3 p;
    // TODO: The *step are also used only for division operations;
    //       Find out if they should receive the same treatment as stepLr.
    p.x = DIST2VOX(r.x, s.grid.xstep);
    p.y = DIST2VOX(r.y, s.grid.ystep);
    p.z = DIST2VOX(r.z, s.grid.zstep);

    // Loop until photon has exceeded its max distance allowed, or escapes
    // the grid.
    while ( dist < s.max_length &&
            p.x >= 0 && p.x < s.grid.dim_x &&
            p.y >= 0 && p.y < s.grid.dim_y &&
            p.z >= 0 && p.z < s.grid.dim_z &&
            (tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim_x, s.grid.dim_y)]) != 0 ) {

        // Calculate scattering length
        rand_need_more(t, tnew);
        Lresid = rand_next_scatlen(t);

        while( dist < s.max_length && Lresid > 0.0 &&
               p.x >= 0 && p.x < s.grid.dim_x &&
               p.y >= 0 && p.y < s.grid.dim_y &&
               p.z >= 0 && p.z < s.grid.dim_z &&
               (tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim_x, s.grid.dim_y)]) != 0 ) {

            /*
            g.II[10] = dist;
            g.II[11] = time;
            g.II[12] = Lnext;
            */

            if(dist > Lnext && dist > s.min_length)
            {
                time = (int) ((dist - s.min_length) * s.stepLr);
                //g.II[11] = time;
                if ( p.x >= s.grid.Ixmin && p.x <= s.grid.Ixmax &&
                     p.y >= s.grid.Iymin && p.y <= s.grid.Iymax &&
                     p.z >= s.grid.Izmin && p.z <= s.grid.Izmax &&
                     time < s.max_time)
                {
                    g.II[LIN3D(p.x,p.y,p.z,s.grid.dim_x,s.grid.dim_y)] += P2pt;
                }
                Lnext += s.grid.minstepsize;
            }

            musr = g.tmusr[tissueIndex];
            step = Lresid * musr;
            // If scattering length is likely within a voxel, jump inside one voxel
            if(s.grid.minstepsize > step) {
                Lresid = 0;
            } else {   // If scattering length is bigger than a voxel, then move 1 voxel
                step = s.grid.minstepsize;
                Lresid -= musr * s.grid.minstepsize;
            }

            r.x += d.x * step;
            r.y += d.y * step;
            r.z += d.z * step;
            dist += step;

            P2pt *= exp(-(g.tmua[tissueIndex]) * step);
            g.lenTiss[LIN2D(photonIndex, tissueIndex, blockDim.x * gridDim.x)] += step;

            /*
            g.II[0] = p.x;
            g.II[1] = p.y;
            g.II[2] = p.z;
            g.II[3] = d.x;
            g.II[4] = d.y;
            g.II[5] = d.z;
            g.II[6] = musr;
            g.II[7] = tissueIndex;
            g.II[8] = photonIndex;
            g.II[9] = Lresid;
            */

            p.x = DIST2VOX(r.x, s.grid.xstep);
            p.y = DIST2VOX(r.y, s.grid.ystep);
            p.z = DIST2VOX(r.z, s.grid.zstep);
        } // Propagate photon

        // Calculate the new scattering angle using henyey-greenstein
        if(tissueIndex) henyey_greenstein(t, g.tg, g.momTiss, tissueIndex, &d);
    } // loop until end of single photon

    // Score exiting photon and save history files
    p.x = DIST2VOX(r.x, s.grid.xstep);
    p.y = DIST2VOX(r.y, s.grid.ystep);
    p.z = DIST2VOX(r.z, s.grid.zstep);

    /*
    g.II[0] = p.x;
    g.II[1] = p.y;
    g.II[2] = p.z;
    g.II[3] = photonIndex;
    g.II[4] = -1;
    */

    if ( p.x >= 0 && p.x < s.grid.dim_x &&
         p.y >= 0 && p.y < s.grid.dim_y &&
         p.z >= 0 && p.z < s.grid.dim_z )
    {
        tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim_x, s.grid.dim_y)];
        //g.II[4] = tissueIndex;
        if( tissueIndex == 0 )
        {
            time = (int) ((dist - s.min_length) * s.stepLr);
            if( p.x >= s.grid.Ixmin && p.x <= s.grid.Ixmax &&
                p.y >= s.grid.Iymin && p.y <= s.grid.Iymax &&
                p.z >= s.grid.Izmin && p.z <= s.grid.Izmax &&
                time < s.max_time )
                g.II[LIN3D(p.x,p.y,p.z,s.grid.dim_x,s.grid.dim_y)] -= P2pt;

            // Loop through number of detectors
            // Did the photon hit a detector?
            for( i = 0; i < s.det.num; i++ )
                if( abs(p.x - g.detLoc[i].x) <= s.det.radius &&
                    abs(p.y - g.detLoc[i].y) <= s.det.radius &&
                    abs(p.z - g.detLoc[i].z) <= s.det.radius ) {
                    //g.II[5] = 123;
                    gpu_set(g.detHit, photonIndex, i);
                }
        }
    }
}

// Make sure the source is at an interface.
void correct_source(Simulation *sim)
{
    char tissueIndex;
    int i, j, k;
    Real x0, y0, z0;

    // Source's position (euclidean).
    x0 = sim->src.r.x;
    y0 = sim->src.r.y;
    z0 = sim->src.r.z;

    i = DIST2VOX(x0, sim->grid.xstep);
    j = DIST2VOX(y0, sim->grid.ystep);
    k = DIST2VOX(z0, sim->grid.zstep);

    tissueIndex = sim->grid.tissueType[i][j][k];

    while( tissueIndex != 0 &&
           i > 0 && i < sim->grid.dim_x &&
           j > 0 && j < sim->grid.dim_y &&
           k > 0 && k < sim->grid.dim_z )
    {
        x0 -= sim->src.d.x * sim->grid.minstepsize;
        y0 -= sim->src.d.y * sim->grid.minstepsize;
        z0 -= sim->src.d.z * sim->grid.minstepsize;
        i = DIST2VOX(x0, sim->grid.xstep);
        j = DIST2VOX(y0, sim->grid.ystep);
        k = DIST2VOX(z0, sim->grid.zstep);
        tissueIndex = sim->grid.tissueType[i][j][k];
    }
    while( tissueIndex == 0 )
    {
        x0 += sim->src.d.x * sim->grid.minstepsize;
        y0 += sim->src.d.y * sim->grid.minstepsize;
        z0 += sim->src.d.z * sim->grid.minstepsize;
        i = DIST2VOX(x0, sim->grid.xstep);
        j = DIST2VOX(y0, sim->grid.ystep);
        k = DIST2VOX(z0, sim->grid.zstep);
        tissueIndex = sim->grid.tissueType[i][j][k];
    }

    // Update the source coordinates 
    sim->src.r.x = x0;
    sim->src.r.y = y0;
    sim->src.r.z = z0;
}

void simulate(ExecConfig conf, Simulation sim, GPUMemory gmem)
{
    // TODO: optimize the number of blocks/threads per block.
    // FIXME: as things stand, the kernel will most likely simulate too
    //        many photons; do something about it.
    run_simulation <<< conf.n_blocks, conf.n_threads_per_block >>>(gmem, sim);

    // Make sure all photons have already been simulated before moving on.
    cudaThreadSynchronize();
}
