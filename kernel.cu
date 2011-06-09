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

#define timed_LIN3D(i,j,k,time,grid) (time * grid.nIxyz + \
                                      ((k) - grid.Imin.z) * grid.nIxy + \
                                      ((j) - grid.Imin.y) * grid.nIstep.x + \
                                      ((i) - grid.Imin.x))

//__constant__ int4 detLoc[MAX_DETECTORS];
//__constant__ float4 tissueProp[MAX_TISSUES];
__constant__ Simulation s;
__constant__ GPUMemory g;

// TODO: do away with the first argument.
__device__ void henyey_greenstein(float *t, float gg, char tissueIndex, int photonIndex, int n_photons, float3 *d)
{
    float3 d0;
    float rand;
    float foo;
    float theta, stheta, ctheta;
    float phi, sphi, cphi;

    // TODO: study more closely the random functions.
    rand = rand_next_aangle(t);
    phi = 2.0 * PI * rand;
    sincosf(phi, &sphi, &cphi);

    rand = rand_next_zangle(t);

    if(gg > EPS) {
        foo = (1.0 - gg * gg) / (1.0 - gg + 2.0 * gg * rand);
        foo *= foo;
        foo = (1.0 + gg * gg - foo) / (2.0 * gg);
        theta = acosf(foo);
        stheta = sinf(theta);
        ctheta = foo;
    } else {  // If g is exactly zero, then use isotropic scattering angle
        theta = 2.0 * PI * rand;
        sincosf(theta, &stheta, &ctheta);
    }

    /*
    if(theta > 0)
        g.momTiss[LIN2D(photonIndex, tissueIndex, n_photons)] += 1 - ctheta;
    */

    d0.x = d->x;
    d0.y = d->y;
    d0.z = d->z;
    if( d->z < 1.0 && d->z > -1.0 ) {
        d->x = stheta * (d0.x*d0.z*cphi - d0.y*sphi) * rsqrtf(1.0 - d0.z*d0.z) + d0.x * ctheta;
        d->y = stheta * (d0.y*d0.z*cphi + d0.x*sphi) * rsqrtf(1.0 - d0.z*d0.z) + d0.y * ctheta;
        d->z = -stheta * cphi * sqrtf(1.0 - d0.z*d0.z) + d0.z * ctheta;
    } else {
        d->x  = stheta * cphi;
        d->y  = stheta * sphi;
        d->z *= ctheta;
    }
}

__global__ void run_simulation(int photons_per_run)
{
    __shared__ int4 shm_detLoc[MAX_DETECTORS + MAX_TISSUES];
    float4 *shm_tissueProp = (float4 *) shm_detLoc + MAX_DETECTORS;

    // Loop index
    int i;

    int threadIndex = LIN2D(threadIdx.x, blockIdx.x, blockDim.x);

    char tissueIndex;   // tissue type of the current voxel
    int time;           // time elapsed since the photon was launched
    float step;
    float musr;

    // Random number generation
    float t[RAND_BUF_LEN], tnew[RAND_BUF_LEN];

    //if(threadIdx.x < MAX_TISSUES)
    //{
        shm_detLoc[threadIdx.x] = g.detLoc[threadIdx.x];
        shm_tissueProp[2*threadIdx.x] = g.tissueProp[2*threadIdx.x];
        shm_tissueProp[2*threadIdx.x + 1] = g.tissueProp[2*threadIdx.x + 1];
    //}
    __syncthreads();

    // Initialize the RNG
    gpu_rng_init(t, tnew, g.seed, threadIndex);

    int photons_run = 0;
    while(photons_run < photons_per_run)
    {
        int photonIndex = LIN2D(photons_run, photons_per_run, threadIndex);
        photons_run++;

        // Set the photon weight to 1 and initialize photon length parameters
        float P2pt = 1.0;   // photon weight
        float dist = 0.0;   // distance traveled so far by the photon 
        float Lnext = s.grid.minstepsize;
        float Lresid = 0.0;

        // Direction cosines of the photon
        float3 d;
        d.x = s.src.d.x; d.y = s.src.d.y; d.z = s.src.d.z;

        // Photon position (euclidean)
        float3 r;
        r.x = s.src.r.x; r.y = s.src.r.y; r.z = s.src.r.z;

        // Photon position (grid)
        int3 p;
        p.x = DIST2VOX(r.x, s.grid.stepr.x);
        p.y = DIST2VOX(r.y, s.grid.stepr.y);
        p.z = DIST2VOX(r.z, s.grid.stepr.z);


        // Loop until photon has exceeded its max distance allowed, or escapes
        // the grid.
        while( dist < s.max_length &&
               p.x >= 0 && p.x < s.grid.dim.x &&
               p.y >= 0 && p.y < s.grid.dim.y &&
               p.z >= 0 && p.z < s.grid.dim.z &&
               (tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim.x, s.grid.dim.y)]) != 0 )
        {
            rand_need_more(t, tnew);

            // Calculate scattering length
            Lresid = rand_next_scatlen(t);

            while( dist < s.max_length && Lresid > 0.0 &&
                   p.x >= 0 && p.x < s.grid.dim.x &&
                   p.y >= 0 && p.y < s.grid.dim.y &&
                   p.z >= 0 && p.z < s.grid.dim.z &&
                   (tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim.x, s.grid.dim.y)]) != 0 )
            {
                if(dist > Lnext && dist > s.min_length)
                {
                    time = (int) ((dist - s.min_length) * s.stepLr);

                    if( p.x >= s.grid.Imin.x && p.x <= s.grid.Imax.x &&
                        p.y >= s.grid.Imin.y && p.y <= s.grid.Imax.y &&
                        p.z >= s.grid.Imin.z && p.z <= s.grid.Imax.z &&
                        time < s.max_time )
                         g.II[timed_LIN3D(p.x, p.y, p.z, time, s.grid)] += P2pt;

                    Lnext += s.grid.minstepsize;
                }

                musr = shm_tissueProp[tissueIndex].x;
                step = Lresid * musr;
                // If scattering length is likely within a voxel, jump inside one voxel
                if(s.grid.minstepsize > step) {
                    Lresid = 0.0;
                } else {   // If scattering length is bigger than a voxel, then move 1 voxel
                    step = s.grid.minstepsize;
                    Lresid -= musr * s.grid.minstepsize;
                }

                r.x += d.x * step;
                r.y += d.y * step;
                r.z += d.z * step;
                dist += step;

                P2pt *= expf(-(shm_tissueProp[tissueIndex].y) * step);
                //g.lenTiss[LIN2D(photonIndex, tissueIndex, s.n_photons)] += step;

                p.x = DIST2VOX(r.x, s.grid.stepr.x);
                p.y = DIST2VOX(r.y, s.grid.stepr.y);
                p.z = DIST2VOX(r.z, s.grid.stepr.z);
            } // Propagate photon

            // Calculate the new scattering angle using henyey-greenstein
            if(tissueIndex != 0)
                henyey_greenstein(t, shm_tissueProp[tissueIndex].z, tissueIndex, photonIndex, s.n_photons, &d);
        } // loop until end of single photon

        // Score exiting photon and save history files
        p.x = DIST2VOX(r.x, s.grid.stepr.x);
        p.y = DIST2VOX(r.y, s.grid.stepr.y);
        p.z = DIST2VOX(r.z, s.grid.stepr.z);

        if ( p.x >= 0 && p.x < s.grid.dim.x &&
             p.y >= 0 && p.y < s.grid.dim.y &&
             p.z >= 0 && p.z < s.grid.dim.z )
        {
            tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim.x, s.grid.dim.y)];
            if( tissueIndex == 0 )
            {
                time = (int) ((dist - s.min_length) * s.stepLr);
                if( p.x >= s.grid.Imin.x && p.x <= s.grid.Imax.x &&
                    p.y >= s.grid.Imin.y && p.y <= s.grid.Imax.y &&
                    p.z >= s.grid.Imin.z && p.z <= s.grid.Imax.z &&
                    time < s.max_time )
                    g.II[timed_LIN3D(p.x, p.y, p.z, time, s.grid)] -= P2pt;

                // Loop through number of detectors
                // Did the photon hit a detector?
                for( i = 0; i < s.det.num; i++ )
                    if( abs(p.x - shm_detLoc[i].x) <= shm_detLoc[i].w &&
                        abs(p.y - shm_detLoc[i].y) <= shm_detLoc[i].w &&
                        abs(p.z - shm_detLoc[i].z) <= shm_detLoc[i].w )
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
    float x0, y0, z0;

    // Source's position (euclidean).
    x0 = sim->src.r.x;
    y0 = sim->src.r.y;
    z0 = sim->src.r.z;

    i = DIST2VOX(x0, sim->grid.stepr.x);
    j = DIST2VOX(y0, sim->grid.stepr.y);
    k = DIST2VOX(z0, sim->grid.stepr.z);

    tissueIndex = sim->grid.tissueType[i][j][k];

    while( tissueIndex != 0 &&
           i > 0 && i < sim->grid.dim.x &&
           j > 0 && j < sim->grid.dim.y &&
           k > 0 && k < sim->grid.dim.z )
    {
        x0 -= sim->src.d.x * sim->grid.minstepsize;
        y0 -= sim->src.d.y * sim->grid.minstepsize;
        z0 -= sim->src.d.z * sim->grid.minstepsize;
        i = DIST2VOX(x0, sim->grid.stepr.x);
        j = DIST2VOX(y0, sim->grid.stepr.y);
        k = DIST2VOX(z0, sim->grid.stepr.z);
        tissueIndex = sim->grid.tissueType[i][j][k];
    }
    while( tissueIndex == 0 )
    {
        x0 += sim->src.d.x * sim->grid.minstepsize;
        y0 += sim->src.d.y * sim->grid.minstepsize;
        z0 += sim->src.d.z * sim->grid.minstepsize;
        i = DIST2VOX(x0, sim->grid.stepr.x);
        j = DIST2VOX(y0, sim->grid.stepr.y);
        k = DIST2VOX(z0, sim->grid.stepr.z);
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
    int photons_per_thread = sim.n_photons / conf.n_threads;
    run_simulation<<< conf.n_blocks, conf.n_threads_per_block >>>(photons_per_thread);
    printf("photons per thread = %d\n", photons_per_thread);

    // Make sure all photons have already been simulated before moving on.
    cudaThreadSynchronize();
}
