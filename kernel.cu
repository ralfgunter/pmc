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
********************************************************************************/

#include "main.h"
#include "logistic_rand_kernel.h"
#include "bitset2d_kernel.h"

#define MOVE(p, r, stepr) \
        (p).x = (r).x * (stepr).x; \
        (p).y = (r).y * (stepr).y; \
        (p).z = (r).z * (stepr).z

//__constant__ int4 detLoc[MAX_DETECTORS];
//__constant__ float4 tissueProp[MAX_TISSUES];
__constant__ Simulation s;
__constant__ GPUMemory g;

// TODO: do away with the first argument.
__device__ void henyey_greenstein(float *t, float gg, uint8_t tissueIndex, uint32_t photonIndex, float3 *d)
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

    if(theta > 0)
        g.momTiss[MAD_HASH((photonIndex << 5) | tissueIndex)] += 1 - ctheta;

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

__global__ void run_simulation(uint32_t *seed, int photons_per_thread, int iteration)
{
    __shared__ int4 detLoc[MAX_DETECTORS + MAX_TISSUES];
    float4 *tissueProp = (float4 *) detLoc + MAX_DETECTORS;

    // Loop index
    int i;

    uint32_t threadIndex = LIN2D(threadIdx.x, blockIdx.x, blockDim.x);

    uint8_t tissueIndex;   // tissue type of the current voxel
    int time;            // time elapsed since the photon was launched
    float step;
    float musr;

    // Random number generation
    float t[RAND_BUF_LEN], tnew[RAND_BUF_LEN];

    detLoc[threadIdx.x] = g.detLoc[threadIdx.x];
    tissueProp[2*threadIdx.x] = g.tissueProp[2*threadIdx.x];
    tissueProp[2*threadIdx.x + 1] = g.tissueProp[2*threadIdx.x + 1];
    __syncthreads();

    gpu_rng_init(t, tnew, seed, threadIndex);

    int photons_run = 0;
    while(photons_run < photons_per_thread)
    {
        uint32_t photonIndex = LIN3D(photons_run, threadIndex, iteration, photons_per_thread, (blockDim.x * gridDim.x));
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
        MOVE(p, r, s.grid.stepr);

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

            while( Lresid > 0.0 && dist < s.max_length &&
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
                        time < s.num_time_steps )
                        g.II[LIN(p.x, p.y, p.z, time, s.grid)] += P2pt;

                    Lnext += s.grid.minstepsize;
                }

                musr = tissueProp[tissueIndex].x;
                step = Lresid * musr;
                // If scattering length is likely within a voxel, jump inside one voxel
                if(s.grid.minstepsize > step) {
                    Lresid = 0.0;
                } else {   // If scattering length is bigger than a voxel, then move one voxel
                    step = s.grid.minstepsize;
                    Lresid -= musr * s.grid.minstepsize;
                }

                r.x += d.x * step;
                r.y += d.y * step;
                r.z += d.z * step;
                dist += step;

                P2pt *= expf(-(tissueProp[tissueIndex].y) * step);
                // FIXME: on 32-bits cards, this only works with up to
                //        (2^5 - 1) tissue types (indexed from 1) and
                //        2^27 photons (indexed from 0).
                g.lenTiss[MAD_HASH((photonIndex << 5) | tissueIndex)] += step;

                MOVE(p, r, s.grid.stepr);
            } // Propagate photon

            // Calculate the new scattering angle using henyey-greenstein
            if(tissueIndex != 0)
                henyey_greenstein(t, tissueProp[tissueIndex].z, tissueIndex, photonIndex, &d);
        } // loop until end of single photon

        // Score exiting photon and save history files
        MOVE(p, r, s.grid.stepr);

        if ( p.x >= 0 && p.x < s.grid.dim.x &&
             p.y >= 0 && p.y < s.grid.dim.y &&
             p.z >= 0 && p.z < s.grid.dim.z )
        {
            tissueIndex = g.tissueType[LIN3D(p.x, p.y, p.z, s.grid.dim.x, s.grid.dim.y)];
            if(tissueIndex == 0)
            {
                time = (int) ((dist - s.min_length) * s.stepLr);
                if( p.x >= s.grid.Imin.x && p.x <= s.grid.Imax.x &&
                    p.y >= s.grid.Imin.y && p.y <= s.grid.Imax.y &&
                    p.z >= s.grid.Imin.z && p.z <= s.grid.Imax.z &&
                    time < s.num_time_steps )
                    g.II[LIN(p.x, p.y, p.z, time, s.grid)] -= P2pt;

                // Loop through number of detectors
                // Did the photon hit a detector?
                for( i = 0; i < s.det.num; i++ )
                    if( absf(p.x - detLoc[i].x) <= detLoc[i].w &&
                        absf(p.y - detLoc[i].y) <= detLoc[i].w &&
                        absf(p.z - detLoc[i].z) <= detLoc[i].w )
                        gpu_set(g.detHit, photonIndex, i);
            }
        }
    }
}

// Make sure the source is at an interface.
void correct_source(Simulation *sim)
{
    uint8_t tissueIndex;
    int3 p;
    float3 r0;

    // Source's position (euclidean).
    r0.x = sim->src.r.x; r0.y = sim->src.r.y; r0.z = sim->src.r.z;

    MOVE(p, r0, sim->grid.stepr);

    tissueIndex = sim->grid.tissueType[p.x][p.y][p.z];

    while( tissueIndex != 0 &&
           p.x > 0 && p.x < sim->grid.dim.x &&
           p.y > 0 && p.y < sim->grid.dim.y &&
           p.z > 0 && p.z < sim->grid.dim.z )
    {
        r0.x -= sim->src.d.x * sim->grid.minstepsize;
        r0.y -= sim->src.d.y * sim->grid.minstepsize;
        r0.z -= sim->src.d.z * sim->grid.minstepsize;
        MOVE(p, r0, sim->grid.stepr);
        tissueIndex = sim->grid.tissueType[p.x][p.y][p.z];
    }
    while( tissueIndex == 0 &&
           p.x > 0 && p.x < sim->grid.dim.x &&
           p.y > 0 && p.y < sim->grid.dim.y &&
           p.z > 0 && p.z < sim->grid.dim.z )
    {
        r0.x += sim->src.d.x * sim->grid.minstepsize;
        r0.y += sim->src.d.y * sim->grid.minstepsize;
        r0.z += sim->src.d.z * sim->grid.minstepsize;
        MOVE(p, r0, sim->grid.stepr);
        tissueIndex = sim->grid.tissueType[p.x][p.y][p.z];
    }

    // Update the source coordinates 
    sim->src.r.x = r0.x;
    sim->src.r.y = r0.y;
    sim->src.r.z = r0.z;
}

void simulate(ExecConfig conf, Simulation sim, GPUMemory gmem)
{
    uint32_t seed;
    uint32_t *temp_seed, *d_seed;
    int photons_per_iteration = sim.n_photons / conf.n_iterations;
    int photons_per_thread = photons_per_iteration / conf.n_threads;
    int iteration = 0;

    printf("photons per thread = %d\n", photons_per_thread);
    printf("photons per iteration = %d\n", photons_per_iteration);

    seed = conf.rand_seed;
    d_seed = gmem.seed;
    for(iteration = 0; iteration < conf.n_iterations; iteration++)
    {
        run_simulation<<< conf.n_blocks, 128 >>>(d_seed, photons_per_thread, iteration);

        // Order a new batch of RNG seeds while the current iteration is being simulated.
        temp_seed = init_rand_seed(seed++, conf);

        // Make sure all photons have already been simulated before moving on.
        cudaThreadSynchronize();

        cudaFree(d_seed);
        d_seed = temp_seed;
    }
}
