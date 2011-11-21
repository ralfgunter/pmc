/********************************************************************************
*          Monte-Carlo Simulation for Light Transport in 3D Volumes             *
*********************************************************************************
*                                                                               *
* Copyright (C) 2002-2008,  David Boas    (dboas <at> nmr.mgh.harvard.edu)      *
*               2008        Jay Dubb      (jdubb <at> nmr.mgh.harvard.edu)      *
*               2008        Qianqian Fang (fangq <at> nmr.mgh.harvard.edu)      *
*               2011        Ralf Gunter   (ralfgunter <at> gmail.com)           *
*                                                                               *
* License:  3-clause BSD License, see LICENSE for details                       *
*                                                                               *
********************************************************************************/

#include "main.h"

uint32_t* init_rand_seed(int seed, ExecConfig conf)
{
    uint32_t *h_seed, *d_seed;
    size_t sizeof_seed;

    if(seed > 0)
        srand(seed);
    else
        srand(time(NULL));

    // Seed used by the RNG.
    sizeof_seed = conf.n_threads * RAND_SEED_LEN * sizeof(uint32_t);
    h_seed = (uint32_t *) malloc(sizeof_seed);
    for(int i = 0; i < conf.n_threads * RAND_SEED_LEN; i++)
        h_seed[i] = rand();

    //DEV_ALLOC(d_seed, sizeof_seed);
    cutilSafeCall(cudaMalloc((void **) &d_seed, sizeof_seed));
    TO_DEVICE(d_seed, h_seed, sizeof_seed);

    free(h_seed);

    return d_seed;
}

void init_params_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem)
{
    uint8_t *h_linear_media_type, *d_media_type;
    int4 *d_det_loc;
    float4 *d_media_prop;

    // Calculate the total number of voxel elements.
    int grid_dim = sim->grid.dim.x * sim->grid.dim.y * sim->grid.dim.z;

    // Linearize media_type, as CUDA cannot handle pointers to pointers.
    h_linear_media_type = (uint8_t *) malloc(grid_dim * sizeof(uint8_t));
    linearize_3d(sim->grid.media_type, h_linear_media_type,
                 sim->grid.dim.x, sim->grid.dim.y, sim->grid.dim.z);

    // Allocate memory on the GPU global memory.
    DEV_ALLOC(&d_det_loc, MAX_DETECTORS * sizeof(int4));
    DEV_ALLOC(&d_media_prop, (MAX_TISSUES + 1) * sizeof(float4));
    DEV_ALLOC(&d_media_type, grid_dim * sizeof(uint8_t));

    // Copy simulation memory to the GPU.
    //cudaMemcpyToSymbol("det_loc", sim->det.info, sim->det.num * sizeof(int4));
    //cudaMemcpyToSymbol("media_prop", sim->tiss.prop, (sim->tiss.num + 1) * sizeof(float4));
    TO_DEVICE(d_det_loc, sim->det.info, MAX_DETECTORS * sizeof(int4));
    TO_DEVICE(d_media_prop, sim->tiss.prop, (MAX_TISSUES + 1) * sizeof(float4));
    TO_DEVICE(d_media_type, h_linear_media_type, grid_dim * sizeof(uint8_t));

    // Update GPU memory structure (so that its pointers can be used elsewhere).
    gmem->det_loc = d_det_loc;
    gmem->media_prop = d_media_prop;
    gmem->media_type = d_media_type;

    // Free temporary memory used on the host.
    free(h_linear_media_type);
}

void init_results_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem)
{
    float *d_fbox;
    float *d_path_length, *d_mom_transfer;
    float *d_temp_path_length, *d_temp_mom_transfer;
    float *h_temp_tissueArrays;
    int8_t *d_det_hit;
    size_t num_temp_tissueArrays, num_tissueArrays, num_fbox;

    // Setup the path length and momentum transfer arrays.
    //num_tissueArrays = (sim->tiss.num + 1) * sim->n_photons;
    num_tissueArrays = 1 << NUM_HASH_BITS;
    num_temp_tissueArrays = conf.n_threads * (sim->tiss.num + 1);
    sim->path_length  = (float *) calloc(num_tissueArrays, sizeof(float));
    sim->mom_transfer = (float *) calloc(num_tissueArrays, sizeof(float));
    h_temp_tissueArrays = (float *) calloc(num_temp_tissueArrays, sizeof(float));

    // Photon fluence.
    num_fbox = sim->grid.nIxyz * sim->num_time_steps;
    sim->fbox = (float *) calloc(num_fbox, sizeof(float));

    // Array of which photons hit which detectors (if any).
    sim->det.hit = (int8_t *) calloc(sim->n_photons, sizeof(int8_t));

    // Allocate memory on the GPU global memory.
    DEV_ALLOC(&d_path_length,  num_tissueArrays * sizeof(float));
    DEV_ALLOC(&d_mom_transfer, num_tissueArrays * sizeof(float));
    DEV_ALLOC(&d_fbox, num_fbox * sizeof(float));
    DEV_ALLOC(&d_det_hit, sim->n_photons * sizeof(int8_t));
    DEV_ALLOC(&d_temp_path_length, num_temp_tissueArrays * sizeof(float));
    DEV_ALLOC(&d_temp_mom_transfer, num_temp_tissueArrays * sizeof(float));

    // Copy simulation memory to the GPU.
    TO_DEVICE(d_path_length, sim->path_length, num_tissueArrays * sizeof(float));
    TO_DEVICE(d_mom_transfer, sim->mom_transfer, num_tissueArrays * sizeof(float));
    TO_DEVICE(d_fbox, sim->fbox, num_fbox * sizeof(float));
    TO_DEVICE(d_det_hit, sim->det.hit, sim->n_photons * sizeof(int8_t));
    TO_DEVICE(d_temp_path_length, h_temp_tissueArrays, num_temp_tissueArrays * sizeof(float));
    TO_DEVICE(d_temp_mom_transfer, h_temp_tissueArrays, num_temp_tissueArrays * sizeof(float));

    // Update GPU memory structure (so that its pointers can be used elsewhere).
    gmem->path_length = d_path_length;
    gmem->mom_transfer = d_mom_transfer;
    gmem->fbox = d_fbox;
    gmem->det_hit = d_det_hit;
    gmem->temp_path_length = d_temp_path_length;
    gmem->temp_mom_transfer = d_temp_mom_transfer;

    // Free temporary memory used on the host.
    free(h_temp_tissueArrays);
}

void copy_mem_symbols(Simulation *sim, GPUMemory *gmem)
{
    cutilSafeCall(cudaMemcpyToSymbol("s", sim, sizeof(Simulation)));
    cutilSafeCall(cudaMemcpyToSymbol("g", gmem, sizeof(GPUMemory)));
}

void init_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem)
{
    init_params_mem(conf, sim, gmem);
    init_results_mem(conf, sim, gmem);
    copy_mem_symbols(sim, gmem);
}

void free_gpu_results_mem_except_fluence(GPUMemory gmem)
{
    // Path length and momentum transfer.
    cutilSafeCall(cudaFree(gmem.path_length));
    cutilSafeCall(cudaFree(gmem.mom_transfer));
    cutilSafeCall(cudaFree(gmem.temp_path_length));
    cutilSafeCall(cudaFree(gmem.temp_mom_transfer));

    cutilSafeCall(cudaFree(gmem.det_hit));
}

void free_gpu_results_mem(GPUMemory gmem)
{
    // Path length and momentum transfer.
    cutilSafeCall(cudaFree(gmem.path_length));
    cutilSafeCall(cudaFree(gmem.mom_transfer));
    cutilSafeCall(cudaFree(gmem.temp_path_length));
    cutilSafeCall(cudaFree(gmem.temp_mom_transfer));

    // Photon fluence.
    cutilSafeCall(cudaFree(gmem.fbox));

    cutilSafeCall(cudaFree(gmem.det_hit));
}

void free_gpu_params_mem(GPUMemory gmem)
{
    // Tissue types.
    cutilSafeCall(cudaFree(gmem.media_type));

    // Detectors' locations and radii.
    cutilSafeCall(cudaFree(gmem.det_loc));

    // Optical properties of the different tissue types.
    cutilSafeCall(cudaFree(gmem.media_prop));
}

void free_cpu_results_mem(Simulation sim)
{
    // Path length and momentum transfer.
    free(sim.path_length);
    free(sim.mom_transfer);

    // Photon fluence.
    free(sim.fbox);

    free(sim.det.hit);
}

void free_cpu_params_mem(Simulation sim)
{
    // Tissue types.
    for(int i = 0; i < sim.grid.dim.x; i++) {
        for(int j = 0; j < sim.grid.dim.y; j++) {
            free(sim.grid.media_type[i][j]);
        }
        free(sim.grid.media_type[i]);
    }
    free(sim.grid.media_type);

    // Detectors' locations and radii.
    free(sim.det.info);

    // Optical properties of the different tissue types.
    free(sim.tiss.prop);
}

void free_mem(Simulation sim, GPUMemory gmem)
{
    free_gpu_params_mem(gmem); free_gpu_results_mem(gmem);
    free_cpu_params_mem(sim);  free_cpu_results_mem(sim);
}

void retrieve(Simulation *sim, GPUMemory *gmem)
{
    //size_t sizeof_tissueArrays = sim->n_photons * (sim->tiss.num + 1) * sizeof(float);
    size_t sizeof_tissueArrays = (1 << NUM_HASH_BITS) * sizeof(float);
    size_t sizeof_fbox = sim->grid.nIxyz * sim->num_time_steps * sizeof(float);
    size_t sizeof_det_hit = sim->n_photons * sizeof(int8_t);

    TO_HOST(sim->path_length, gmem->path_length, sizeof_tissueArrays);
    TO_HOST(sim->mom_transfer, gmem->mom_transfer, sizeof_tissueArrays);
    TO_HOST(sim->fbox, gmem->fbox, sizeof_fbox);
    TO_HOST(sim->det.hit, gmem->det_hit, sizeof_det_hit);
}
