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

template <class type>
void linearize_3d(type ***t, type *l, int dim_x, int dim_y, int dim_z)
{
    for (int x = 0; x < dim_x; x++)
        for (int y = 0; y < dim_y; y++)
            for (int z = 0; z < dim_z; z++)
                l[LIN3D(x,y,z,dim_x,dim_y)] = t[x][y][z];
}

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

    cudaMalloc((void **) &d_seed, sizeof_seed);
    TO_DEVICE(d_seed, h_seed, sizeof_seed);

    free(h_seed);

    return d_seed;
}

void init_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem)
{
    float *d_fbox;
    float *d_path_length, *d_mom_transfer;
    uint8_t *h_linear_media_type, *d_media_type;
    uint32_t *d_det_hit_matrix;
    uint32_t *d_seed;
    int4 *d_det_loc;
    float4 *d_media_prop;
    size_t num_tissueArrays, num_fbox;

    // Calculate the total number of voxel elements.
    int grid_dim = sim->grid.dim.x * sim->grid.dim.y * sim->grid.dim.z;

    // Linearize media_type, as CUDA cannot handle pointers to pointers.
    h_linear_media_type = (uint8_t *) malloc(grid_dim * sizeof(uint8_t));
    linearize_3d(sim->grid.media_type, h_linear_media_type,
                 sim->grid.dim.x, sim->grid.dim.y, sim->grid.dim.z);

    // Setup the path length and momentum transfer arrays.
    //num_tissueArrays = (sim->tiss.num + 1) * sim->n_photons;
    num_tissueArrays = 1 << NUM_HASH_BITS; // 128 MBs used by each array; must be a power of 2
    sim->path_length = (float *) calloc(num_tissueArrays, sizeof(float));
    sim->mom_transfer = (float *) calloc(num_tissueArrays, sizeof(float));

    // Photon fluence.
    num_fbox = sim->grid.nIxyz * sim->num_time_steps;
    sim->fbox = (float *) calloc(num_fbox, sizeof(float));

    // Bitset indicating which detectors (if any) were hit by which photons.
    sim->det_hit = bitset_new(sim->n_photons, sim->det.num);

    d_seed = init_rand_seed(conf.rand_seed, conf);

    // Allocate memory on the GPU global memory.
    // TODO: use constant memory where appropriate 
    cudaMalloc((void **) &d_det_loc, MAX_DETECTORS * sizeof(int4));
    cudaMalloc((void **) &d_media_prop, (MAX_TISSUES + 1) * sizeof(float4));
    cudaMalloc((void **) &d_media_type, grid_dim * sizeof(uint8_t));
    cudaMalloc((void **) &d_path_length, num_tissueArrays * sizeof(float));
    cudaMalloc((void **) &d_mom_transfer, num_tissueArrays * sizeof(float));
    cudaMalloc((void **) &d_fbox,      num_fbox           * sizeof(float));
    cudaMalloc((void **) &d_det_hit_matrix, bitset_size(sim->det_hit) * sizeof(uint32_t));

    int gpu_mem_spent = sizeof(int4) * MAX_DETECTORS
                      + sizeof(float4) * (MAX_TISSUES + 1)
                      + sizeof(uint8_t) * grid_dim
                      + sizeof(float) * num_tissueArrays
                      + sizeof(float) * num_tissueArrays
                      + sizeof(float) * num_fbox
                      + sizeof(uint32_t) * bitset_size(sim->det_hit)
                      + sizeof(uint32_t) * conf.n_threads * RAND_SEED_LEN;
    printf("memory spent = %dMB\n", gpu_mem_spent / (1024 * 1024));

    // Copy simulation memory to the GPU.
    //cudaMemcpyToSymbol("det_loc", sim->det.info, sim->det.num * sizeof(int4));
    //cudaMemcpyToSymbol("media_prop", sim->tiss.prop, (sim->tiss.num + 1) * sizeof(float4));
    cudaMemcpyToSymbol("s", sim, sizeof(Simulation));
    TO_DEVICE(d_det_loc, sim->det.info, MAX_DETECTORS * sizeof(int4));
    TO_DEVICE(d_media_prop, sim->tiss.prop, (MAX_TISSUES + 1) * sizeof(float4));
    TO_DEVICE(d_media_type, h_linear_media_type, grid_dim * sizeof(uint8_t));
    TO_DEVICE(d_path_length, sim->path_length, num_tissueArrays * sizeof(float));
    TO_DEVICE(d_mom_transfer, sim->mom_transfer, num_tissueArrays * sizeof(float));
    TO_DEVICE(d_fbox,      sim->fbox,      num_fbox           * sizeof(float));
    TO_DEVICE(d_det_hit_matrix, sim->det_hit.matrix, bitset_size(sim->det_hit) * sizeof(uint32_t));

    // Update GPU memory structure (so that its pointers can be used elsewhere).
    gmem->det_loc = d_det_loc;
    gmem->media_prop = d_media_prop;
    gmem->media_type = d_media_type;
    gmem->path_length = d_path_length;
    gmem->mom_transfer = d_mom_transfer;
    gmem->fbox = d_fbox;
    gmem->det_hit = sim->det_hit;
    gmem->det_hit.matrix = d_det_hit_matrix;
    gmem->seed = d_seed;
    cudaMemcpyToSymbol("g", gmem, sizeof(GPUMemory));

    // Free temporary memory used on the host.
    free(h_linear_media_type);
}

void free_gpu_results_mem(GPUMemory gmem)
{
    // Path length and momentum transfer.
    cudaFree(gmem.path_length);
    cudaFree(gmem.mom_transfer);

    // Photon fluence.
    cudaFree(gmem.fbox);

    // Bitset of the detectors which were hit by a given photon.
    cudaFree(gmem.det_hit.matrix);  // TODO: properly handle this 
}

void free_gpu_params_mem(GPUMemory gmem)
{
    // Tissue types.
    cudaFree(gmem.media_type);

    // Detectors' locations and radii.
    cudaFree(gmem.det_loc);

    // Optical properties of the different tissue types.
    cudaFree(gmem.media_prop);

    // Random number generation.
    cudaFree(gmem.seed);
}

void free_cpu_results_mem(Simulation sim)
{
    // Path length and momentum transfer.
    free(sim.path_length);
    free(sim.mom_transfer);

    // Photon fluence.
    free(sim.fbox);

    // Bitset of the detectors which were hit by a given photon.
    bitset_free(sim.det_hit);
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
    size_t sizeof_det_hit = bitset_size(sim->det_hit) * sizeof(uint32_t);

    TO_HOST(sim->path_length, gmem->path_length, sizeof_tissueArrays);
    TO_HOST(sim->mom_transfer, gmem->mom_transfer, sizeof_tissueArrays);
    TO_HOST(sim->fbox, gmem->fbox, sizeof_fbox);
    TO_HOST(sim->det_hit.matrix, gmem->det_hit.matrix, sizeof_det_hit);
}
